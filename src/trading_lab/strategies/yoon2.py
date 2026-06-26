"""Handler for the yoon2 strategy — Kalman-filtered MACD timing (single symbol).

macd_raw.txt(Pine indicator)의 *칼만 처리된 라인만* 사용한다. raw MACD/Signal/
Hist는 쓰지 않는다. 진입 트리거는 ``kalHistDelta``(= 칼만 히스토그램의 1차 변화,
모멘텀의 가속/감속) 부호 전환이며, 롱·숏 양방향 stop-and-reverse로 운용한다.

칼만 평활이 이 전략의 생명줄이다: 델타 전환은 가장 선행하지만 가장 시끄러운
신호라, ``kalman_q/r`` + ``confirm_bars`` + ``min_hist_gap_atr`` + 0선 필터로
잔물결(whipsaw)을 통제한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.paths import ROOT
from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from di_kalman_mw.run import make_synthetic_ohlcv  # noqa: E402
from di_kalman_mw.run import load_data as _load_csv  # noqa: E402
from indicators.kalman import kalman_1d  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from strategy_execution import chronological_splits  # noqa: E402

_DASHBOARD_METRIC_KEYS = (
    "trades", "hit_rate", "total_return", "sharpe", "max_drawdown",
    "profit_factor", "expectancy",
)


class Yoon2Handler:
    # ------------------------------------------------------------------ data
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        if synthetic:
            return make_synthetic_ohlcv(
                int(config.get("synthetic_bars", 9000)),
                int(config.get("seed", 7)),
                str(config.get("interval", "1d")),
            )
        if csv_path is not None:
            return _load_csv(csv_path)
        return load_yfinance(
            symbol, config.get("interval", "1d"), config.get("period", "max")
        )

    # ------------------------------------------------------------- artifacts
    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        df = raw.dropna(subset=["open", "high", "low", "close"]).sort_index()
        feats = self._indicators(df, config)

        split = chronological_splits(
            df.index,
            float(config.get("train_frac", 0.6)),
            float(config.get("validation_frac", 0.2)),
        )
        # 익절 사다리 분위수는 in-sample(identification) 구간 분포에서만
        # 추정해 validation/test에 적용한다(룩어헤드 방지).
        tp_mults, tp_fracs = self._tp_multiples(
            feats, split == "identification", config
        )
        trades_all, position, bar_return = self._simulate(
            df, feats, config, tp_mults=tp_mults, tp_fracs=tp_fracs
        )

        window = df.index if phase == "all" else df.index[split == phase]

        forecast = feats.loc[window]
        equity = (1.0 + bar_return.loc[window]).cumprod().rename("equity")
        trades = self._phase_trades(trades_all, window)
        metrics = self._metrics(
            trades, bar_return.loc[window], equity, bars_per_year, phase
        )
        metadata = {
            "timeframe": str(config.get("interval", "1d")),
            "entry_trigger": str(config.get("entry_trigger", "delta_turn")),
            "trigger": "kal_hist_delta_turn",
            "direction": str(config.get("direction", "both")),
            "kalman_base": str(config.get("kalman_base", "MACD Line")),
            "total_trades_all": int(len(trades_all)),
        }
        if tp_mults:
            metadata["tp_atr_mults"] = [round(float(m), 3) for m in tp_mults]
            metadata["tp_quantiles"] = [
                float(q) for q in config.get("tp_quantiles", [0.33, 0.66, 0.90])
            ]
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
        )

    # ------------------------------------------------------------ indicators
    @staticmethod
    def _indicators(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        fast = int(config.get("fast_len", 12))
        slow = int(config.get("slow_len", 26))
        signal = int(config.get("signal_len", 9))
        q = float(config.get("kalman_q", 0.01))
        r = float(config.get("kalman_r", 0.10))
        base = str(config.get("kalman_base", "MACD Line"))

        close = df["close"].astype(float)
        fast_ema = close.ewm(span=fast, adjust=False).mean().rename("fast_ema")
        slow_ema = close.ewm(span=slow, adjust=False).mean().rename("slow_ema")
        macd_line = (fast_ema - slow_ema).rename("macd_line")

        if base == "Fast/Slow EMA":
            kal_macd = kalman_1d(fast_ema, q, r) - kalman_1d(slow_ema, q, r)
        else:  # "MACD Line": filter the MACD line directly (Pine default)
            kal_macd = kalman_1d(macd_line, q, r)
        kal_macd = kal_macd.rename("kal_macd")

        kal_signal_base = kal_macd.ewm(span=signal, adjust=False).mean()
        kal_signal = kalman_1d(kal_signal_base, q, r).rename("kal_signal")
        kal_hist = (kal_macd - kal_signal).rename("kal_hist")
        kal_hist_delta = kal_hist.diff().rename("kal_hist_delta")

        out = pd.DataFrame(index=df.index)
        for col in ("open", "high", "low", "close"):
            if col in df:
                out[col] = df[col].astype(float)
        out["atr"] = Yoon2Handler._atr(df, int(config.get("atr_len", 14)))
        out["kal_macd"] = kal_macd
        out["kal_signal"] = kal_signal
        out["kal_hist"] = kal_hist
        out["kal_hist_delta"] = kal_hist_delta
        return out

    @staticmethod
    def _atr(df: pd.DataFrame, length: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        # Wilder RMA
        return tr.ewm(alpha=1.0 / length, adjust=False).mean()

    # -------------------------------------------------------------- signals
    @staticmethod
    def _signals(
        feats: pd.DataFrame, config: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confirmed + filtered long/short entry signals.

        ``entry_trigger``로 트리거 계열을 고른다(부호 전환 판정 로직은 공통):
        - ``delta_turn``(기본): ``kal_hist_delta``(히스토그램 1차 변화)의 부호 전환
          = 모멘텀 가속/감속 반전. 가장 선행·가장 시끄러움.
        - ``cross``: ``kal_hist``(=``kal_macd − kal_signal``)의 0교차 = **라인 크로싱**
          (kal_macd가 kal_signal을 상향/하향 돌파). 덜 선행·덜 시끄러움.
        """
        confirm = max(1, int(config.get("confirm_bars", 2)))
        gap_atr = float(config.get("min_hist_gap_atr", 0.0))
        zero_filter = bool(config.get("macd_zero_filter", False))
        trigger = str(config.get("entry_trigger", "delta_turn"))

        hist = feats["kal_hist"].to_numpy(float)
        macd = feats["kal_macd"].to_numpy(float)
        atr = feats["atr"].to_numpy(float)
        # cross = 히스토그램 0교차(라인 크로싱), delta_turn = 히스토 델타 전환
        series = hist if trigger == "cross" else feats["kal_hist_delta"].to_numpy(float)
        n = len(series)

        long_sig = np.zeros(n, dtype=bool)
        short_sig = np.zeros(n, dtype=bool)
        for i in range(confirm, n):
            window = series[i - confirm + 1 : i + 1]
            prior = series[i - confirm]
            if np.isnan(prior) or np.isnan(window).any():
                continue
            gap_ok = True
            if gap_atr > 0.0:
                gap_ok = abs(hist[i]) >= gap_atr * atr[i]
            # 트리거 계열이 confirm봉 연속 양(+)이고 직전이 ≤0 → 롱
            if (window > 0).all() and prior <= 0 and gap_ok:
                if not zero_filter or macd[i] > 0:
                    long_sig[i] = True
            # 트리거 계열이 confirm봉 연속 음(−)이고 직전이 ≥0 → 숏
            elif (window < 0).all() and prior >= 0 and gap_ok:
                if not zero_filter or macd[i] < 0:
                    short_sig[i] = True
        return long_sig, short_sig

    # ----------------------------------------------------------- take-profit
    @staticmethod
    def _tp_multiples(
        feats: pd.DataFrame, train_mask: np.ndarray, config: dict[str, Any]
    ) -> tuple[list[float] | None, list[float] | None]:
        """train 구간 ``|kal_hist_delta| / ATR`` 분포 → ATR 배수 익절 사다리.

        히스토그램 델타는 가격(달러) 단위라 ATR로 나누면 *변동성 대비 봉당
        모멘텀 가속* 비율(무차원)이 된다. 그 분위수에 ``atr_scale``을 곱해 봉당
        가속을 다봉 익절 거리(ATR 배수)로 환산한다. 분포는 train에서만 추정해
        validation/test에 그대로 적용한다(룩어헤드 없음).
        """
        if not bool(config.get("tp_enabled", False)):
            return None, None
        quantiles = [float(q) for q in config.get("tp_quantiles", [0.33, 0.66, 0.90])]
        fractions = [
            float(f) for f in config.get("tp_fractions", [1 / 3, 1 / 3, 1 / 3])
        ]
        scale = float(config.get("tp_atr_scale", 30.0))
        min_samples = int(config.get("tp_min_samples", 30))

        ratio = (feats["kal_hist_delta"].abs() / feats["atr"]).to_numpy(float)
        ratio = ratio[np.asarray(train_mask, dtype=bool)]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if ratio.size < min_samples:
            return None, None
        mults = sorted(scale * float(np.quantile(ratio, q)) for q in quantiles)
        return mults, fractions[: len(mults)]

    # ------------------------------------------------------------ execution
    def _simulate(
        self,
        df: pd.DataFrame,
        feats: pd.DataFrame,
        config: dict[str, Any],
        *,
        tp_mults: list[float] | None = None,
        tp_fracs: list[float] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        long_sig, short_sig = self._signals(feats, config)
        index = df.index
        op = df["open"].to_numpy(float)
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        cl = df["close"].to_numpy(float)
        atr = feats["atr"].to_numpy(float)
        n = len(df)

        atr_mult = float(config.get("atr_stop_mult", 0.0))  # 0 = 손절 off
        max_hold = int(config.get("max_hold_bars", 0))  # 0 = 시간청산 off
        costs = config.get("costs", {})
        cost_frac = (
            float(costs.get("fee_bps_per_side", 5.0))
            + float(costs.get("slippage_bps", 5.0))
        ) / 1e4

        direction = str(config.get("direction", "both"))  # both | long | short
        allow_long = direction in ("both", "long")
        allow_short = direction in ("both", "short")
        use_tp = bool(tp_mults)
        tp_mults = tp_mults or []
        tp_fracs = tp_fracs or []

        warmup = int(config.get("slow_len", 26)) + int(config.get("signal_len", 9))
        position = np.zeros(n)  # 보유 익스포저(바별, 부분익절로 분수 가능)
        cost_hits = np.zeros(n)  # 바별 차감 비용(진입 1회 + 부분/전량 청산)
        trades: list[dict[str, Any]] = []

        i = warmup
        while i < n - 1:
            d = (
                1 if (long_sig[i] and allow_long)
                else (-1 if (short_sig[i] and allow_short) else 0)
            )
            if d == 0:
                i += 1
                continue

            entry_j = i + 1  # next_open 체결
            entry_price = op[entry_j]
            atr_e = atr[i]
            stop = (
                entry_price - d * atr_mult * atr_e if atr_mult > 0.0 else np.nan
            )
            # 익절 사다리 가격(거리 오름차순, 진입 방향으로 +/-)
            tp_prices = [entry_price + d * m * atr_e for m in tp_mults]
            tp_done = [False] * len(tp_prices)

            remaining = 1.0
            partials: list[tuple[int, float, float, float]] = []  # j, price, tp, frac
            reverse = False
            opp_sig_bar = None
            term_j = term_price = None
            term_reason = ""
            full_tp_bar = None

            j = entry_j
            while j < n:
                # (1) 손절(인트라바) — 잔량 전량 청산. 익절보다 먼저 보수적 확인.
                if atr_mult > 0.0 and remaining > 1e-9 and (
                    (d == 1 and lo[j] <= stop) or (d == -1 and hi[j] >= stop)
                ):
                    term_j, term_price, term_reason = j, stop, "stop_loss"
                    break
                # (2) 익절 사다리 — 같은 봉에서 여러 단계 동시 체결 허용.
                if use_tp and remaining > 1e-9:
                    for t in range(len(tp_prices)):
                        if tp_done[t]:
                            continue
                        if (d == 1 and hi[j] >= tp_prices[t]) or (
                            d == -1 and lo[j] <= tp_prices[t]
                        ):
                            tp_done[t] = True
                            frac = min(tp_fracs[t], remaining)
                            remaining -= frac
                            partials.append((j, tp_prices[t], tp_prices[t], frac))
                    if remaining <= 1e-9:
                        full_tp_bar = j  # TP로 전량 청산
                        break
                # (3) 시간청산 — 잔량 전량.
                if max_hold > 0 and (j - entry_j) >= max_hold:
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price = nxt, op[nxt]
                    else:
                        term_j, term_price = j, cl[j]
                    term_reason = "horizon"
                    break
                # (4) 반대 신호 → 다음 봉 잔량 청산. 양방향이면 리버스.
                if (d == 1 and short_sig[j]) or (d == -1 and long_sig[j]):
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price, term_reason = nxt, op[nxt], "opposite"
                        reverse = direction == "both"
                        opp_sig_bar = j
                    else:
                        term_j, term_price, term_reason = j, cl[j], "end_of_data"
                    break
                j += 1

            if full_tp_bar is None and term_j is None:  # 데이터 끝까지 잔량 보유
                term_j, term_price, term_reason = n - 1, cl[n - 1], "end_of_data"

            entry_reason = self._entry_reason(d, feats, i, config)
            # 부분 익절 거래행
            for pj, pprice, tp, frac in partials:
                gross = d * (pprice / entry_price - 1.0)
                trades.append(self._trade_row(
                    d, index, entry_j, entry_price, stop, tp, pj, pprice,
                    "take_profit", frac, gross - 2.0 * cost_frac, entry_reason,
                ))
                cost_hits[pj] += cost_frac * frac
            # 잔량(터미널) 거래행
            if remaining > 1e-9 and term_j is not None:
                gross = d * (term_price / entry_price - 1.0)
                trades.append(self._trade_row(
                    d, index, entry_j, entry_price, stop, np.nan, term_j,
                    term_price, term_reason, remaining,
                    gross - 2.0 * cost_frac, entry_reason,
                ))
                cost_hits[term_j] += cost_frac * remaining
            cost_hits[entry_j] += cost_frac  # 진입 비용(전량 1회)

            # 바별 익스포저: 트랜치별 보유구간 합산(분수). 다음 트레이드가 경계
            # 바를 덮어쓰도록 구간 대입(원래 stop-and-reverse 의미 유지).
            last_bar = full_tp_bar if full_tp_bar is not None else term_j
            hold = np.zeros(last_bar - entry_j + 1)
            for pj, _pprice, _tp, frac in partials:
                hold[: (pj - entry_j) + 1] += frac
            if remaining > 1e-9 and term_j is not None:
                hold[: (term_j - entry_j) + 1] += remaining
            position[entry_j : last_bar + 1] = d * hold

            # 다음 탐색 위치
            if reverse:
                i = opp_sig_bar  # 반대 신호 봉으로 되돌아가 즉시 반대 진입
            else:
                i = last_bar + 1

        # 바별 전략 수익(부분익절 분수 반영) − 진입/청산 비용
        ret = np.zeros(n)
        ret[1:] = cl[1:] / cl[:-1] - 1.0
        bar_return = position * ret - cost_hits

        trades_df = pd.DataFrame(trades, columns=[
            "direction", "entry_time", "entry_price", "exit_time",
            "exit_price", "stop_loss_price", "take_profit_price",
            "net_return", "exit_reason", "entry_reason", "size_frac",
        ])
        return (
            trades_df,
            pd.Series(position, index=index, name="position"),
            pd.Series(bar_return, index=index, name="bar_return"),
        )

    @staticmethod
    def _trade_row(
        d: int, index: pd.Index, entry_j: int, entry_price: float,
        stop: float, tp_price: float, exit_j: int, exit_price: float,
        reason: str, frac: float, net_return: float, entry_reason: str,
    ) -> dict[str, Any]:
        return {
            "direction": d,
            "entry_time": index[entry_j],
            "entry_price": entry_price,
            "exit_time": index[exit_j],
            "exit_price": exit_price,
            "stop_loss_price": stop,
            "take_profit_price": tp_price,
            "net_return": net_return,
            "exit_reason": reason,
            "entry_reason": entry_reason,
            "size_frac": float(frac),
        }

    @staticmethod
    def _entry_reason(
        d: int, feats: pd.DataFrame, i: int, config: dict[str, Any]
    ) -> str:
        side = "롱" if d == 1 else "숏"
        macd = float(feats["kal_macd"].iloc[i])
        if str(config.get("entry_trigger", "delta_turn")) == "cross":
            trig = "kalMACD↑Signal 상향돌파" if d == 1 else "kalMACD↓Signal 하향돌파"
        else:
            turn = "바닥 반등" if d == 1 else "꼭지 둔화"
            trig = f"히스토그램 {turn}(델타 전환)"
        return f"{side} · {trig} · kalMACD {macd:+.3f}"

    # --------------------------------------------------------------- phase
    @staticmethod
    def _phase_trades(
        trades_all: pd.DataFrame, window: pd.Index
    ) -> pd.DataFrame:
        if trades_all.empty:
            return trades_all
        mask = trades_all["entry_time"].isin(window)
        return trades_all[mask].reset_index(drop=True)

    # ------------------------------------------------------------- metrics
    @staticmethod
    def _metrics(
        trades: pd.DataFrame,
        bar_return: pd.Series,
        equity: pd.Series,
        bars_per_year: int,
        phase: str,
    ) -> dict[str, Any]:
        n_trades = int(len(trades))
        wins = trades[trades["net_return"] > 0] if n_trades else trades
        gains = trades.loc[trades["net_return"] > 0, "net_return"].sum()
        losses = -trades.loc[trades["net_return"] < 0, "net_return"].sum()

        std = float(bar_return.std())
        sharpe = (
            float(bar_return.mean()) / std * np.sqrt(bars_per_year)
            if std > 0 else None
        )
        if equity.empty:
            total_return = 0.0
            max_dd = None
        else:
            total_return = float(equity.iloc[-1]) - 1.0
            peak = equity.cummax()
            max_dd = float((equity / peak - 1.0).min())
        return {
            "trades": n_trades,
            "hit_rate": (len(wins) / n_trades) if n_trades else None,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "profit_factor": (
                float(gains / losses) if losses > 0 else None
            ),
            "expectancy": (
                float(trades["net_return"].mean()) if n_trades else None
            ),
            "phase": phase,
        }
