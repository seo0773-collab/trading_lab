"""Handler for heatmap1 — Volume Profile 신호 전략 (단일종목).

롤링 볼륨 프로파일에서 POC/VAH/VAL(밸류에어리어)을 산출하고(lookahead 없음),
가격이 그 레벨과 상호작용할 때 매매한다. 기본 모드는 ``va_reversion``
(밸류에어리어 경계 이탈 후 복귀 → POC를 향한 평균회귀). ``va_breakout``은 VA 밖
모멘텀 이탈을 추세추종한다.

데이터 소스 무관: asset_class='equity'면 yfinance, 'crypto'면 ccxt(fetch_ohlcv.py).
2D 히트맵은 forecast(1D) 계약 밖이라 forecast에는 poc/vah/val 1D 레벨만 싣는다.
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
from run_kalman_pipeline import load_yfinance  # noqa: E402
from strategy_execution import chronological_splits  # noqa: E402
from volume_profile import (  # noqa: E402
    build_heatmap,
    build_relative_heatmap,
    rolling_profile_levels,
)

_DASHBOARD_METRIC_KEYS = (
    "trades", "hit_rate", "total_return", "sharpe", "max_drawdown",
    "profit_factor", "expectancy",
)


class Heatmap1Handler:
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
                int(config.get("synthetic_bars", 1500)),
                int(config.get("seed", 7)),
                str(config.get("interval", "1d")),
            )
        if csv_path is not None:
            return _load_csv(csv_path)
        if str(config.get("asset_class", "equity")) == "crypto":
            return self._load_crypto(symbol, config)
        return load_yfinance(
            symbol, config.get("interval", "1d"), config.get("period", "max")
        )

    @staticmethod
    def _load_crypto(symbol: str, config: dict[str, Any]) -> pd.DataFrame:
        """ccxt 경로 — scripts/fetch_ohlcv.py 재사용(로컬에서만, 네트워크 필요)."""
        import ccxt  # lazy: 크립토 라이브 수집 시에만

        from fetch_ohlcv import fetch_ohlcv

        exchange = getattr(ccxt, str(config.get("exchange", "binance")))(
            {"enableRateLimit": True}
        )
        exchange.load_markets()
        since = pd.Timestamp(
            config.get("since", "2021-01-01"), tz="UTC"
        )
        df = fetch_ohlcv(
            exchange, symbol, str(config.get("interval", "1d")),
            int(since.timestamp() * 1000),
        )
        return df

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
        levels = self._levels(df, config)
        feats = self._features(df, levels)
        trades_all, position, bar_return = self._simulate(df, levels, config)

        split = chronological_splits(
            df.index,
            float(config.get("train_frac", 0.6)),
            float(config.get("validation_frac", 0.2)),
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
            "asset_class": str(config.get("asset_class", "equity")),
            "signal_mode": str(config.get("signal_mode", "va_reversion")),
            "va_pct": float(config.get("va_pct", 0.70)),
            "lookback": int(config.get("lookback", 120)),
            "long_only": bool(config.get("long_only", True)),
            "total_trades_all": int(len(trades_all)),
        }
        extras: dict[str, Any] = {}
        heat = self._heatmap_frame(df.loc[window], config)
        if heat is not None:
            extras["heatmap"] = heat
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
            extras=extras,
        )

    # ---------------------------------------------------------------- levels
    def _levels(self, df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        """롤링 프로파일 → poc/vah/val 레벨 시계열. 하위 전략이 override 가능."""
        return rolling_profile_levels(
            df,
            lookback=int(config.get("lookback", 120)),
            bins=int(config.get("profile_bins", 60)),
            va_pct=float(config.get("va_pct", 0.70)),
            cumulative=bool(config.get("cumulative", False)),
            scale=str(config.get("price_scale", "linear")),
        )

    # --------------------------------------------------------------- heatmap
    @staticmethod
    def _heatmap_frame(
        dfw: pd.DataFrame, config: dict[str, Any]
    ) -> pd.DataFrame | None:
        """가격×시간 2D 히트맵을 wide 프레임으로 — 대시보드 'heatmap' 패널용.

        forecast(1D) 계약 밖이라 extras로 싣는다. 가격행/시간열을 적당히
        다운샘플해 직렬화 크기를 통제한다(time 컬럼 + 가격 bin 컬럼).
        """
        if len(dfw) < 3:
            return None
        rows = int(config.get("heatmap_rows", 120))
        max_cols = int(config.get("heatmap_max_cols", 360))
        cumulative = bool(
            config.get("heatmap_cumulative", config.get("cumulative", False))
        )
        scale = str(config.get("price_scale", "linear"))
        if str(config.get("heatmap_axis", "relative")) == "relative":
            # 시점별 인과 상대위치 축: 과거 좁은 구간도 세로 해상도 100% (미래 미참조).
            H, edges, taxis = build_relative_heatmap(
                dfw, rows=rows, max_cols=max_cols, cumulative=cumulative,
                lookback=int(config.get("lookback", 120)), scale=scale,
            )
        else:  # 절대가격 전역격자(기존) — 시간 블록 합산으로 열 다운샘플
            H, edges, taxis = build_heatmap(
                dfw, rows=rows, cumulative=cumulative, scale=scale,
            )
            cols = H.shape[1]
            if cols > max_cols:  # 블록 합산(stride 선택 대신 — 볼륨 보존·밀도 유지)
                stride = (cols + max_cols - 1) // max_cols
                pad = (-cols) % stride
                if pad:
                    H = np.concatenate([H, np.zeros((H.shape[0], pad))], axis=1)
                H = H.reshape(H.shape[0], -1, stride).sum(axis=2)
                taxis = taxis[::stride][: H.shape[1]]
        centers = 0.5 * (edges[:-1] + edges[1:])
        frame = pd.DataFrame(H.T, columns=[f"{c:.4f}" for c in centers])
        frame.insert(0, "time", pd.to_datetime(taxis).astype(str))
        return frame

    # ------------------------------------------------------------- features
    @staticmethod
    def _features(df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in ("open", "high", "low", "close"):
            if col in df:
                out[col] = df[col].astype(float)
        out["poc"] = levels["poc"]
        out["vah"] = levels["vah"]
        out["val"] = levels["val"]
        return out

    # -------------------------------------------------------------- signals
    @staticmethod
    def _signals(
        df: pd.DataFrame, levels: pd.DataFrame, config: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """모드별 롱/숏 진입 신호 (lookahead 없음: t의 레벨은 t 이하 데이터 산출)."""
        mode = str(config.get("signal_mode", "va_reversion"))
        cl = df["close"].to_numpy(float)
        vah = levels["vah"].to_numpy(float)
        val = levels["val"].to_numpy(float)
        n = len(df)
        long_sig = np.zeros(n, dtype=bool)
        short_sig = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if np.isnan(val[i]) or np.isnan(val[i - 1]):
                continue
            if mode == "va_breakout":
                # VA 상단 상향 돌파 → 롱 / 하단 하향 이탈 → 숏
                long_sig[i] = cl[i - 1] <= vah[i - 1] and cl[i] > vah[i]
                short_sig[i] = cl[i - 1] >= val[i - 1] and cl[i] < val[i]
            else:  # va_reversion
                # VA 하단 아래로 갔다가 복귀 → 롱(POC 회귀) / 상단 위에서 복귀 → 숏
                long_sig[i] = cl[i - 1] < val[i - 1] and cl[i] >= val[i]
                short_sig[i] = cl[i - 1] > vah[i - 1] and cl[i] <= vah[i]
        return long_sig, short_sig

    @staticmethod
    def _targets(
        mode: str, d: int, poc: float, vah: float, val: float, buf: float
    ) -> tuple[float, float]:
        """진입 방향별 (take_profit, stop) 가격."""
        width = max(vah - val, 1e-9)
        if mode == "va_breakout":
            if d == 1:
                return vah + buf * width, poc  # 추세 연장 목표 / POC 되돌림 손절
            return val - buf * width, poc
        # va_reversion: POC 회귀 목표 / VA 경계 밖 손절
        if d == 1:
            return poc, val - buf * width
        return poc, vah + buf * width

    # ------------------------------------------------------------ execution
    def _simulate(
        self, df: pd.DataFrame, levels: pd.DataFrame, config: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        long_sig, short_sig = self._signals(df, levels, config)
        index = df.index
        op = df["open"].to_numpy(float)
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        cl = df["close"].to_numpy(float)
        poc = levels["poc"].to_numpy(float)
        vah = levels["vah"].to_numpy(float)
        val = levels["val"].to_numpy(float)
        n = len(df)

        mode = str(config.get("signal_mode", "va_reversion"))
        long_only = bool(config.get("long_only", True))
        allow_short = not long_only
        buf = float(config.get("stop_buffer_frac", 0.5))
        max_hold = int(config.get("max_hold_bars", 60))
        min_hold = int(config.get("min_hold_bars", 1))
        costs = config.get("costs", {})
        cost_frac = (
            float(costs.get("fee_bps_per_side", 5.0))
            + float(costs.get("slippage_bps", 5.0))
        ) / 1e4
        warmup = int(config.get("lookback", 120))

        position = np.zeros(n)
        cost_hits = np.zeros(n)
        trades: list[dict[str, Any]] = []

        i = warmup
        while i < n - 1:
            d = 1 if long_sig[i] else (-1 if (short_sig[i] and allow_short) else 0)
            if d == 0:
                i += 1
                continue
            entry_j = i + 1  # next_open 체결
            entry_price = op[entry_j]
            tp_price, stop_price = self._targets(
                mode, d, poc[i], vah[i], val[i], buf
            )
            term_j = term_price = None
            reason = ""
            j = entry_j
            while j < n:
                held = j - entry_j
                # (1) 손절 — 인트라바, 보수적으로 먼저 확인
                if (d == 1 and lo[j] <= stop_price) or (d == -1 and hi[j] >= stop_price):
                    term_j, term_price, reason = j, stop_price, "va_stop"
                    break
                # (2) 익절 — POC 도달(또는 추세 목표)
                if held >= min_hold and (
                    (d == 1 and hi[j] >= tp_price) or (d == -1 and lo[j] <= tp_price)
                ):
                    term_j, term_price, reason = j, tp_price, "poc_target"
                    break
                # (3) 시간청산
                if max_hold > 0 and held >= max_hold:
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price = nxt, op[nxt]
                    else:
                        term_j, term_price = j, cl[j]
                    reason = "horizon"
                    break
                # (4) 반대 신호 → 다음 봉 청산
                if held >= min_hold and (
                    (d == 1 and short_sig[j]) or (d == -1 and long_sig[j])
                ):
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price, reason = nxt, op[nxt], "opposite"
                    else:
                        term_j, term_price, reason = j, cl[j], "end_of_data"
                    break
                j += 1
            if term_j is None:
                term_j, term_price, reason = n - 1, cl[n - 1], "end_of_data"

            gross = d * (term_price / entry_price - 1.0)
            trades.append({
                "direction": d,
                "entry_time": index[entry_j],
                "entry_price": entry_price,
                "exit_time": index[term_j],
                "exit_price": term_price,
                "stop_loss_price": stop_price,
                "take_profit_price": tp_price,
                "net_return": gross - 2.0 * cost_frac,
                "exit_reason": reason,
                "entry_reason": self._entry_reason(d, mode, poc[i], vah[i], val[i]),
            })
            position[entry_j : term_j + 1] = d
            cost_hits[entry_j] += cost_frac
            cost_hits[term_j] += cost_frac
            i = term_j + 1

        ret = np.zeros(n)
        ret[1:] = cl[1:] / cl[:-1] - 1.0
        bar_return = position * ret - cost_hits

        trades_df = pd.DataFrame(trades, columns=[
            "direction", "entry_time", "entry_price", "exit_time", "exit_price",
            "stop_loss_price", "take_profit_price", "net_return", "exit_reason",
            "entry_reason",
        ])
        return (
            trades_df,
            pd.Series(position, index=index, name="position"),
            pd.Series(bar_return, index=index, name="bar_return"),
        )

    @staticmethod
    def _entry_reason(
        d: int, mode: str, poc: float, vah: float, val: float
    ) -> str:
        side = "롱" if d == 1 else "숏"
        if mode == "va_breakout":
            edge = "VAH 상향돌파" if d == 1 else "VAL 하향이탈"
        else:
            edge = "VAL 복귀(POC 회귀)" if d == 1 else "VAH 복귀(POC 회귀)"
        return f"{side} · {edge} · POC {poc:.2f} VA[{val:.2f},{vah:.2f}]"

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
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "expectancy": (
                float(trades["net_return"].mean()) if n_trades else None
            ),
            "phase": phase,
        }
