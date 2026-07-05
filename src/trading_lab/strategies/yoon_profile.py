"""Handler for yoon_profile — 멀티플 × 기울기 2D 상태 가중치 사이징 (단일종목).

SMA200 멀티플(가격 위치) × SMA200 기울기 z-각도(레짐)의 expanding percentile
3×3 그리드 상태별 가중치로 long-only 비중을 정한다. 셀 가중치는
reports/profile_sizing/yoon_profile_study.json 검증 결과에 근거한다.

데이터 소스: asset_class='crypto'(기본)면 data/raw/<심볼>_<TF>.parquet 로컬
캐시를 우선 사용하고 없으면 ccxt로 수집, 'equity'면 yfinance.
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

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.cycle_state import zigzag_moves  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.run import rebased_equity, slice_window  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402
from profile_sizing.yoon_profile import run_yoon_profile_pipeline  # noqa: E402


class YoonProfileHandler:
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
                int(config.get("synthetic_bars", 3000)),
                int(config.get("seed", 7)),
                str(config.get("interval", "1d")),
            )
        if csv_path is not None:
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.columns = [c.lower() for c in df.columns]
            return df
        if str(config.get("asset_class", "crypto")) == "crypto":
            return self._load_crypto(symbol, config)
        from run_kalman_pipeline import load_yfinance  # noqa: E402
        return load_yfinance(
            symbol, config.get("interval", "1d"), config.get("period", "max")
        )

    @staticmethod
    def _load_crypto(symbol: str, config: dict[str, Any]) -> pd.DataFrame:
        """로컬 parquet(data/raw) 우선, 없으면 ccxt 수집(네트워크 필요)."""
        interval = str(config.get("interval", "1d"))
        pair = YoonProfileHandler._crypto_pair(symbol)
        local = ROOT / "data" / "raw" / f"{pair.replace('/', '')}_{interval}.parquet"
        if local.exists():
            return pd.read_parquet(local)

        import ccxt  # lazy: 로컬 캐시가 없을 때만

        from fetch_ohlcv import fetch_ohlcv
        from run_kalman_pipeline import load_yfinance  # noqa: E402

        exchange = getattr(ccxt, str(config.get("exchange", "binance")))(
            {"enableRateLimit": True}
        )
        exchange.load_markets()
        since = pd.Timestamp(config.get("since", "2017-08-17"), tz="UTC")
        try:
            return fetch_ohlcv(
                exchange, pair, interval, int(since.timestamp() * 1000)
            )
        except Exception:
            # yoon_profile 계열은 crypto 기본 config를 공유한다. UI에서 SPY/QQQ 같은
            # 주식 심볼을 고르면 Binance에는 없으므로 yfinance로 폴백한다.
            return load_yfinance(
                symbol, interval, str(config.get("period", "max"))
            )

    @staticmethod
    def _crypto_pair(symbol: str) -> str:
        s = str(symbol).strip().upper()
        if "/" in s:
            return s
        if s.endswith("-USD"):
            return f"{s[:-4]}/USDT"
        if s.endswith("USDT"):
            return f"{s[:-4]}/USDT"
        return f"{s}/USDT"

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        cfg = config_from_dict(config)
        out = run_yoon_profile_pipeline(raw, config)

        index = self._naive_index(raw)
        window = slice_window(index, phase, cfg)

        forecast = out["forecast"].copy()
        forecast.index = index
        forecast = forecast.loc[window]

        port_ret = out["port_ret"].copy()
        port_ret.index = index
        equity = rebased_equity(port_ret, window)
        port_ret = port_ret.reindex(window).fillna(0.0)
        bnh_ret = out["buy_hold"].pct_change().fillna(0.0)
        bnh_ret.index = index
        benchmark = rebased_equity(bnh_ret, window)
        bnh_ret = bnh_ret.reindex(window).fillna(0.0)
        perf = performance(equity, port_ret, cfg.interval)
        bnh_perf = performance(benchmark, bnh_ret, cfg.interval)
        trades = self._slice_trades(out["trades"], window)

        metrics = self._metrics(perf, bnh_perf, trades, phase)
        state = forecast[["mult_bin", "angle_bin"]]
        valid = (state["mult_bin"] >= 0) & (state["angle_bin"] >= 0)
        metadata = {
            "n_bars": int(len(window)),
            "timeframe": cfg.interval,
            "cycle": f"SMA{int(config.get('cycle', {}).get('length', 200))}",
            "avg_exposure": float(forecast["actual_weight"].mean())
            if len(forecast) else 0.0,
            "state_coverage": float(valid.mean()) if len(state) else 0.0,
        }
        extras = {
            "state_occupancy": self._state_occupancy(forecast),
            "perf_vs_bnh": self._perf_table(perf, bnh_perf),
        }
        field_points = self._state_field_points(forecast, config)
        if not field_points.empty:
            extras["state_field"] = field_points
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
            benchmark=benchmark,
            extras=extras,
        )

    @staticmethod
    def _naive_index(raw: pd.DataFrame) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
        return idx.tz_localize(None) if idx.tz is not None else idx

    @staticmethod
    def _slice_trades(trades: pd.DataFrame, window) -> pd.DataFrame:
        if trades.empty or len(window) == 0:
            return trades
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        if entries.tz is not None:
            entries = entries.tz_localize(None)
            trades = trades.copy()
            trades["entry_time"] = entries
            exits = pd.DatetimeIndex(
                pd.to_datetime(trades["exit_time"])
            ).tz_localize(None)
            trades["exit_time"] = exits
        mask = np.asarray((entries >= window[0]) & (entries <= window[-1]))
        return trades[mask].reset_index(drop=True)

    @staticmethod
    def _metrics(perf: dict, bnh: dict, trades: pd.DataFrame, phase: str) -> dict:
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)
        total = perf.get("total_return")
        bnh_ret = bnh.get("total_return")
        return {
            "trades": n,
            "hit_rate": float((rets > 0).mean()) if n else None,
            "total_return": total,
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bnh_ret,
            "buy_hold_sharpe": bnh.get("sharpe"),
            "buy_hold_max_drawdown": bnh.get("max_drawdown"),
            "excess_return_vs_bnh": (
                total - bnh_ret
                if total is not None and bnh_ret is not None else None
            ),
            "phase": phase,
        }

    @staticmethod
    def _state_field_points(
        forecast: pd.DataFrame, config: dict[str, Any]
    ) -> pd.DataFrame:
        """연속 2D 상태 필드 점 프레임 (``field_dots`` 패널용).

        x/y = expanding percentile rank(angle/mult)를 격자 좌표(-0.5..2.5)로
        사상하되, 봉의 실제 bin 셀 안에 머물도록 클립한다(rank와 quantile 경계의
        미세 불일치로 점이 이웃 셀에 찍히는 것을 방지). move_dir/move_id =
        종가 zigzag로 라벨한 큰 상승/하락 leg — 대시보드가 초록/빨강 선으로 잇는다.

        임계값 노브용으로 ``knob_thresholds`` 그리드(기본 10~50%) 각각의 라벨을
        ``move_dir_<pct>``/``move_id_<pct>`` 컬럼으로 함께 담는다 — 패널이 렌더
        시점에 재계산 없이 임계값을 골라 궤적을 다시 그릴 수 있다. 평문
        move_dir/move_id는 config 기본 임계값(``move_threshold``)의 라벨이다.
        """
        need = {"mult_pct", "angle_pct", "mult_bin", "angle_bin",
                "close", "target_weight"}
        if not need.issubset(forecast.columns):
            return pd.DataFrame()
        valid = forecast[
            (forecast["mult_bin"] >= 0) & (forecast["angle_bin"] >= 0)
            & np.isfinite(forecast["mult_pct"])
            & np.isfinite(forecast["angle_pct"])
        ]
        if valid.empty:
            return pd.DataFrame()

        grid = int((config.get("cycle") or {}).get("grid_bins", 3))
        field_cfg = config.get("state_field") or {}
        threshold = float(field_cfg.get("move_threshold", 0.3))
        reversal = field_cfg.get("reversal")   # 기본: threshold/2 (zigzag_moves)
        knob = field_cfg.get("knob_thresholds") or (
            0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5
        )

        def coord(pct: pd.Series, bins: pd.Series) -> np.ndarray:
            raw = pct.to_numpy(dtype=float) * grid - 0.5
            b = bins.to_numpy(dtype=float)
            return np.clip(raw, b - 0.48, b + 0.48).round(4)

        mb = valid["mult_bin"].astype(int).to_numpy()
        ab = valid["angle_bin"].astype(int).to_numpy()
        out = pd.DataFrame({
            "time": pd.DatetimeIndex(valid.index).strftime("%Y-%m-%d %H:%M:%S"),
            "cell": [f"M{m}A{a}" for m, a in zip(mb, ab)],
            "x": coord(valid["angle_pct"], valid["angle_bin"]),
            "y": coord(valid["mult_pct"], valid["mult_bin"]),
            "target_weight": valid["target_weight"].astype(float).round(3),
        })
        default_pct = int(round(threshold * 100))
        for th_pct in sorted({int(round(float(t) * 100)) for t in knob}
                             | {default_pct}):
            # 노브 그리드는 기본 규칙(reversal=threshold/2), config의 reversal
            # 오버라이드는 기본 임계값에만 적용한다.
            rev = (
                float(reversal)
                if reversal is not None and th_pct == default_pct else None
            )
            moves = zigzag_moves(
                valid["close"], threshold=th_pct / 100.0, reversal=rev
            )
            out[f"move_dir_{th_pct}"] = moves["move_dir"].to_numpy()
            out[f"move_id_{th_pct}"] = moves["move_id"].to_numpy()
        out["move_dir"] = out[f"move_dir_{default_pct}"]
        out["move_id"] = out[f"move_id_{default_pct}"]
        return out

    @staticmethod
    def _state_occupancy(forecast: pd.DataFrame) -> pd.DataFrame:
        """phase 구간의 상태 점유율/평균 비중 요약 표."""
        valid = forecast[(forecast["mult_bin"] >= 0) & (forecast["angle_bin"] >= 0)]
        if valid.empty:
            return pd.DataFrame()
        grp = valid.groupby(["mult_bin", "angle_bin"]).agg(
            bars=("actual_weight", "size"),
            avg_weight=("actual_weight", "mean"),
        ).reset_index()
        grp["share"] = (grp["bars"] / len(valid)).round(3)
        grp["avg_weight"] = grp["avg_weight"].round(3)
        return grp

    @staticmethod
    def _perf_table(perf: dict, bnh: dict) -> pd.DataFrame:
        def pct(v):
            return None if v is None else round(float(v) * 100.0, 2)

        return pd.DataFrame([
            {"metric": "총수익률 %", "strategy": pct(perf.get("total_return")),
             "buy_hold": pct(bnh.get("total_return"))},
            {"metric": "CAGR %", "strategy": pct(perf.get("cagr")),
             "buy_hold": pct(bnh.get("cagr"))},
            {"metric": "MDD %", "strategy": pct(perf.get("max_drawdown")),
             "buy_hold": pct(bnh.get("max_drawdown"))},
            {"metric": "Sharpe",
             "strategy": None if perf.get("sharpe") is None else round(perf["sharpe"], 3),
             "buy_hold": None if bnh.get("sharpe") is None else round(bnh["sharpe"], 3)},
        ])
