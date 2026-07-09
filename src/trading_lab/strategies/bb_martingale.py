"""Handler for bb-martingale-v1 — 단일종목 볼린저밴드 항상-롱숏 마틴게일 평균회귀.

리서치 계보: scripts/bb_crossing_sweep.py(최적 BB length=20/mult=2.5/offset=0 스윕)
→ scripts/bb_martingale_meanreversion.py(밴드이탈=플립 방식으로 재정의 + 물타기 사이징)
→ scripts/bb_martingale_regime_gate.py(ADX 추세가속 레짐게이트). 이 핸들러는 세 스크립트의
핵심 로직(bollinger_bands/compute_adx/simulate_martingale)을 그대로 재사용해 대시보드
파이프라인(단일종목, phase 슬라이스)에 연결한다.

손절 없이 마틴게일 최대단계로만 리스크를 캡하는 구조라 파산(자본<=0) 가능성이 남아있다
(reports/bb_martingale_sweep.md 실측: mult=3.0에서 파산율 25%+). live_eligible=False,
페이퍼 검증 전 보류.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.market_data import load_cumulative_yfinance
from trading_lab.paths import ROOT
from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bb_crossing_sweep import bollinger_bands, compute_adx  # noqa: E402
from bb_martingale_meanreversion import simulate_martingale  # noqa: E402
from profile_sizing.metrics import equity_metrics  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402


class BbMartingaleV1Handler:
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
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.columns = [c.lower() for c in df.columns]
            return df
        interval = str(config.get("interval", "1d"))
        period = str(config.get("period", "max"))
        return load_cumulative_yfinance(symbol, interval, period)

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        bb_cfg = config.get("bb") or {}
        length = int(bb_cfg.get("length", 20))
        mult = float(bb_cfg.get("mult", 2.5))
        offset = int(bb_cfg.get("offset", 0))

        mg_cfg = config.get("martingale") or {}
        add_step = float(mg_cfg.get("add_step", 0.03))
        mg_mult = float(mg_cfg.get("mult", 1.5))
        max_levels = int(mg_cfg.get("max_levels", 5))
        base_lot_fraction = float(mg_cfg.get("base_lot_fraction", 0.05))

        cost_cfg = config.get("costs") or {}
        fee_side = (float(cost_cfg.get("fee_bps_per_side", 20.103))
                    + float(cost_cfg.get("slippage_bps", 5.0))) / 1e4

        gate_cfg = config.get("regime_gate") or {}
        gate_enabled = bool(gate_cfg.get("enabled", True))
        adx_threshold = float(gate_cfg.get("adx_threshold", 25.0))
        adx_slope_lookback = int(gate_cfg.get("adx_slope_lookback", 10))

        train_frac = float(config.get("train_frac", 0.6))
        validation_frac = float(config.get("validation_frac", 0.2))

        df = raw.rename(columns=str.lower)
        index = self._naive_index(df)
        close = df["close"].copy()
        close.index = index

        basis, upper, lower = bollinger_bands(close, length, mult, offset)

        adx = None
        regime_ok = None
        if gate_enabled:
            hlc = df[["high", "low", "close"]].copy()
            hlc.index = index
            adx = compute_adx(hlc)
            bad = (adx > adx_threshold) & (adx > adx.shift(adx_slope_lookback))
            regime_ok = (~bad.fillna(False)).to_numpy()

        warmup = min(length + offset + 5, max(len(close) - 1, 0))
        trades_raw, equity, stats = simulate_martingale(
            close, upper, lower, trade_from=warmup,
            add_step=add_step, mult=mg_mult, max_levels=max_levels,
            regime_ok=regime_ok, base_lot_fraction=base_lot_fraction, fee_side=fee_side,
        )

        forecast = pd.DataFrame(
            {"close": close, "basis": basis, "upper": upper, "lower": lower}, index=index)
        if adx is not None:
            forecast["adx"] = adx
            forecast["regime_ok"] = pd.Series(regime_ok, index=index).astype(int)

        trades = self._format_trades(trades_raw)

        window = self._phase_window(index, phase, train_frac, validation_frac)
        forecast_w = forecast.loc[window]
        equity_w = self._rebase(equity, window)
        benchmark_w = self._rebase(close, window)
        trades_w = self._slice_trades(trades, window)

        metrics = self._metrics(equity_w, trades_w, stats, phase)
        metadata = {
            "n_bars": int(len(window)),
            "timeframe": str(config.get("interval", "1d")),
            "ruined": bool(stats["ruined"]),
            "max_exposure": float(stats["max_exposure"]),
            "regime_gate_enabled": gate_enabled,
        }
        return StrategyArtifacts(
            forecast=forecast_w, trades=trades_w, equity=equity_w, metrics=metrics,
            metadata=metadata, horizon=0, benchmark=benchmark_w,
        )

    @staticmethod
    def _naive_index(raw: pd.DataFrame) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
        return idx.tz_localize(None) if idx.tz is not None else idx

    @staticmethod
    def _phase_window(index: pd.DatetimeIndex, phase: str,
                       train_frac: float, validation_frac: float) -> pd.DatetimeIndex:
        if phase == "all":
            return index
        n = len(index)
        t_end = int(n * train_frac)
        v_end = int(n * (train_frac + validation_frac))
        if phase == "validation":
            return index[t_end:v_end]
        if phase == "test":
            return index[v_end:]
        return index

    @staticmethod
    def _rebase(series: pd.Series, window: pd.DatetimeIndex) -> pd.Series:
        seg = series.reindex(window)
        if len(seg) == 0:
            return seg.rename("equity")
        base = seg.iloc[0]
        if not np.isfinite(base) or base <= 0:
            return seg.fillna(0.0).rename("equity")
        return (seg / base).rename("equity")

    @staticmethod
    def _format_trades(trades_raw: pd.DataFrame) -> pd.DataFrame:
        columns = ["direction", "entry_time", "entry_price", "exit_time", "exit_price",
                   "net_return", "exit_reason", "levels_used", "total_size"]
        if trades_raw.empty:
            return pd.DataFrame(columns=columns)
        out = trades_raw.rename(columns={"avg_entry_price": "entry_price"}).copy()
        return out[columns]

    @staticmethod
    def _slice_trades(trades: pd.DataFrame, window: pd.DatetimeIndex) -> pd.DataFrame:
        if trades.empty or len(window) == 0:
            return trades
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        mask = np.asarray((entries >= window[0]) & (entries <= window[-1]))
        return trades[mask].reset_index(drop=True)

    @staticmethod
    def _metrics(equity: pd.Series, trades: pd.DataFrame, stats: dict, phase: str) -> dict:
        perf = equity_metrics(equity, ann=252.0)
        n = int(len(trades))
        hit = float((trades["net_return"] > 0).mean()) if n else None
        return {
            "trades": n,
            "hit_rate": hit,
            "total_return": perf["total_return"],
            "sharpe": perf["sharpe"],
            "max_drawdown": perf["max_drawdown"],
            "cagr": perf["cagr"],
            "volatility": perf["volatility"],
            "ruined": bool(stats["ruined"]),
            "max_exposure": float(stats["max_exposure"]),
            "phase": phase,
        }
