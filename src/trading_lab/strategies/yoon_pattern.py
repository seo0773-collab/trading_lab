"""yoon_pattern — L틱 패턴 투영(유사 패턴 앙상블) 기대값 사이징 단일종목 전략.

직전 L틱(기본 300 일봉) 패턴과 가장 유사한 과거 구간 top-M의 '직후 H틱' 실현수익
중앙값을 투영 점수로 쓰고, 점수 버킷별 기대값(무누수 워크포워드)을 long-only
비중으로 매핑한다. 근거·기각 이력은 docs/strategy/specs/yoon_pattern.md 참조.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.pattern_projection import (  # noqa: E402
    run_pattern_expectancy_pipeline,
)
from profile_sizing.run import rebased_equity, slice_window  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402


class YoonPatternHandler:
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        cfg = config_from_dict(config)
        if synthetic:
            return make_synthetic_ohlcv(cfg.synthetic_bars, cfg.seed, cfg.interval)
        if csv_path is not None:
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.columns = [c.lower() for c in df.columns]
            return df
        from run_kalman_pipeline import load_yfinance  # noqa: E402
        return load_yfinance(symbol, cfg.interval, cfg.period)

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
        pattern = dict(config.get("pattern") or {})
        risk_overlay = dict(config.get("risk_overlay") or {})
        out = run_pattern_expectancy_pipeline(raw, cfg, pattern,
                                              risk_overlay=risk_overlay)

        index = self._naive_index(raw)
        window = slice_window(index, phase, cfg)

        forecast = out["forecast"].copy()
        forecast.index = index
        forecast = forecast.loc[window]

        equity = rebased_equity(out["port_ret"], window)
        port_ret = out["port_ret"].reindex(window).fillna(0.0)
        benchmark = rebased_equity(out["buy_hold"].pct_change().fillna(0.0), window)
        bnh_ret = out["buy_hold"].pct_change().fillna(0.0).reindex(window).fillna(0.0)
        perf = performance(equity, port_ret, cfg.interval)
        bnh_perf = performance(benchmark, bnh_ret, cfg.interval)
        trades = self._slice_trades(out["trades"], window)

        metrics = self._metrics(perf, bnh_perf, trades, phase)
        metadata = {
            "n_bars": int(len(window)),
            "timeframe": cfg.interval,
            "pattern_length": int(pattern.get("length", 300)),
            "pattern_horizon": int(pattern.get("horizon",
                                               cfg.expectancy.primary_horizon)),
            "pattern_method": str(pattern.get("method", "analog")),
            "risk_overlay": ("vol_target %.0f%%" % (float(
                risk_overlay.get("vol_target", 0.15)) * 100)
                if risk_overlay.get("enabled") else "off"),
            "avg_exposure": float(forecast["actual_weight"].mean())
            if "actual_weight" in forecast and len(forecast) else 0.0,
        }
        extras = {
            "expectancy_table": self._latest_expectancy_table(
                out["expectancy_tables"]
            ),
            "perf_vs_bnh": self._perf_table(perf, bnh_perf),
        }
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
    def _latest_expectancy_table(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty or "bar" not in table:
            return pd.DataFrame()
        latest = table[table["bar"] == table["bar"].max()].copy()
        return latest.drop(columns=["bar"], errors="ignore")

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
