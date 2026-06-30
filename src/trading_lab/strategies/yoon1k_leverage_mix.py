"""yoon1k leverage mix — defensive yoon1k core plus leveraged ETF sleeve."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from trading_lab.strategies.base import StrategyArtifacts
from trading_lab.strategies.yoon1k import Yoon1kHandler

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402

_TRADE_COLS = [
    "direction", "entry_time", "entry_price", "exit_time",
    "exit_price", "net_return", "exit_reason", "entry_reason", "symbol",
]


class Yoon1kLeverageMixHandler(Yoon1kHandler):
    """Blend yoon1k with buy-and-hold leveraged ETFs at the top level."""

    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        core_cfg = self._core_config(config)
        raw = super().load_data(symbol, core_cfg, csv_path=csv_path, synthetic=synthetic)
        overlays = self._load_overlays(config, synthetic=synthetic)
        if overlays is None:
            return raw
        return pd.concat([raw, overlays], axis=1)

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        core_cfg = self._core_config(config)
        core_art = super().build_artifacts(
            raw, core_cfg, symbol=symbol, phase="all", bars_per_year=bars_per_year,
        )

        core = (core_art.equity / core_art.equity.iloc[0]).rename(
            str(config.get("core_strategy", "yoon1k"))
        )
        mix = config.get("leverage_mix") or {}
        core_weight = float(mix.get("core_weight", 1.0))
        weights = []
        navs = []
        labels = []
        sub_trades = []
        if core_weight > 0.0:
            weights.append(core_weight)
            navs.append(core)
            labels.append(core.name)
            sub_trades.append(core_art.trades)

        fx = self._fx(raw, str(config.get("fx_symbol", core_cfg.get("fx_symbol", ""))))
        for overlay in mix.get("overlays", []):
            sym = str(overlay["symbol"]).upper()
            close = raw["__OVERLAY__"][sym]["close"].dropna().astype(float)
            nav = (close / close.iloc[0]).rename(sym)
            if str(overlay.get("currency", "USD")) != str(config.get("base_currency", "KRW")):
                nav = self._to_base(nav, fx).rename(sym)
            navs.append(nav)
            weights.append(float(overlay.get("weight", 0.0)))
            labels.append(str(overlay.get("label", sym)))

        if not navs:
            raise ValueError("leverage_mix must include a positive core or overlay weight")

        idx = navs[0].index
        for nav in navs[1:]:
            idx = idx.intersection(nav.index)
        idx = idx.sort_values()
        wsum = sum(weights) or 1.0
        weights = [w / wsum for w in weights]
        freq = str(config.get("rebalance_freq", mix.get("rebalance_freq", "monthly")))
        values = self._combine_values(
            [(nav.reindex(idx), weight) for nav, weight in zip(navs, weights)],
            idx,
            freq,
            labels,
        )
        combo = values.sum(axis=1)

        window = self._slice(idx, phase, config)
        equity_raw = combo.reindex(window)
        equity = (equity_raw / equity_raw.iloc[0]).rename("equity")
        ret = equity.pct_change().fillna(0.0)
        perf = performance(equity, ret, str(config.get("interval", "1d")))

        benchmark = None
        bperf = {}
        if core_art.benchmark is not None:
            b = core_art.benchmark.reindex(window).ffill().dropna()
            if not b.empty:
                benchmark = (b / b.iloc[0]).rename("benchmark")
                bperf = performance(
                    benchmark,
                    benchmark.pct_change().fillna(0.0),
                    str(config.get("interval", "1d")),
                )

        forecast = pd.DataFrame({"close": equity}, index=window)
        for label, nav in zip(labels, navs):
            leg = nav.reindex(window).ffill()
            forecast[f"leg_{label}"] = leg / leg.iloc[0]

        trades = self._slice_trades(sub_trades, window)
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)

        shares = values.reindex(window).div(values.reindex(window).sum(axis=1), axis=0)
        shares = shares.fillna(0.0)
        shares.index.name = "time"

        metrics = {
            "trades": n,
            "hit_rate": float((rets > 0).mean()) if n else None,
            "total_return": perf.get("total_return"),
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bperf.get("total_return"),
            "buy_hold_sharpe": bperf.get("sharpe"),
            "buy_hold_max_drawdown": bperf.get("max_drawdown"),
            "buy_hold_cagr": bperf.get("cagr"),
            "benchmark_kind": str(config.get("benchmark_label", "50/50")),
            "phase": phase,
        }
        metadata = {
            "core_strategy": str(config.get("core_strategy", "yoon1k")),
            "leverage_mix": mix,
            "weights": dict(zip(labels, weights)),
            "rebalance_freq": freq,
            "base_currency": str(config.get("base_currency", "KRW")),
            "fx_symbol": str(config.get("fx_symbol", core_cfg.get("fx_symbol", ""))),
        }
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades[_TRADE_COLS] if not trades.empty else trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            benchmark=benchmark,
            extras={
                "sleeve_weights": shares.reset_index(),
                "perf_vs_bnh": self._perf_table(
                    perf, bperf, label=str(config.get("version", "yoon1k_levmix")),
                ),
            },
        )

    @staticmethod
    def _core_config(config: dict[str, Any]) -> dict[str, Any]:
        core_id = str(config.get("core_strategy", "yoon1k"))
        path = ROOT / "configs" / "strategies" / f"{core_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_overlays(config: dict[str, Any], *, synthetic: bool) -> pd.DataFrame | None:
        overlays = (config.get("leverage_mix") or {}).get("overlays", [])
        if not overlays:
            return None

        panels = {}
        if synthetic:
            bars = int(config.get("synthetic_bars", 2000))
            seed = int(config.get("seed", 7))
            interval = str(config.get("interval", "1d"))
            for i, overlay in enumerate(overlays):
                panels[str(overlay["symbol"]).upper()] = make_synthetic_ohlcv(
                    bars, seed + 100 + i, interval,
                )
        else:
            from run_kalman_pipeline import load_yfinance  # noqa: E402

            interval = str(config.get("interval", "1d"))
            period = str(config.get("period", "max"))
            for overlay in overlays:
                sym = str(overlay["symbol"]).upper()
                panels[sym] = load_yfinance(sym, interval, period)
        return pd.concat({"__OVERLAY__": pd.concat(panels, axis=1)}, axis=1)

    @staticmethod
    def _perf_table(perf: dict, bench: dict, *, label: str = "yoon1k_levmix") -> pd.DataFrame:
        rows = [
            {
                "라인": label,
                "CAGR": perf.get("cagr"),
                "MDD": perf.get("max_drawdown"),
                "Sharpe": perf.get("sharpe"),
                "Total Return": perf.get("total_return"),
            }
        ]
        if bench:
            rows.append({
                "라인": "50/50 B&H",
                "CAGR": bench.get("cagr"),
                "MDD": bench.get("max_drawdown"),
                "Sharpe": bench.get("sharpe"),
                "Total Return": bench.get("total_return"),
            })
        return pd.DataFrame(rows)
