"""Handler for the di-kalman-mw-v1 strategy.

Runs the standalone research pipeline (scripts/di_kalman_mw) on common
dashboard data (yfinance / synthetic / csv) and maps its outputs onto the
shared StrategyArtifacts contract so it renders in the common result view.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from trading_lab.paths import ROOT
from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from di_kalman_mw import run as runner  # noqa: E402
from di_kalman_mw.config import combo_config  # noqa: E402
from di_kalman_mw.viz import split_metrics_table  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

# DI exit reasons -> shared presentation labels (EXIT_REASON_LABELS keys).
_EXIT_REASON_MAP = {
    "stop": "stop_loss",
    "opposite_pattern": "opposite",
    "time_stop": "horizon",
    # "take_profit" / "end_of_data" already match.
}

_SHAPE_LABELS = {
    "diverging": "확산",
    "converging": "수렴",
    "parallel": "유지",
    "": "-",
}

_DASHBOARD_METRIC_KEYS = (
    "num_trades", "total_return", "max_drawdown", "sharpe", "win_rate",
    "profit_factor", "expectancy",
)


class DiKalmanMwHandler:
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        if synthetic:
            return runner.make_synthetic_ohlcv(
                int(config.get("synthetic_bars", 9000)),
                int(config.get("seed", 7)),
                str(config.get("interval", "1d")),
            )
        if csv_path is not None:
            return runner.load_data(csv_path)
        return load_yfinance(
            symbol, config.get("interval", "1d"), config.get("period", "max")
        )

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        timeframe = str(config.get("interval", "1d"))
        cfg = self._strategy_config(config)
        run_metrics, art = runner.run_pipeline(
            raw, symbol, timeframe, cfg, outdir=None
        )

        equity_df = art["equity"]  # columns: bar_return, equity, split
        window = self._phase_window(equity_df, phase)
        bar_return = equity_df.loc[window, "bar_return"]
        equity = (1.0 + bar_return).cumprod().rename("equity")

        forecast = self._forecast_frame(raw, art).loc[window]
        trades = self._map_trades(art["trades"], phase)
        metrics = self._map_metrics(run_metrics, phase)
        extras = {
            "split_breakdown": split_metrics_table(run_metrics).reset_index(),
        }
        metadata = {
            "n_events": run_metrics.get("n_events"),
            "entry_variant": run_metrics.get("entry_variant"),
            "insufficient_train_data": run_metrics.get(
                "insufficient_train_data"
            ),
            "timeframe": timeframe,
        }
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
            extras=extras,
        )

    @staticmethod
    def _strategy_config(config: dict[str, Any]):
        cfg = combo_config(str(config.get("combo", "A")))
        cfg = dataclasses.replace(
            cfg,
            direction_mode=config.get("direction", cfg.direction_mode),
            indicators=dataclasses.replace(
                cfg.indicators,
                kalman_q=float(config.get("kalman_q", cfg.indicators.kalman_q)),
                di_len=int(config.get("di_len", cfg.indicators.di_len)),
                atr_len=int(config.get("atr_len", cfg.indicators.atr_len)),
            ),
            extremes=dataclasses.replace(
                cfg.extremes,
                reversal_mult=float(
                    config.get("reversal_mult", cfg.extremes.reversal_mult)
                ),
                reversal_std_window=int(
                    config.get("reversal_window", cfg.extremes.reversal_std_window)
                ),
            ),
            costs=dataclasses.replace(
                cfg.costs, cost_mult=float(config.get("cost_mult", cfg.costs.cost_mult))
            ),
            similarity_ev=dataclasses.replace(
                cfg.similarity_ev,
                enabled=str(
                    config.get("similarity_ev", "off")
                ).lower() == "on",
                entry_margin=float(
                    config.get(
                        "similarity_entry_margin",
                        cfg.similarity_ev.entry_margin,
                    )
                ),
            ),
            online=dataclasses.replace(
                cfg.online,
                enabled=str(config.get("online_revaluation", "off")).lower()
                == "on",
            ),
            split=dataclasses.replace(
                cfg.split,
                train_frac=float(config.get("train_frac", cfg.split.train_frac)),
                validation_frac=float(
                    config.get("validation_frac", cfg.split.validation_frac)
                ),
            ),
        )
        # combo determines the entry variant; only override when explicit.
        if config.get("entry_variant"):
            cfg = dataclasses.replace(
                cfg,
                signal=dataclasses.replace(
                    cfg.signal, entry_variant=str(config["entry_variant"])
                ),
            )
        return cfg

    @staticmethod
    def _phase_window(equity_df: pd.DataFrame, phase: str) -> pd.Index:
        if phase == "all":
            return equity_df.index
        return equity_df.index[equity_df["split"] == phase]

    @staticmethod
    def _forecast_frame(raw: pd.DataFrame, art: dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame(index=raw.index)
        for column in ("open", "high", "low", "close"):
            if column in raw:
                frame[column] = raw[column].astype(float)
        frame["plus_di_kalman"] = art["plus_kalman"].astype(float)
        frame["minus_di_kalman"] = art["minus_kalman"].astype(float)
        frame["atr"] = art["atr"].astype(float)
        return frame

    @staticmethod
    def _map_trades(di_trades: pd.DataFrame, phase: str) -> pd.DataFrame:
        columns = [
            "direction", "entry_time", "entry_price", "exit_time",
            "exit_price", "stop_loss_price", "take_profit_price",
            "net_return", "exit_reason", "entry_reason",
        ]
        source = (
            di_trades if phase == "all"
            else di_trades[di_trades["split"] == phase]
        )
        if source.empty:
            return pd.DataFrame(columns=columns)
        out = pd.DataFrame(index=range(len(source)))
        src = source.reset_index(drop=True)
        out["direction"] = src["direction"].map({"long": 1, "short": -1})
        out["entry_time"] = pd.to_datetime(src["entry_time"])
        out["entry_price"] = src["entry_price"].astype(float)
        out["exit_time"] = pd.to_datetime(src["exit_time"])
        out["exit_price"] = src["exit_price"].astype(float)
        out["stop_loss_price"] = src["stop_price"].astype(float)
        out["take_profit_price"] = src["take_profit_price"].astype(float)
        out["net_return"] = src["pnl_pct"].astype(float)
        out["exit_reason"] = src["exit_reason"].map(
            lambda value: _EXIT_REASON_MAP.get(str(value), str(value))
        )
        out["entry_reason"] = [
            DiKalmanMwHandler._entry_reason(row) for _, row in src.iterrows()
        ]
        return out[columns]

    @staticmethod
    def _entry_reason(row: pd.Series) -> str:
        tier = str(row.get("tier", "") or "")
        tier_label = {"strong": "강", "weak": "약"}.get(tier, "")
        shape_label = _SHAPE_LABELS.get(str(row.get("setup_shape", "") or ""), "-")
        side = "롱" if row["direction"] == "long" else "숏"
        text = (
            f"{str(row['entry_variant']).upper()} 진입 · "
            f"{tier_label}{shape_label} · {side} · "
            f"pressure {float(row['pressure_score']):.2f}"
        )
        cont = row.get("p_continuation")
        if cont is not None and pd.notna(cont):
            text += f" · P(cont) {float(cont):.2f}"
        return text

    @staticmethod
    def _map_metrics(run_metrics: dict[str, Any], phase: str) -> dict[str, Any]:
        if phase == "all":
            source = {k: run_metrics.get(k) for k in _DASHBOARD_METRIC_KEYS}
        else:
            source = run_metrics.get(f"{phase}_metrics", {})
        return {
            "trades": int(source.get("num_trades") or 0),
            "hit_rate": source.get("win_rate"),
            "total_return": source.get("total_return") or 0.0,
            "sharpe": source.get("sharpe"),
            "max_drawdown": source.get("max_drawdown"),
            "profit_factor": source.get("profit_factor"),
            "expectancy": source.get("expectancy"),
            "phase": phase,
        }
