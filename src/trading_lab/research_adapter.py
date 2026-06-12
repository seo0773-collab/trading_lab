from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ROOT


SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from conf_filter_backtest import build_signals  # noqa: E402
from flat_chart import FlatChartConfig, compute_features  # noqa: E402
from run_kalman_pipeline import (  # noqa: E402
    HORIZONS,
    calibrate_sigma,
    identify_params,
    load_csv,
    load_yfinance,
    make_synthetic,
    run_filter,
)
from strategy_execution import (  # noqa: E402
    ExecutionConfig,
    chronological_splits,
    run_execution,
    summarize_execution,
)


def forecast_columns() -> list[str]:
    return [
        "open", "close", "mult_close", "m_fast", "m_filt", "m_slow",
        "q_scale",
    ] + [
        f"{prefix}_{horizon}"
        for horizon in HORIZONS
        for prefix in ("mhat", "sig", "pup", "price_mid", "price_lo", "price_hi")
    ]


def load_market_data(
    symbol: str, config: dict[str, Any], *,
    csv_path: Path | None = None, synthetic: bool = False,
) -> pd.DataFrame:
    if synthetic:
        return make_synthetic()
    if csv_path is not None:
        return load_csv(str(csv_path))
    return load_yfinance(symbol, config["interval"], config["period"])


def build_forecast(
    raw: pd.DataFrame, config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean = raw.dropna(subset=["open", "high", "low", "close"]).sort_index()
    features = compute_features(
        clean,
        FlatChartConfig(
            cycle_len=config["cycle_len"],
            fast_window=config["fast_window"],
            slow_window=config["slow_window"],
        ),
    )
    params, identification = identify_params(
        features, ident_frac=config["identification_frac"]
    )
    result = run_filter(features, params)
    calibration = calibrate_sigma(
        result, ident_frac=config["identification_frac"]
    )
    metadata = {
        "raw_bars": int(len(clean)),
        "forecast_bars": int(len(result)),
        "start": str(result.index.min()),
        "end": str(result.index.max()),
        "identification": _json_value(identification),
        "sigma_calibration": _json_value(calibration),
    }
    return result[forecast_columns()].copy(), metadata


def execute_strategy(
    forecast: pd.DataFrame, config: dict[str, Any], *,
    symbol: str, phase: str, bars_per_year: int,
):
    signals, _ = build_signals(forecast, config["horizon"])
    split = chronological_splits(
        forecast.index,
        config["identification_frac"],
        config["validation_frac"],
    )
    entry_split = None if phase == "all" else phase
    result = run_execution(
        forecast,
        signals["price_dir"],
        signals["price_conf"],
        ExecutionConfig(
            horizon=config["horizon"],
            fee_bps=config["fee_bps_per_side"],
            conf_quantile=config["confidence_quantile"],
            quantile_window=config["quantile_window"],
            execution=config["execution"],
            exit_on_opposite=config["exit_on_opposite"],
            long_only=config["long_only"],
        ),
        asset=symbol,
        expected_edge=signals["price_edge"],
        mult_direction=signals["mult_dir"],
        split=split,
        entry_split=entry_split,
    )
    metrics = summarize_execution(result, bars_per_year)
    metrics.update({
        "symbol": symbol,
        "phase": phase,
        "bars": int(len(forecast)),
        "entry_after_signal": bool(
            result.trades.empty
            or (result.trades["entry_time"] > result.trades["signal_time"]).all()
        ),
    })
    return result, metrics


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
