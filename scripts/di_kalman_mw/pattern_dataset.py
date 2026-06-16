"""Leakage-safe P1..P4 feature table with P5 and price-path outcomes."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .extreme_transition import TransitionInstance
from .extremes import Extreme

SCHEMA_VERSION = "p1-p4-to-p5-v1"

CATEGORICAL_FEATURE_COLUMNS = ("line", "pattern", "shape")
NUMERIC_FEATURE_COLUMNS = (
    "p2_rel",
    "p3_rel",
    "p4_rel",
    "leg1_norm",
    "leg2_norm",
    "leg3_norm",
    "leg1_bars",
    "leg2_bars",
    "leg3_bars",
    "retr_ratio",
    "leg3_ratio",
    "p3_vs_p1_norm",
    "width_ratio",
    "span_bars",
    "plus_extreme_mean_4",
    "minus_extreme_mean_4",
    "di_pressure_spread",
    "atr_pct",
    "return_volatility_48",
)
OUTCOME_COLUMNS = (
    "has_p5",
    "p5_value",
    "p5_dv_norm",
    "p5_vs_p3_norm",
    "continuation",
    "bars_to_p5",
    "bars_to_p5_confirmation",
    "directional_return_to_p5",
    "directional_mfe_to_p5",
    "directional_mae_to_p5",
    "directional_return_20",
    "directional_return_48",
)


def _confirmed_values(
    extremes: Iterable[Extreme], decision_idx: int, count: int = 4
) -> list[float]:
    values = [
        float(extreme.value)
        for extreme in extremes
        if extreme.confirmation_idx <= decision_idx
    ]
    return values[-count:]


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _direction(line: str, pattern: str) -> int:
    long_setup = (line == "plus" and pattern == "W") or (
        line == "minus" and pattern == "M"
    )
    return 1 if long_setup else -1


def _price_outcomes(
    df: pd.DataFrame,
    decision_idx: int,
    outcome_idx: int,
    direction: int,
    labels: np.ndarray,
) -> dict[str, float]:
    n = len(df)
    entry_idx = decision_idx + 1
    if entry_idx >= n:
        return {}
    entry = float(df["open"].iloc[entry_idx])
    if not np.isfinite(entry) or entry <= 0:
        return {}

    result: dict[str, float] = {}
    if outcome_idx >= entry_idx and outcome_idx < n:
        path = df.iloc[entry_idx:outcome_idx + 1]
        exit_price = float(df["close"].iloc[outcome_idx])
        result["directional_return_to_p5"] = direction * (
            exit_price / entry - 1.0
        )
        if direction > 0:
            result["directional_mfe_to_p5"] = float(
                path["high"].max() / entry - 1.0
            )
            result["directional_mae_to_p5"] = float(
                path["low"].min() / entry - 1.0
            )
        else:
            result["directional_mfe_to_p5"] = float(
                1.0 - path["low"].min() / entry
            )
            result["directional_mae_to_p5"] = float(
                1.0 - path["high"].max() / entry
            )

    for horizon in (20, 48):
        exit_idx = decision_idx + horizon
        if (
            exit_idx < n
            and labels[exit_idx] == labels[decision_idx]
        ):
            result[f"directional_return_{horizon}"] = direction * (
                float(df["close"].iloc[exit_idx]) / entry - 1.0
            )
    return result


def build_pattern_frame(
    df: pd.DataFrame,
    atr_series: pd.Series,
    instances: list[TransitionInstance],
    plus_extremes: list[Extreme],
    minus_extremes: list[Extreme],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build a versioned table whose model features contain no P5 outcome."""
    returns = df["close"].pct_change()
    volatility = returns.rolling(48, min_periods=20).std()
    rows: list[dict] = []

    for instance_id, x in enumerate(instances):
        decision_idx = x.p4_conf_idx
        if not 0 <= decision_idx < len(df):
            continue
        values = np.asarray(x.window_val, dtype=float)
        indices = np.asarray(x.window_idx, dtype=int)
        if values.size != 4 or indices.size != 4:
            continue
        scale = max(float(x.mean_leg), 1e-9)
        plus_values = _confirmed_values(plus_extremes, decision_idx)
        minus_values = _confirmed_values(minus_extremes, decision_idx)
        plus_mean = _mean_or_nan(plus_values)
        minus_mean = _mean_or_nan(minus_values)
        pressure_denom = plus_mean + minus_mean
        pressure = (
            (plus_mean - minus_mean) / pressure_denom
            if np.isfinite(pressure_denom) and pressure_denom > 1e-9
            else float("nan")
        )
        close = float(df["close"].iloc[decision_idx])
        atr_value = float(atr_series.iloc[decision_idx])
        direction = _direction(x.line, x.pattern)

        row = {
            "schema_version": SCHEMA_VERSION,
            "instance_id": instance_id,
            "decision_time": df.index[decision_idx],
            "decision_idx": decision_idx,
            "decision_split": str(labels[decision_idx]),
            "line": x.line,
            "pattern": x.pattern,
            "shape": x.shape,
            "predicted_direction": "long" if direction > 0 else "short",
            "p4_value": float(values[3]),
            "mean_leg": scale,
            "p2_rel": (values[1] - values[0]) / scale,
            "p3_rel": (values[2] - values[0]) / scale,
            "p4_rel": (values[3] - values[0]) / scale,
            "leg1_norm": abs(values[1] - values[0]) / scale,
            "leg2_norm": abs(values[2] - values[1]) / scale,
            "leg3_norm": abs(values[3] - values[2]) / scale,
            "leg1_bars": float(indices[1] - indices[0]),
            "leg2_bars": float(indices[2] - indices[1]),
            "leg3_bars": float(indices[3] - indices[2]),
            "retr_ratio": float(x.features["retr_ratio"]),
            "leg3_ratio": float(x.features["leg3_ratio"]),
            "p3_vs_p1_norm": float(x.features["p3_vs_p1_norm"]),
            "width_ratio": float(x.width_ratio),
            "span_bars": float(x.features["span_bars"]),
            "plus_extreme_mean_4": plus_mean,
            "minus_extreme_mean_4": minus_mean,
            "di_pressure_spread": pressure,
            "atr_pct": atr_value / close if close > 0 else float("nan"),
            "return_volatility_48": float(volatility.iloc[decision_idx]),
            "has_p5": bool(x.has_p5),
            "p5_value": float(x.p5_value),
            "p5_dv_norm": float(x.dv_norm),
            "p5_vs_p3_norm": float(x.p5_vs_p3_norm),
            "continuation": bool(x.continuation) if x.has_p5 else None,
            "bars_to_p5": (
                float(x.p5_idx - x.p4_idx) if x.has_p5 else float("nan")
            ),
            "bars_to_p5_confirmation": (
                float(x.p5_conf_idx - x.p4_conf_idx)
                if x.has_p5 else float("nan")
            ),
            "outcome_idx": x.p5_conf_idx if x.has_p5 else -1,
            "outcome_split": (
                str(labels[x.p5_conf_idx])
                if x.has_p5 and 0 <= x.p5_conf_idx < len(labels)
                else None
            ),
        }
        if x.has_p5:
            row.update(
                _price_outcomes(
                    df, decision_idx, x.p5_conf_idx, direction, labels
                )
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in OUTCOME_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def completed_split_frame(
    frame: pd.DataFrame, split: str
) -> pd.DataFrame:
    """Select rows whose decision and observed P5 both belong to ``split``."""
    return frame[
        frame["has_p5"].astype(bool)
        & (frame["decision_split"] == split)
        & (frame["outcome_split"] == split)
    ].copy()
