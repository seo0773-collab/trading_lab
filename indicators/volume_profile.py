"""Cycle-multiple range-volume profile calculations."""

import numpy as np
import pandas as pd


def calculate_profile_bins(
    df: pd.DataFrame,
    min_mult: float = 0.0,
    max_mult: float = 5.0,
    bins: int = 400,
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if min_mult >= max_mult:
        raise ValueError("min_mult must be less than max_mult")

    missing = sorted({"cm_low", "cm_high"}.difference(df.columns))
    if missing:
        raise ValueError(f"Missing profile columns: {missing}")

    edges = np.linspace(min_mult, max_mult, bins + 1)
    values = np.zeros(bins, dtype=float)
    lows = df["cm_low"].to_numpy(dtype=float)
    highs = df["cm_high"].to_numpy(dtype=float)
    volume = df.get("Volume", pd.Series(1.0, index=df.index))
    volumes = volume.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

    for low, high, bar_volume in zip(lows, highs, volumes):
        if not np.isfinite(low) or not np.isfinite(high) or bar_volume <= 0:
            continue
        if low > high:
            low, high = high, low
        if low > max_mult or high < min_mult:
            continue

        low = max(low, min_mult)
        high = min(high, max_mult)
        if high <= low:
            index = np.searchsorted(edges, low, side="right") - 1
            if 0 <= index < bins:
                values[index] += bar_volume
            continue

        start = max(0, np.searchsorted(edges, low, side="right") - 1)
        end = min(bins - 1, np.searchsorted(edges, high, side="left"))
        bar_range = high - low
        for index in range(start, end + 1):
            overlap = max(
                min(high, edges[index + 1]) - max(low, edges[index]), 0.0
            )
            values[index] += bar_volume * overlap / bar_range

    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "mult": (edges[:-1] + edges[1:]) / 2.0,
            "value": values,
        }
    )


def weighted_percentile(profile: pd.DataFrame, percentile: float) -> float:
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")

    weights = profile["value"].fillna(0.0).clip(lower=0.0).to_numpy(float)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    index = min(
        int(np.searchsorted(np.cumsum(weights), percentile * total)),
        len(profile) - 1,
    )
    return float(profile["mult"].iloc[index])


def summarize_profile(
    profile: pd.DataFrame,
    lower_percentile: float = 0.15,
    upper_percentile: float = 0.85,
) -> dict[str, float]:
    values = profile["value"].fillna(0.0).clip(lower=0.0).to_numpy(float)
    total = float(values.sum())
    poc = (
        float(profile["mult"].iloc[int(np.argmax(values))])
        if total > 0
        else float("nan")
    )
    return {
        "total": total,
        "poc": poc,
        "lower_percentile": weighted_percentile(profile, lower_percentile),
        "upper_percentile": weighted_percentile(profile, upper_percentile),
        "lower_percentile_level": float(lower_percentile),
        "upper_percentile_level": float(upper_percentile),
    }


profile_bins = calculate_profile_bins
