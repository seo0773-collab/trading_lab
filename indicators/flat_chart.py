"""Flat-chart calculation pipeline shared by UI and strategy code."""

import pandas as pd

from .cycle_base import add_base_cycle
from .cycle_multiple import add_cycle_multiple
from .gaussian_profile import add_gaussian_expectation, fit_gaussian_profile
from .volume_profile import calculate_profile_bins, summarize_profile


def _validate_ohlcv(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("OHLCV data is empty")
    missing = sorted({"Open", "High", "Low", "Close"}.difference(df.columns))
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}")


def build_flat_chart(
    df: pd.DataFrame,
    mode: str = "kalman",
    length: int = 200,
    bins: int = 400,
    *,
    min_mult: float = 0.0,
    max_mult: float = 5.0,
    lower_percentile: float = 0.15,
    upper_percentile: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return calculated bars, profile bins, and a combined summary."""
    _validate_ohlcv(df)
    calculated = add_base_cycle(df, mode=mode, length=length)
    calculated = add_cycle_multiple(calculated)
    profile = calculate_profile_bins(
        calculated, min_mult=min_mult, max_mult=max_mult, bins=bins
    )
    summary = summarize_profile(
        profile,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    fit = fit_gaussian_profile(profile)
    profile = add_gaussian_expectation(profile, fit)
    summary.update({"mu": fit["mu"], "sigma": fit["sigma"]})
    return calculated, profile, summary


__all__ = ["build_flat_chart"]
