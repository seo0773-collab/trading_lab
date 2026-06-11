"""Base-cycle calculations shared by dashboards and backtests."""

import pandas as pd

from .kalman import kalman_cv


def calculate_base_cycle(
    close: pd.Series,
    mode: str = "kalman",
    length: int = 200,
    *,
    q: float | None = None,
    r: float = 1.0,
) -> pd.Series:
    if length < 2:
        raise ValueError("length must be at least 2")

    values = close.astype(float)
    normalized_mode = mode.lower()
    if normalized_mode == "kalman":
        result = kalman_cv(values, q=q, r=r, equiv_len=length)
    elif normalized_mode == "sma":
        result = values.rolling(length, min_periods=length).mean()
    else:
        raise ValueError("mode must be either 'kalman' or 'sma'")

    return result.rename("base_cycle")


def add_base_cycle(
    df: pd.DataFrame,
    mode: str = "kalman",
    length: int = 200,
    *,
    q: float | None = None,
    r: float = 1.0,
) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("OHLCV data must contain a Close column")

    out = df.copy()
    out["base_cycle"] = calculate_base_cycle(
        out["Close"], mode=mode, length=length, q=q, r=r
    )
    return out
