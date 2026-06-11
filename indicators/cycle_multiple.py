"""Normalize OHLC prices by a base-cycle series."""

import numpy as np
import pandas as pd

OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def add_cycle_multiple(
    df: pd.DataFrame, base_column: str = "base_cycle"
) -> pd.DataFrame:
    missing = [column for column in (*OHLC_COLUMNS, base_column) if column not in df]
    if missing:
        raise ValueError(f"Missing columns for cycle multiple: {missing}")

    out = df.copy()
    base = out[base_column].astype(float).replace(0.0, np.nan)
    for column in OHLC_COLUMNS:
        out[f"cm_{column.lower()}"] = out[column].astype(float) / base
    return out
