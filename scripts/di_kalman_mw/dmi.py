"""Wilder DMI / ATR and Kalman smoothing of DI lines (plan 5A.4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from indicators.kalman import kalman_1d  # noqa: E402


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return wilder_rma(true_range(df), length).rename("atr")


def dmi(df: pd.DataFrame, di_len: int = 14) -> tuple[pd.Series, pd.Series]:
    """Standard Wilder DMI. Returns (+DI, -DI) in the 0..100 range."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )
    tr_smooth = wilder_rma(true_range(df), di_len)
    plus_di = (100.0 * wilder_rma(plus_dm, di_len) / tr_smooth).rename("plus_di")
    minus_di = (100.0 * wilder_rma(minus_dm, di_len) / tr_smooth).rename("minus_di")
    return plus_di, minus_di


def kalman_di(series: pd.Series, q: float, r: float) -> pd.Series:
    return kalman_1d(series, q=q, r=r)
