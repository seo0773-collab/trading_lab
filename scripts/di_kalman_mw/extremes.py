"""Causal extreme extraction with reversal-threshold confirmation (plan 5A.1).

Extreme idx and confirmation_idx are kept separate; downstream code must
only act on confirmation_idx (plan 16).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExtremeConfig


@dataclass(frozen=True)
class Extreme:
    idx: int
    kind: str  # "H" | "L"
    value: float
    confirmation_idx: int


def _nan_argext(values: np.ndarray, lo: int, hi: int, find_max: bool) -> int:
    window = values[lo:hi + 1]
    j = int(np.nanargmax(window)) if find_max else int(np.nanargmin(window))
    return lo + j


def extract_extremes(series: pd.Series, cfg: ExtremeConfig) -> list[Extreme]:
    values = series.to_numpy(dtype=float)
    thresholds = (
        series.rolling(cfg.reversal_std_window, min_periods=cfg.reversal_std_window)
        .std()
        .to_numpy(dtype=float)
        * cfg.reversal_mult
    )
    n = len(values)
    extremes: list[Extreme] = []
    # state 0: initial (track both directions), -1: last was H (seek low),
    # +1: last was L (seek high). Alternation is enforced by construction.
    state = 0
    run_max_i = -1
    run_min_i = -1

    for i in range(n):
        x = values[i]
        if not np.isfinite(x):
            continue
        if run_max_i < 0:
            run_max_i = run_min_i = i
            continue
        if state >= 0 and x >= values[run_max_i]:
            run_max_i = i
        if state <= 0 and x <= values[run_min_i]:
            run_min_i = i
        t = thresholds[i]
        if not np.isfinite(t) or t <= 0:
            continue
        if state >= 0 and run_max_i < i and values[run_max_i] - x >= t:
            extremes.append(
                Extreme(run_max_i, "H", float(values[run_max_i]), i)
            )
            state = -1
            run_min_i = _nan_argext(values, run_max_i + 1, i, find_max=False)
            continue
        if state <= 0 and run_min_i < i and x - values[run_min_i] >= t:
            extremes.append(
                Extreme(run_min_i, "L", float(values[run_min_i]), i)
            )
            state = 1
            run_max_i = _nan_argext(values, run_min_i + 1, i, find_max=True)

    return extremes
