"""Chronological split and train sufficiency checks (plan 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitConfig, sufficiency_rule


def split_labels(n: int, cfg: SplitConfig) -> np.ndarray:
    labels = np.empty(n, dtype=object)
    train_end = int(n * cfg.train_frac)
    val_end = int(n * (cfg.train_frac + cfg.validation_frac))
    labels[:train_end] = "train"
    labels[train_end:val_end] = "validation"
    labels[val_end:] = "test"
    return labels


def train_sufficiency(
    timeframe: str, train_index: pd.DatetimeIndex, n_train_events: int
) -> dict:
    rule = sufficiency_rule(timeframe)
    years = 0.0
    if len(train_index) >= 2:
        years = (
            (train_index[-1] - train_index[0]).total_seconds()
            / (365.25 * 24 * 3600.0)
        )
    checks = {
        "years": years >= rule["min_years"],
        "candles": len(train_index) >= rule["min_candles"],
        "events": (
            rule["min_events"] is not None
            and n_train_events >= rule["min_events"]
        ),
    }
    sufficient = any(checks.values())
    return {
        "sufficient": sufficient,
        "insufficient_train_data": not sufficient,
        "train_years": round(years, 2),
        "train_candles": int(len(train_index)),
        "train_events": int(n_train_events),
        "rule": rule,
        "checks": checks,
    }
