"""2-Pass expected-value statistics (plan 7, 7A).

Pass 1 simulates every train pattern event with the baseline exit rule
(no EV filter) and aggregates win_probability / avg_win / avg_loss per
(direction, pressure_aligned) bucket. Pass 2 applies the frozen stats to
signals in every split; validation/test never update them.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .config import CostConfig, StatsConfig

EPS = 1e-9


def simulate_baseline_trade(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_values: np.ndarray,
    event,
    cfg: StatsConfig,
    *,
    signal_idx: int | None = None,
    outcome_end_exclusive: int | None = None,
) -> float | None:
    """Hypothetical trade for Pass 1: next-open entry, ATR stop, fixed R TP,
    time stop. Returns gross pnl_pct (costs are handled in the EV formula).

    When ``outcome_end_exclusive`` is set, an outcome is usable only when the
    stop, target, or time-stop result is fully observed before that boundary.
    This prevents a train event from learning from validation prices.
    """
    n = len(open_)
    decision_i = event.event_idx if signal_idx is None else signal_idx
    if decision_i is None:
        return None
    entry_i = decision_i + 1
    boundary = n if outcome_end_exclusive is None else min(
        n, outcome_end_exclusive
    )
    if decision_i < 0 or entry_i >= boundary:
        return None
    a = atr_values[decision_i]
    if not np.isfinite(a) or a <= 0:
        return None
    d = 1.0 if event.direction == "long" else -1.0
    entry = open_[entry_i]
    stop = entry - d * a * cfg.baseline_atr_stop_mult
    risk = d * (entry - stop)
    tp = entry + d * risk * cfg.baseline_rr_target
    natural_last = min(entry_i + cfg.baseline_max_hold_bars, n - 1)
    last = min(natural_last, boundary - 1)
    for i in range(entry_i, last + 1):
        if d > 0:
            if low[i] <= stop:
                px = open_[i] if open_[i] <= stop else stop
                return px / entry - 1.0
            if high[i] >= tp:
                px = open_[i] if open_[i] >= tp else tp
                return px / entry - 1.0
        else:
            if high[i] >= stop:
                px = open_[i] if open_[i] >= stop else stop
                return d * (px / entry - 1.0)
            if low[i] <= tp:
                px = open_[i] if open_[i] <= tp else tp
                return d * (px / entry - 1.0)
    if natural_last >= boundary:
        return None
    return d * (close[last] / entry - 1.0)


def _bucket(pnls: list[float]) -> dict:
    arr = np.asarray(pnls, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    return {
        "n": int(arr.size),
        "win_probability": float(wins.size / arr.size) if arr.size else 0.0,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(np.abs(losses.mean())) if losses.size else 0.0,
    }


def build_train_stats(
    df: pd.DataFrame,
    atr_series: pd.Series,
    train_events: list,
    cfg: StatsConfig,
    *,
    decision_index: Callable[[Any], int | None] | None = None,
    outcome_end_exclusive: int | None = None,
    entry_variant: str = "p4",
) -> dict:
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_values = atr_series.to_numpy(dtype=float)

    rows: list[tuple[str, bool, float]] = []
    incomplete = 0
    for ev in train_events:
        signal_idx = decision_index(ev) if decision_index is not None else None
        if decision_index is not None and signal_idx is None:
            incomplete += 1
            continue
        pnl = simulate_baseline_trade(
            open_,
            high,
            low,
            close,
            atr_values,
            ev,
            cfg,
            signal_idx=signal_idx,
            outcome_end_exclusive=outcome_end_exclusive,
        )
        if pnl is not None:
            rows.append((ev.direction, bool(ev.pressure_aligned), float(pnl)))
        else:
            incomplete += 1

    stats: dict = {
        "buckets": {},
        "directions": {},
        "global": _bucket([p for _, _, p in rows]),
        "entry_variant": entry_variant,
        "n_candidates": len(train_events),
        "n_simulated": len(rows),
        "n_incomplete": incomplete,
        "outcome_end_exclusive": outcome_end_exclusive,
    }
    for direction in ("long", "short"):
        d_pnls = [p for d, _, p in rows if d == direction]
        if d_pnls:
            stats["directions"][direction] = _bucket(d_pnls)
        for aligned in (True, False):
            b = [p for d, a, p in rows if d == direction and a == aligned]
            if b:
                stats["buckets"][f"{direction}|aligned={aligned}"] = _bucket(b)
    return stats


def lookup_stats(
    stats: dict, direction: str, pressure_aligned: bool, cfg: StatsConfig
) -> dict | None:
    """Fallback chain (plan 7A): bucket -> direction -> global."""
    key = f"{direction}|aligned={bool(pressure_aligned)}"
    b = stats.get("buckets", {}).get(key)
    if b and b["n"] >= cfg.min_bucket_trades:
        return {**b, "bucket": key}
    b = stats.get("directions", {}).get(direction)
    if b and b["n"] >= cfg.min_bucket_trades:
        return {**b, "bucket": direction}
    g = stats.get("global")
    if g and g["n"] >= cfg.min_global_trades:
        return {**g, "bucket": "global"}
    return None


def continuation_factor(p_continuation: float) -> float:
    """plan 6/7: map a next-extreme continuation probability to a [0.5, 1.5]
    win/loss tilt. Neutral (0.5 -> 1.0) leaves the expected value unchanged;
    NaN (no transition estimate) is treated as neutral."""
    if not np.isfinite(p_continuation):
        return 1.0
    return float(np.clip(0.5 + p_continuation, 0.5, 1.5))


def expected_values(
    st: dict, pressure_rr_factor: float, costs: CostConfig,
    p_continuation: float = float("nan"),
) -> tuple[float, float]:
    """plan 7: raw and pressure-adjusted expected value.

    The pressure-adjusted value additionally tilts win/loss by the pattern's
    continuation probability (continuation_factor): a high P(continuation)
    inflates the win leg and deflates the loss leg, a low one does the reverse.
    """
    p = st["win_probability"]
    avg_win = st["avg_win"]
    avg_loss = st["avg_loss"]
    c = costs.round_trip_cost
    raw = p * avg_win - (1.0 - p) * avg_loss - c
    cf = continuation_factor(p_continuation)
    adjusted = (
        p * avg_win * pressure_rr_factor * cf
        - (1.0 - p) * avg_loss / max(pressure_rr_factor * cf, EPS)
        - c
    )
    return raw, adjusted
