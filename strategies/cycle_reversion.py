"""Example long-only cycle-multiple mean-reversion strategy for vectorbt."""

import numpy as np
import pandas as pd

from indicators.flat_chart import build_flat_chart


def build_target_allocations(
    cycle_close: pd.Series,
    lower: float,
    midpoint: float,
    upper: float,
    partial_target: float = 0.5,
) -> pd.Series:
    """Create target-percent orders: enter 100%, trim, then fully exit."""
    if not 0.0 <= partial_target < 1.0:
        raise ValueError("partial_target must be in [0, 1)")
    if not lower < upper:
        raise ValueError("lower must be less than upper")

    targets = pd.Series(np.nan, index=cycle_close.index, name="target_percent")
    position = 0.0
    was_below = False

    for index, value in cycle_close.items():
        if not np.isfinite(value):
            continue
        if position == 0.0:
            if value < lower:
                was_below = True
            elif was_below and value >= lower:
                position = 1.0
                targets.loc[index] = position
                was_below = False
        elif position > 0.0 and value >= upper:
            position = 0.0
            targets.loc[index] = position
        elif position == 1.0 and value >= midpoint:
            position = partial_target
            targets.loc[index] = position

    return targets


def run_cycle_reversion_backtest(
    ohlcv: pd.DataFrame,
    *,
    mode: str = "kalman",
    length: int = 100,
    bins: int = 200,
    fees: float = 0.001,
    slippage: float = 0.001,
    init_cash: float = 10_000.0,
    partial_target: float = 0.5,
):
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "vectorbt is required; install requirements-backtest.txt"
        ) from exc

    calculated, profile, summary = build_flat_chart(
        ohlcv, mode=mode, length=length, bins=bins
    )
    midpoint = summary["poc"] if np.isfinite(summary["poc"]) else summary["mu"]
    targets = build_target_allocations(
        calculated["cm_close"],
        summary["lower_percentile"],
        midpoint,
        summary["upper_percentile"],
        partial_target=partial_target,
    )
    portfolio = vbt.Portfolio.from_orders(
        calculated["Close"],
        size=targets,
        size_type="targetpercent",
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )
    return portfolio, calculated, profile, summary, targets
