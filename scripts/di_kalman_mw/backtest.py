"""Backtest engine implementing the fill rules of plan 12 / 12A.

- Entry at next-bar open after the signal bar, slippage applied adversely.
- Stop / TP touch decided on bar high/low; stop wins a same-bar tie.
- Gap-through opens fill at the open price.
- Trailing stop level is updated on bar close and acts as an intrabar
  stop from the next bar onward.
- Opposite-pattern exit and time stop are evaluated on close and filled
  at the next bar open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig, ExitConfig

TRADE_COLUMNS = [
    "signal_time", "entry_time", "exit_time", "direction",
    "entry_price", "exit_price", "stop_price", "take_profit_price",
    "size", "pnl", "pnl_pct", "bars_held", "exit_reason",
    "entry_variant", "pressure_score", "pressure_rr_factor",
    "pressure_aligned", "raw_expected_value",
    "pressure_adjusted_expected_value", "tier", "setup_shape",
    "p_continuation", "similarity_expected_return",
    "similarity_ev_lower_bound", "similarity_effective_n",
    "similarity_confidence", "similarity_fallback",
    "mfe", "mae", "split",
]


def stop_take_profit(
    direction: str,
    entry_price: float,
    sig_idx: int,
    df: pd.DataFrame,
    atr_series: pd.Series,
    exits: ExitConfig,
    pressure_rr_factor: float,
    p_continuation: float = float("nan"),
) -> tuple[float, float | None]:
    """Stop / take-profit prices from signal-bar information only.

    For pressure_rr take-profits, the target is additionally scaled by the
    pattern's continuation probability when exits.continuation_rr is set
    (plan 10): a higher P(continuation) widens the target, a lower one
    tightens it. NaN continuation leaves the target unchanged.
    """
    a = float(atr_series.iloc[sig_idx])
    if not np.isfinite(a) or a <= 0:
        return float("nan"), None
    d = 1.0 if direction == "long" else -1.0
    if exits.stop_type == "swing":
        lo = max(0, sig_idx - exits.swing_lookback + 1)
        if d > 0:
            stop = float(df["low"].iloc[lo:sig_idx + 1].min()) - a * exits.swing_buffer_mult
        else:
            stop = float(df["high"].iloc[lo:sig_idx + 1].max()) + a * exits.swing_buffer_mult
    else:
        stop = entry_price - d * a * exits.atr_stop_mult
    risk = d * (entry_price - stop)
    if not np.isfinite(risk) or risk <= 0:
        return float("nan"), None
    if exits.tp_type == "fixed_r":
        tp = entry_price + d * risk * exits.rr_target
    elif exits.tp_type == "pressure_rr":
        rr = exits.base_rr * pressure_rr_factor
        if exits.continuation_rr and np.isfinite(p_continuation):
            rr *= float(np.clip(0.5 + p_continuation, 0.5, 1.5))
        rr = float(np.clip(rr, exits.rr_min, exits.rr_max))
        tp = entry_price + d * risk * rr
    else:
        tp = None
    return stop, tp


def run_backtest(
    df: pd.DataFrame,
    atr_series: pd.Series,
    signals: list,
    events: list,
    exits: ExitConfig,
    costs: CostConfig,
    direction_mode: str,
    split_labels: np.ndarray,
    entry_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    index = df.index
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_values = atr_series.to_numpy(dtype=float)

    allowed = {
        "both": ("long", "short"),
        "long": ("long",),
        "short": ("short",),
    }[direction_mode]
    sig_at: dict[int, object] = {}
    for s in signals:
        if s.signal in allowed and s.signal_idx not in sig_at:
            sig_at[s.signal_idx] = s
    event_dirs_at: dict[int, set[str]] = {}
    for ev in events:
        event_dirs_at.setdefault(ev.event_idx, set()).add(ev.direction)

    fee_round_trip = 2.0 * costs.fee
    slip = costs.slippage

    bar_ret = np.zeros(n)
    trades: list[dict] = []

    pos = 0
    sig = None
    entry_i = -1
    entry_px = np.nan
    ref = np.nan  # last mark-to-market price for bar returns
    stop = np.nan
    init_stop = np.nan
    tp: float | None = None
    hh_close = -np.inf
    ll_close = np.inf
    mfe = 0.0
    mae = 0.0
    pending_entry = None
    pending_exit = ""

    def close_trade(i: int, raw_px: float, reason: str) -> None:
        nonlocal pos
        px = raw_px * (1.0 - pos * slip)
        gross = pos * (px / entry_px - 1.0)
        net = gross - fee_round_trip
        bar_ret[i] += pos * (px / ref - 1.0) - fee_round_trip
        trades.append({
            "signal_time": index[sig.signal_idx],
            "entry_time": index[entry_i],
            "exit_time": index[i],
            "direction": "long" if pos > 0 else "short",
            "entry_price": entry_px,
            "exit_price": px,
            "stop_price": init_stop,
            "take_profit_price": tp if tp is not None else np.nan,
            "size": 1.0,
            "pnl": net,
            "pnl_pct": net,
            "bars_held": i - entry_i,
            "exit_reason": reason,
            "entry_variant": entry_variant,
            "pressure_score": sig.event.pressure_score,
            "pressure_rr_factor": sig.event.pressure_rr_factor,
            "pressure_aligned": bool(sig.event.pressure_aligned),
            "raw_expected_value": sig.raw_expected_value,
            "pressure_adjusted_expected_value": sig.pressure_adjusted_expected_value,
            "tier": getattr(sig.event, "tier", ""),
            "setup_shape": getattr(sig.event, "setup_shape", ""),
            "p_continuation": getattr(sig, "continuation_score", float("nan")),
            "similarity_expected_return": getattr(
                sig, "similarity_expected_return", float("nan")
            ),
            "similarity_ev_lower_bound": getattr(
                sig, "similarity_ev_lower_bound", float("nan")
            ),
            "similarity_effective_n": getattr(
                sig, "similarity_effective_n", float("nan")
            ),
            "similarity_confidence": getattr(
                sig, "similarity_confidence", float("nan")
            ),
            "similarity_fallback": getattr(
                sig, "similarity_fallback", ""
            ),
            "mfe": mfe,
            "mae": mae,
            "split": str(split_labels[sig.signal_idx]),
        })
        pos = 0

    for i in range(n):
        split_end = (
            i == n - 1 or split_labels[i + 1] != split_labels[i]
        )
        # 1) scheduled next-open exits (plan 12A.5)
        if pos != 0 and pending_exit:
            close_trade(i, open_[i], pending_exit)
            pending_exit = ""
        # 2) pending entry at this bar's open (plan 12A.1)
        if pending_entry is not None:
            s = pending_entry
            pending_entry = None
            if pos == 0:
                d = 1 if s.signal == "long" else -1
                e_px = open_[i] * (1.0 + d * slip)
                stop_px, tp_px = stop_take_profit(
                    s.signal, e_px, s.signal_idx, df, atr_series, exits,
                    s.event.pressure_rr_factor,
                    getattr(s, "continuation_score", float("nan")),
                )
                if np.isfinite(stop_px):
                    pos = d
                    sig = s
                    entry_i = i
                    entry_px = ref = e_px
                    stop = init_stop = stop_px
                    tp = tp_px
                    hh_close = -np.inf
                    ll_close = np.inf
                    mfe = mae = 0.0
        # 3) intrabar stop / TP, stop first (plan 12A.2-4)
        if pos > 0:
            if low[i] <= stop:
                close_trade(i, open_[i] if open_[i] <= stop else stop, "stop")
            elif tp is not None and high[i] >= tp:
                close_trade(i, open_[i] if open_[i] >= tp else tp, "take_profit")
        elif pos < 0:
            if high[i] >= stop:
                close_trade(i, open_[i] if open_[i] >= stop else stop, "stop")
            elif tp is not None and low[i] <= tp:
                close_trade(i, open_[i] if open_[i] <= tp else tp, "take_profit")
        # 4) close-stage updates and close-evaluated exits (plan 12A.5)
        if pos != 0:
            bar_ret[i] += pos * (close[i] / ref - 1.0)
            ref = close[i]
            hh_close = max(hh_close, close[i])
            ll_close = min(ll_close, close[i])
            if pos > 0:
                mfe = max(mfe, high[i] / entry_px - 1.0)
                mae = min(mae, low[i] / entry_px - 1.0)
            else:
                mfe = max(mfe, 1.0 - low[i] / entry_px)
                mae = min(mae, 1.0 - high[i] / entry_px)
            if exits.trailing and np.isfinite(atr_values[i]):
                if pos > 0:
                    stop = max(stop, hh_close - atr_values[i] * exits.trail_mult)
                else:
                    stop = min(stop, ll_close + atr_values[i] * exits.trail_mult)
            opposite = "short" if pos > 0 else "long"
            if exits.opposite_exit and opposite in event_dirs_at.get(i, ()):
                pending_exit = "opposite_pattern"
            elif i - entry_i >= exits.max_hold_bars:
                pending_exit = "time_stop"
            if split_end:
                close_trade(
                    i,
                    close[i],
                    "end_of_data" if i == n - 1 else "split_boundary",
                )
                pending_exit = ""
        if split_end:
            pending_entry = None
        # 5) accept a signal for entry at the next bar open
        s = sig_at.get(i)
        if not split_end and s is not None and (pos == 0 or pending_exit):
            pending_entry = s

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity_df = pd.DataFrame(
        {
            "bar_return": bar_ret,
            "equity": np.cumprod(1.0 + bar_ret),
            "split": split_labels,
        },
        index=index,
    )
    return trades_df, equity_df
