"""Execution engine for confidence-filtered directional strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "asset", "signal_time", "entry_time", "entry_price",
    "exit_signal_time", "exit_time", "exit_price", "direction",
    "exit_reason", "gross_return", "fee_return", "net_return",
    "holding_bars", "position_size", "price_edge", "confidence_threshold",
    "mult_price_conflict", "split",
]


@dataclass(frozen=True)
class ExecutionConfig:
    horizon: int = 72
    short_horizon: int | None = None
    fee_bps: float = 10.0
    conf_quantile: float = 0.85
    quantile_window: int = 2000
    execution: str = "next_open"
    exit_on_opposite: bool = True
    long_only: bool = False
    edge_mult: float = 0.0
    short_size: float = 1.0


@dataclass
class ExecutionResult:
    trades: pd.DataFrame
    equity: pd.Series
    bar_returns: pd.Series


def rolling_conf_threshold(conf: pd.Series, q: float, window: int) -> pd.Series:
    """Rolling threshold using only bars strictly before the signal bar."""
    return conf.rolling(window, min_periods=window // 2).quantile(q).shift(1)


def chronological_splits(
    index: pd.Index, identification_frac: float = 0.4,
    validation_frac: float = 0.3,
) -> pd.Series:
    if not 0 < identification_frac < 1:
        raise ValueError("identification_frac must be between 0 and 1")
    if not 0 < validation_frac < 1 - identification_frac:
        raise ValueError("validation_frac leaves no test segment")
    n = len(index)
    identification_end = int(n * identification_frac)
    validation_end = int(n * (identification_frac + validation_frac))
    labels = np.full(n, "test", dtype=object)
    labels[:identification_end] = "identification"
    labels[identification_end:validation_end] = "validation"
    return pd.Series(labels, index=index, name="split")


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)


def run_execution(
    df: pd.DataFrame,
    direction: pd.Series,
    confidence: pd.Series,
    cfg: ExecutionConfig,
    *,
    asset: str,
    expected_edge: pd.Series | None = None,
    mult_direction: pd.Series | None = None,
    entry_allowed: pd.Series | None = None,
    split: pd.Series | None = None,
    entry_split: str | None = None,
) -> ExecutionResult:
    """Run close or next-open execution without overlapping positions.

    Signals are evaluated at each bar close. For ``next_open``, every entry,
    opposite-signal exit, and reversal is executed at the following bar open.
    A horizon exit is executed at the open exactly ``horizon`` bars after entry.
    An open position at the end of the sample is closed at the final close.
    """
    if cfg.execution not in {"close", "next_open"}:
        raise ValueError("execution must be 'close' or 'next_open'")
    required = {"close"}
    if cfg.execution == "next_open":
        required.add("open")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"price data missing columns: {missing}")
    if not df.index.equals(direction.index) or not df.index.equals(confidence.index):
        raise ValueError("price and signal indexes must match")

    n = len(df)
    if n == 0:
        empty = pd.Series(dtype=float, index=df.index)
        return ExecutionResult(_empty_trades(), empty, empty)

    direction = direction.reindex(df.index).fillna(0.0)
    confidence = confidence.reindex(df.index)
    expected_edge = (
        expected_edge.reindex(df.index) if expected_edge is not None
        else pd.Series(np.nan, index=df.index)
    )
    mult_direction = (
        mult_direction.reindex(df.index).fillna(0.0) if mult_direction is not None
        else pd.Series(0.0, index=df.index)
    )
    entry_allowed = (
        entry_allowed.reindex(df.index).fillna(False).astype(bool)
        if entry_allowed is not None else pd.Series(True, index=df.index)
    )
    split = (
        split.reindex(df.index) if split is not None
        else pd.Series("all", index=df.index)
    )

    threshold = rolling_conf_threshold(
        confidence, cfg.conf_quantile, cfg.quantile_window
    )
    close = df["close"].to_numpy(float)
    open_ = (
        df["open"].to_numpy(float) if cfg.execution == "next_open"
        else close
    )
    fee = cfg.fee_bps / 1e4
    bar_ret = np.zeros(n)
    trades: list[dict] = []

    pos = 0
    entry_i = -1
    entry_price = np.nan
    entry_signal_i = -1
    entry_edge = np.nan
    entry_threshold = np.nan
    entry_conflict = False
    entry_split_value = ""
    entry_size = 1.0
    pending: dict | None = None

    def signal_at(i: int) -> tuple[int, int]:
        raw = 0
        if (
            np.isfinite(threshold.iloc[i])
            and np.isfinite(confidence.iloc[i])
            and confidence.iloc[i] >= threshold.iloc[i]
            and direction.iloc[i] != 0
        ):
            raw = int(direction.iloc[i])
            if cfg.long_only and raw < 0:
                raw = 0
        entry = raw
        if entry != 0 and not entry_allowed.iloc[i]:
            entry = 0
        if entry != 0 and entry_split is not None and split.iloc[i] != entry_split:
            entry = 0
        if entry != 0 and cfg.edge_mult > 0:
            edge = expected_edge.iloc[i]
            if not (np.isfinite(edge) and edge >= cfg.edge_mult * 2 * fee):
                entry = 0
        return raw, entry

    def enter(i: int, signal_i: int, new_pos: int, px: float) -> None:
        nonlocal pos, entry_i, entry_price, entry_signal_i
        nonlocal entry_edge, entry_threshold, entry_conflict, entry_split_value
        nonlocal entry_size
        pos = new_pos
        entry_i = i
        entry_price = px
        entry_signal_i = signal_i
        entry_edge = float(expected_edge.iloc[signal_i])
        entry_threshold = float(threshold.iloc[signal_i])
        md = int(mult_direction.iloc[signal_i])
        entry_conflict = md != 0 and md != new_pos
        entry_split_value = str(split.iloc[signal_i])
        entry_size = cfg.short_size if new_pos < 0 else 1.0

    def exit_position(i: int, signal_i: int, px: float, reason: str) -> None:
        nonlocal pos
        gross = entry_size * pos * (px / entry_price - 1.0)
        fee_return = entry_size * 2 * fee
        trades.append({
            "asset": asset,
            "signal_time": df.index[entry_signal_i],
            "entry_time": df.index[entry_i],
            "entry_price": entry_price,
            "exit_signal_time": df.index[signal_i],
            "exit_time": df.index[i],
            "exit_price": px,
            "direction": pos,
            "exit_reason": reason,
            "gross_return": gross,
            "fee_return": fee_return,
            "net_return": gross - fee_return,
            "holding_bars": i - entry_i,
            "position_size": entry_size,
            "price_edge": entry_edge,
            "confidence_threshold": entry_threshold,
            "mult_price_conflict": entry_conflict,
            "split": entry_split_value,
        })
        bar_ret[i] += gross - fee_return
        pos = 0

    for i in range(n):
        if cfg.execution == "next_open":
            if pending is not None:
                if pending["exit"] and pos != 0:
                    exit_position(i, pending["signal_i"], open_[i], pending["reason"])
                if pending["entry"] != 0:
                    enter(i, pending["signal_i"], pending["entry"], open_[i])
                pending = None

        raw_signal, entry_signal = signal_at(i)

        if cfg.execution == "close":
            if pos == 0:
                if entry_signal != 0:
                    enter(i, i, entry_signal, close[i])
            else:
                held = i - entry_i
                holding_limit = (
                    cfg.short_horizon
                    if pos < 0 and cfg.short_horizon is not None
                    else cfg.horizon
                )
                opposite = (
                    cfg.exit_on_opposite
                    and raw_signal != 0
                    and raw_signal != pos
                )
                if held >= holding_limit or opposite or i == n - 1:
                    reason = "opposite" if opposite else (
                        "horizon" if held >= holding_limit else "end_of_data"
                    )
                    exit_position(i, i, close[i], reason)
                    if opposite and entry_signal != 0:
                        enter(i, i, entry_signal, close[i])
            continue

        if i == n - 1:
            continue
        if pos == 0:
            if entry_signal != 0:
                pending = {
                    "exit": False, "entry": entry_signal,
                    "signal_i": i, "reason": "",
                }
        else:
            held_after_next_open = (i + 1) - entry_i
            holding_limit = (
                cfg.short_horizon
                if pos < 0 and cfg.short_horizon is not None
                else cfg.horizon
            )
            opposite = (
                cfg.exit_on_opposite
                and raw_signal != 0
                and raw_signal != pos
            )
            horizon_exit = held_after_next_open >= holding_limit
            if opposite or horizon_exit:
                pending = {
                    "exit": True,
                    "entry": entry_signal if opposite else 0,
                    "signal_i": i,
                    "reason": "opposite" if opposite else "horizon",
                }

    if pos != 0:
        exit_position(n - 1, n - 1, close[-1], "end_of_data")

    returns = pd.Series(bar_ret, index=df.index, name="strategy_return")
    equity = (1.0 + returns).cumprod().rename("equity")
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return ExecutionResult(trade_frame, equity, returns)


def summarize_execution(
    result: ExecutionResult, bars_per_year: int,
) -> dict[str, float | int]:
    trades = result.trades
    returns = result.bar_returns
    if trades.empty:
        return {
            "trades": 0, "hit_rate": np.nan, "avg_gross_bps": np.nan,
            "avg_net_bps": np.nan, "median_net_bps": np.nan,
            "total_return": float(result.equity.iloc[-1] - 1.0),
            "sharpe": 0.0, "max_drawdown": 0.0,
            "long_trades": 0, "short_trades": 0,
        }
    sd = float(returns.std())
    sharpe = (
        float(returns.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
    )
    drawdown = result.equity / result.equity.cummax() - 1.0
    return {
        "trades": int(len(trades)),
        "hit_rate": float((trades["gross_return"] > 0).mean()),
        "avg_gross_bps": float(trades["gross_return"].mean() * 1e4),
        "avg_net_bps": float(trades["net_return"].mean() * 1e4),
        "median_net_bps": float(trades["net_return"].median() * 1e4),
        "total_return": float(result.equity.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "long_trades": int((trades["direction"] > 0).sum()),
        "short_trades": int((trades["direction"] < 0).sum()),
    }
