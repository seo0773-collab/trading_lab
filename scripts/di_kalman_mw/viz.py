"""Plotly figures for the DI Kalman M/W dashboard page (presentation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SPLIT_FILL = {
    "train": "rgba(99, 110, 250, 0.07)",
    "validation": "rgba(255, 161, 90, 0.12)",
    "test": "rgba(239, 85, 59, 0.12)",
}
DIRECTION_COLOR = {"long": "#2ca02c", "short": "#d62728"}


def _split_shapes(index: pd.DatetimeIndex, labels: np.ndarray) -> list[dict]:
    shapes = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            shapes.append({
                "type": "rect", "xref": "x", "yref": "paper",
                "x0": index[start], "x1": index[i - 1], "y0": 0, "y1": 1,
                "fillcolor": SPLIT_FILL.get(str(labels[start]), "rgba(0,0,0,0)"),
                "line": {"width": 0}, "layer": "below",
            })
            start = i
    return shapes


def build_price_figure(
    df: pd.DataFrame, trades: pd.DataFrame, labels: np.ndarray
) -> go.Figure:
    """Close price with entry/exit markers and split shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="close",
        line={"width": 1, "color": "#636efa"},
    ))
    for direction, color in DIRECTION_COLOR.items():
        rows = trades[trades["direction"] == direction] if len(trades) else trades
        if not len(rows):
            continue
        fig.add_trace(go.Scatter(
            x=rows["entry_time"], y=rows["entry_price"], mode="markers",
            name=f"{direction} entry",
            marker={
                "symbol": "triangle-up" if direction == "long" else "triangle-down",
                "size": 10, "color": color,
            },
        ))
        fig.add_trace(go.Scatter(
            x=rows["exit_time"], y=rows["exit_price"], mode="markers",
            name=f"{direction} exit",
            marker={"symbol": "x", "size": 8, "color": color},
            text=rows["exit_reason"],
            hovertemplate="%{x}<br>exit %{y:.2f}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(
        shapes=_split_shapes(df.index, labels),
        height=420, margin={"l": 40, "r": 20, "t": 30, "b": 30},
        legend={"orientation": "h"},
    )
    return fig


def build_di_figure(
    df: pd.DataFrame,
    plus_kalman: pd.Series,
    minus_kalman: pd.Series,
    plus_extremes: list,
    minus_extremes: list,
    events: list,
) -> go.Figure:
    """+DI / -DI Kalman lines with confirmed extremes and setup events."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=plus_kalman, name="+DI kalman",
        line={"width": 1.2, "color": "#2ca02c"},
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=minus_kalman, name="-DI kalman",
        line={"width": 1.2, "color": "#d62728"},
    ))
    for name, extremes, color in (
        ("+DI extremes", plus_extremes, "#2ca02c"),
        ("-DI extremes", minus_extremes, "#d62728"),
    ):
        if not extremes:
            continue
        fig.add_trace(go.Scatter(
            x=[df.index[e.idx] for e in extremes],
            y=[e.value for e in extremes],
            mode="markers", name=name,
            marker={
                "size": 6, "color": color,
                "symbol": [
                    "circle" if e.kind == "H" else "circle-open"
                    for e in extremes
                ],
            },
            text=[f"{e.kind} (확정 bar {e.confirmation_idx})" for e in extremes],
            hovertemplate="%{x}<br>%{y:.2f}<br>%{text}<extra></extra>",
        ))
    for direction, color in DIRECTION_COLOR.items():
        evs = [ev for ev in events if ev.direction == direction]
        if not evs:
            continue
        fig.add_trace(go.Scatter(
            x=[df.index[ev.event_idx] for ev in evs],
            y=[plus_kalman.iloc[ev.event_idx] for ev in evs],
            mode="markers", name=f"{direction} setup",
            marker={"symbol": "star", "size": 11, "color": color},
            text=[
                f"+DI {ev.plus_pattern} / -DI {ev.minus_pattern}, "
                f"score {ev.pressure_score:.2f}"
                for ev in evs
            ],
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(
        height=420, margin={"l": 40, "r": 20, "t": 30, "b": 30},
        legend={"orientation": "h"},
    )
    return fig


def build_equity_figure(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["equity"], name="equity",
        line={"width": 1.4, "color": "#636efa"},
    ))
    fig.update_layout(
        shapes=_split_shapes(equity.index, equity["split"].to_numpy()),
        height=320, margin={"l": 40, "r": 20, "t": 30, "b": 30},
    )
    return fig


def split_metrics_table(metrics: dict) -> pd.DataFrame:
    """Train/validation/test metric rows for display."""
    keys = (
        "num_trades", "total_return", "profit_factor", "win_rate",
        "expectancy", "max_drawdown", "sharpe", "avg_bars_held",
    )
    rows = []
    for split_name in ("train", "validation", "test"):
        m = metrics[f"{split_name}_metrics"]
        label = f"{split_name} (in-sample)" if split_name == "train" else split_name
        rows.append({"split": label, **{k: m.get(k) for k in keys}})
    return pd.DataFrame(rows).set_index("split")
