"""Plotly figures and signal tables for the Streamlit wave viewer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def classify_target_events(targets: pd.Series) -> pd.DataFrame:
    """Convert target allocations into entry, partial-exit, and exit events."""
    events = []
    previous = 0.0
    for timestamp, target in targets.dropna().items():
        target = float(target)
        if previous == 0.0 and target > 0.0:
            event = "BUY"
        elif target == 0.0 and previous > 0.0:
            event = "EXIT"
        elif 0.0 < target < previous:
            event = "PARTIAL"
        elif target > previous:
            event = "ADD"
        else:
            event = "REBALANCE"
        events.append({"Timestamp": timestamp, "Event": event, "Target": target})
        previous = target
    return pd.DataFrame(events, columns=["Timestamp", "Event", "Target"])


def build_signal_table(
    calculated: pd.DataFrame, targets: pd.Series, orders: pd.DataFrame
) -> pd.DataFrame:
    events = classify_target_events(targets)
    if events.empty:
        return events

    events["Close"] = calculated["Close"].reindex(events["Timestamp"]).to_numpy()
    events["Cycle Multiple"] = calculated["cm_close"].reindex(
        events["Timestamp"]
    ).to_numpy()
    if not orders.empty:
        order_view = orders[["Timestamp", "Price", "Size", "Fees", "Side"]].copy()
        order_view["Timestamp"] = pd.to_datetime(order_view["Timestamp"])
        events = events.merge(order_view, on="Timestamp", how="left")
    return events


def _add_signal_markers(
    figure: go.Figure,
    calculated: pd.DataFrame,
    events: pd.DataFrame,
    row: int,
) -> None:
    styles = {
        "BUY": ("triangle-up", "#00c853", "Buy"),
        "PARTIAL": ("diamond", "#ffb300", "Partial exit"),
        "EXIT": ("triangle-down", "#ff1744", "Full exit"),
        "ADD": ("triangle-up", "#00b0ff", "Add"),
    }
    for event_name, (symbol, color, label) in styles.items():
        selected = events[events["Event"] == event_name]
        if selected.empty:
            continue
        timestamps = pd.to_datetime(selected["Timestamp"])
        prices = calculated["Close"].reindex(timestamps)
        figure.add_trace(
            go.Scatter(
                x=timestamps,
                y=prices,
                mode="markers",
                name=label,
                marker={"symbol": symbol, "size": 13, "color": color, "line": {"width": 1, "color": "#111"}},
                customdata=np.column_stack(
                    [selected["Target"].to_numpy(), selected["Cycle Multiple"].to_numpy()]
                ),
                hovertemplate=(
                    f"{label}<br>%{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}"
                    "<br>Target: %{customdata[0]:.0%}"
                    "<br>Cycle: %{customdata[1]:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def build_wave_figure(
    calculated: pd.DataFrame,
    summary: dict[str, float],
    targets: pd.Series,
    portfolio,
    events: pd.DataFrame,
) -> go.Figure:
    value = portfolio.value()
    drawdown = (value / value.cummax() - 1.0) * 100.0
    allocation = targets.ffill().fillna(0.0) * 100.0

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.48, 0.24, 0.12, 0.16],
        subplot_titles=(
            "Price, base cycle, and strategy signals",
            "Cycle multiple wave",
            "Target allocation",
            "Portfolio equity and drawdown",
        ),
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
    )
    figure.add_trace(
        go.Candlestick(
            x=calculated.index,
            open=calculated["Open"],
            high=calculated["High"],
            low=calculated["Low"],
            close=calculated["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=calculated.index,
            y=calculated["base_cycle"],
            mode="lines",
            name="Base cycle",
            line={"color": "#29b6f6", "width": 1.7},
        ),
        row=1,
        col=1,
    )
    _add_signal_markers(figure, calculated, events, row=1)

    figure.add_trace(
        go.Scatter(
            x=calculated.index,
            y=calculated["cm_close"],
            mode="lines",
            name="Cycle multiple close",
            line={"color": "#ab47bc", "width": 1.4},
            fill="tozeroy",
            fillcolor="rgba(171,71,188,0.08)",
        ),
        row=2,
        col=1,
    )
    levels = [
        (summary["lower_percentile"], "Lower", "#00c853"),
        (summary["poc"], "POC", "#ffb300"),
        (summary["mu"], "Gaussian mu", "#29b6f6"),
        (summary["upper_percentile"], "Upper", "#ff1744"),
    ]
    for level, label, color in levels:
        if np.isfinite(level):
            figure.add_hline(
                y=level,
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top left",
                row=2,
                col=1,
            )

    figure.add_trace(
        go.Scatter(
            x=allocation.index,
            y=allocation,
            mode="lines",
            line_shape="hv",
            name="Target allocation %",
            line={"color": "#26a69a", "width": 1.6},
            fill="tozeroy",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=value.index,
            y=value,
            mode="lines",
            name="Equity",
            line={"color": "#42a5f5", "width": 1.7},
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            name="Drawdown %",
            line={"color": "#ef5350", "width": 1.1},
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    figure.update_layout(
        height=1180,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 55, "r": 55, "t": 90, "b": 40},
        uirevision="wave-viewer",
    )
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Multiple", row=2, col=1)
    figure.update_yaxes(title_text="Allocation %", range=[-5, 105], row=3, col=1)
    figure.update_yaxes(title_text="Equity", row=4, col=1, secondary_y=False)
    figure.update_yaxes(title_text="DD %", row=4, col=1, secondary_y=True)
    return figure


def build_profile_figure(profile: pd.DataFrame, summary: dict[str, float]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=profile["value"],
            y=profile["mult"],
            orientation="h",
            name="Observed volume",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=profile["expected"],
            y=profile["mult"],
            mode="lines",
            name="Gaussian expected",
            line={"color": "#ffb300", "width": 2},
        )
    )
    for level, label, color in [
        (summary["lower_percentile"], "Lower", "#00c853"),
        (summary["poc"], "POC", "#ff9800"),
        (summary["upper_percentile"], "Upper", "#ff1744"),
    ]:
        if np.isfinite(level):
            figure.add_hline(y=level, line_dash="dot", line_color=color, annotation_text=label)
    figure.update_layout(
        height=700,
        xaxis_title="Volume",
        yaxis_title="Cycle multiple",
        legend={"orientation": "h"},
    )
    return figure
