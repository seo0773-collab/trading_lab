from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EXIT_REASON_LABELS = {
    "horizon": "보유기간 종료",
    "opposite": "반대 신호",
    "end_of_data": "데이터 종료",
    "stop_loss": "손절",
    "take_profit": "익절",
}


def account_value_series(equity: pd.Series, initial_capital: float) -> pd.Series:
    return (equity.astype(float) * float(initial_capital)).rename("account_value")


def build_trade_overview(trades: pd.DataFrame) -> dict[str, float | int]:
    overview: dict[str, float | int] = {}
    for key, direction in (("long", 1), ("short", -1)):
        selected = (
            trades[trades["direction"] == direction]
            if not trades.empty else trades
        )
        count = int(len(selected))
        closed = (
            selected["exit_time"].notna()
            if count and "exit_time" in selected
            else pd.Series(False, index=selected.index)
        )
        overview[f"{key}_trades"] = count
        overview[f"{key}_avg_return"] = (
            float(selected["net_return"].astype(float).mean())
            if count else np.nan
        )
        overview[f"{key}_close_rate"] = (
            float(closed.mean()) if count else np.nan
        )
    return overview


def build_trade_report(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    initial_capital: float,
    horizon: int,
    execution: str,
) -> pd.DataFrame:
    columns = [
        "trade_number", "side", "entry_time", "entry_price",
        "exit_time", "exit_price", "stop_loss_price", "take_profit_price",
        "exit_reason", "net_return_pct", "account_value_after", "entry_reason",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    report = pd.DataFrame(index=trades.index)
    report["trade_number"] = np.arange(1, len(trades) + 1)
    report["side"] = trades["direction"].map({1: "롱", -1: "숏"}).fillna("기타")
    report["entry_time"] = pd.to_datetime(trades["entry_time"])
    report["entry_price"] = trades["entry_price"].astype(float)
    report["exit_time"] = pd.to_datetime(trades["exit_time"])
    report["exit_price"] = trades["exit_price"].astype(float)
    report["stop_loss_price"] = _optional_numeric(trades, "stop_loss_price")
    report["take_profit_price"] = _optional_numeric(trades, "take_profit_price")
    report["exit_reason"] = trades["exit_reason"].map(
        lambda value: EXIT_REASON_LABELS.get(str(value), str(value))
    )
    report["net_return_pct"] = trades["net_return"].astype(float) * 100.0

    account = account_value_series(equity, initial_capital)
    account.index = pd.DatetimeIndex(account.index)
    exit_index = pd.DatetimeIndex(report["exit_time"])
    report["account_value_after"] = account.reindex(
        exit_index, method="ffill"
    ).to_numpy()

    execution_label = "다음 시가" if execution == "next_open" else "신호 종가"
    report["entry_reason"] = [
        _entry_reason(trade, horizon, execution_label)
        for _, trade in trades.iterrows()
    ]
    return report[columns]


def build_price_figure(
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=forecast.index, y=forecast["close"], mode="lines", name="종가",
        line={"color": "#d7e3f4", "width": 1.2},
    ))

    mid = f"price_mid_{horizon}"
    low = f"price_lo_{horizon}"
    high = f"price_hi_{horizon}"
    if mid in forecast:
        target_index, target_mid = _target_aligned(forecast[mid], horizon)
        figure.add_trace(go.Scatter(
            x=target_index, y=target_mid, mode="lines",
            name=f"{horizon}봉 예측 중앙값",
            line={"color": "#42a5f5", "width": 1.4},
        ))
        if low in forecast and high in forecast:
            _, target_low = _target_aligned(forecast[low], horizon)
            _, target_high = _target_aligned(forecast[high], horizon)
            figure.add_trace(go.Scatter(
                x=target_index, y=target_high, mode="lines", name="예측 상단",
                line={"color": "rgba(66,165,245,0.25)", "width": 0.7},
                hoverinfo="skip",
            ))
            figure.add_trace(go.Scatter(
                x=target_index, y=target_low, mode="lines", name="예측 구간",
                line={"color": "rgba(66,165,245,0.25)", "width": 0.7},
                fill="tonexty", fillcolor="rgba(66,165,245,0.10)",
                hoverinfo="skip",
            ))

    if not trades.empty:
        numbered = trades.copy()
        numbered["trade_number"] = np.arange(1, len(numbered) + 1)
        for direction, label, color, marker_symbol in [
            (1, "롱 진입", "#00c853", "triangle-up"),
            (-1, "숏 진입", "#ff9800", "triangle-down"),
        ]:
            selected = numbered[numbered["direction"] == direction]
            if selected.empty:
                continue
            figure.add_trace(go.Scatter(
                x=selected["entry_time"], y=selected["entry_price"],
                mode="markers", name=label,
                marker={"color": color, "size": 10, "symbol": marker_symbol},
                customdata=selected[["trade_number"]].to_numpy(),
                hovertemplate=(
                    f"{label} #%{{customdata[0]}}<br>%{{x|%Y-%m-%d %H:%M}}"
                    "<br>%{y:,.4f}<extra></extra>"
                ),
            ))
        figure.add_trace(go.Scatter(
            x=numbered["exit_time"], y=numbered["exit_price"],
            mode="markers", name="청산",
            marker={"color": "#ef5350", "size": 9, "symbol": "x"},
            customdata=np.column_stack([
                numbered["trade_number"], numbered["net_return"] * 100.0,
            ]),
            hovertemplate=(
                "청산 #%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M}"
                "<br>%{y:,.4f}<br>손익 %{customdata[1]:+.2f}%<extra></extra>"
            ),
        ))

    figure.update_layout(
        title=f"{symbol} 테스트 가격 및 예측 파형", height=560,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark", uirevision=f"price-{symbol}-{horizon}",
    )
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_yaxes(title_text="가격")
    return figure


def build_indicator_figure(
    forecast: pd.DataFrame,
    *,
    horizon: int,
    confidence_quantile: float,
    quantile_window: int,
) -> go.Figure:
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.58, 0.42],
        subplot_titles=("사이클/필터 파형", "예상 가격 변동폭과 진입 임계값"),
    )
    for column, label, color, width in [
        ("mult_close", "Cycle multiple", "#eceff1", 1.0),
        ("m_fast", "Fast", "#26c6da", 1.1),
        ("m_filt", "Filtered", "#7e57c2", 1.4),
        ("m_slow", "Slow", "#ffa726", 1.2),
    ]:
        if column in forecast:
            figure.add_trace(go.Scatter(
                x=forecast.index, y=forecast[column], mode="lines", name=label,
                line={"color": color, "width": width},
            ), row=1, col=1)

    mid = f"price_mid_{horizon}"
    if mid in forecast:
        edge = (forecast[mid] / forecast["close"] - 1.0).abs() * 100.0
        threshold = (
            edge.rolling(quantile_window, min_periods=max(1, quantile_window // 2))
            .quantile(confidence_quantile).shift(1)
        )
        figure.add_trace(go.Scatter(
            x=forecast.index, y=edge, mode="lines", name="예상 변동폭 %",
            line={"color": "#42a5f5", "width": 1.1},
            fill="tozeroy", fillcolor="rgba(66,165,245,0.08)",
        ), row=2, col=1)
        figure.add_trace(go.Scatter(
            x=forecast.index, y=threshold, mode="lines",
            name=f"과거 기준 {confidence_quantile:.0%} 임계값",
            line={"color": "#ef5350", "width": 1.3, "dash": "dot"},
        ), row=2, col=1)

    figure.update_layout(
        height=650, hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark", uirevision=f"indicators-{horizon}",
    )
    figure.update_yaxes(title_text="배수", row=1, col=1)
    figure.update_yaxes(title_text="변동폭 %", row=2, col=1)
    return figure


def build_account_figure(
    equity: pd.Series,
    *,
    initial_capital: float,
    symbol: str,
) -> go.Figure:
    account = account_value_series(equity, initial_capital)
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=account.index, y=account, mode="lines", name="계좌 금액",
        line={"color": "#66bb6a", "width": 1.8},
        fill="tozeroy", fillcolor="rgba(102,187,106,0.08)",
    ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, mode="lines", name="낙폭 %",
        line={"color": "#ef5350", "width": 1.0},
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
    ), secondary_y=True)
    figure.update_layout(
        title=f"{symbol} 계좌 금액 파형", height=430,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 45, "r": 45, "t": 85, "b": 40},
        template="plotly_dark", uirevision=f"account-{symbol}",
    )
    figure.update_yaxes(title_text="계좌 금액", tickprefix="$", secondary_y=False)
    figure.update_yaxes(title_text="낙폭 %", ticksuffix="%", secondary_y=True)
    return figure


def _entry_reason(trade: pd.Series, horizon: int, execution_label: str) -> str:
    direction = "상승" if float(trade["direction"]) > 0 else "하락"
    edge = _percent_value(trade.get("price_edge"))
    threshold = _percent_value(trade.get("confidence_threshold"))
    conflict = (
        " · 사이클 방향 충돌"
        if bool(trade.get("mult_price_conflict", False)) else ""
    )
    return (
        f"{horizon}봉 예상 {direction} · 예상 변동폭 {edge} · "
        f"진입 임계 {threshold} · {execution_label} 체결{conflict}"
    )


def _percent_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(numeric) else f"{numeric * 100:.2f}%"


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _target_aligned(
    values: pd.Series, horizon: int,
) -> tuple[pd.Index, np.ndarray]:
    if horizon <= 0 or horizon >= len(values):
        return values.index, values.to_numpy()
    return values.index[horizon:], values.iloc[:-horizon].to_numpy()
