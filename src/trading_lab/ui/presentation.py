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
    "rebalance": "리밸런싱",
    "signal_flip": "신호 반전",
    "defense_cut": "방어 축소",
}

PRICE_COLUMNS = ("open", "high", "low", "close")

DERIVED_EDGE = "expected_edge_pct"
DERIVED_THRESHOLD = "entry_threshold_pct"

DERIVED_LABELS = {
    DERIVED_EDGE: "예상 변동폭 %",
    DERIVED_THRESHOLD: "진입 임계값 %",
}

WAVEFORM_PALETTE = (
    "#eceff1", "#26c6da", "#7e57c2", "#ffa726",
    "#42a5f5", "#ef5350", "#66bb6a", "#ec407a",
)


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

    if "entry_reason" in trades:
        report["entry_reason"] = trades["entry_reason"].astype(str).to_numpy()
    else:
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


def indicator_series(
    forecast: pd.DataFrame,
    *,
    horizon: int,
    confidence_quantile: float,
    quantile_window: int,
) -> dict[str, pd.Series]:
    """forecast 아티팩트에서 선택 가능한 보조지표 시리즈를 추출합니다.

    OHLC를 제외한 모든 숫자 컬럼이 자동으로 노출되며, 전략이
    price_mid_{horizon} 컬럼을 제공하면 파생 지표(예상 변동폭 %,
    진입 임계값 %)가 추가됩니다.
    """
    series: dict[str, pd.Series] = {}
    for column in forecast.columns:
        if column in PRICE_COLUMNS:
            continue
        values = pd.to_numeric(forecast[column], errors="coerce")
        if values.notna().any():
            series[str(column)] = values

    mid = f"price_mid_{horizon}"
    if horizon > 0 and mid in forecast and "close" in forecast:
        edge = (forecast[mid] / forecast["close"] - 1.0).abs() * 100.0
        series[DERIVED_EDGE] = edge
        series[DERIVED_THRESHOLD] = (
            edge.rolling(quantile_window, min_periods=max(1, quantile_window // 2))
            .quantile(confidence_quantile).shift(1)
        )
    return series


def build_waveform_figure(
    series_map: dict[str, pd.Series],
    *,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """선택된 보조지표를 스케일이 비슷한 것끼리 같은 패널에 겹쳐 그립니다."""
    display = {**DERIVED_LABELS, **(labels or {})}
    panes: list[dict[str, Any]] = []
    for name, values in series_map.items():
        family = "derived" if name in DERIVED_LABELS else "column"
        scale = _scale_of(values)
        for pane in panes:
            if pane["family"] != family:
                continue
            if family == "derived" or _same_scale(scale, pane["scale"]):
                pane["names"].append(name)
                break
        else:
            panes.append({"family": family, "scale": scale, "names": [name]})

    rows = max(1, len(panes))
    figure = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=min(0.1, 0.3 / rows),
    )
    color_index = 0
    for row, pane in enumerate(panes, start=1):
        for name in pane["names"]:
            values = series_map[name]
            figure.add_trace(go.Scatter(
                x=values.index, y=values, mode="lines",
                name=display.get(name, name),
                line={
                    "color": WAVEFORM_PALETTE[color_index % len(WAVEFORM_PALETTE)],
                    "width": 1.2,
                },
            ), row=row, col=1)
            color_index += 1
        if pane["family"] == "derived":
            figure.update_yaxes(title_text="%", row=row, col=1)

    figure.update_layout(
        height=max(360, 110 + 230 * rows), hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark",
        uirevision="waveform-" + "|".join(series_map),
    )
    return figure


def build_price_indicator_figure(
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
    series_map: dict[str, pd.Series],
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """가격과 보조지표를 TradingView 형태의 공유 X축 패널로 구성합니다."""
    price = build_price_figure(
        forecast, trades, symbol=symbol, horizon=horizon,
    )
    waveform = (
        build_waveform_figure(series_map, labels=labels)
        if series_map else None
    )
    indicator_rows = len({
        trace.yaxis or "y" for trace in waveform.data
    }) if waveform else 0
    rows = 1 + indicator_rows
    row_heights = [0.62] + [0.38 / indicator_rows] * indicator_rows if indicator_rows else [1.0]
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025 if indicator_rows else 0.0,
        row_heights=row_heights,
    )

    for trace in price.data:
        figure.add_trace(trace, row=1, col=1)
    figure.update_yaxes(title_text="가격", row=1, col=1)

    if waveform:
        axis_rows: dict[str, int] = {}
        for trace in waveform.data:
            axis = trace.yaxis or "y"
            row = axis_rows.setdefault(axis, len(axis_rows) + 2)
            figure.add_trace(trace, row=row, col=1)
        for axis, row in axis_rows.items():
            axis_number = 1 if axis == "y" else int(axis[1:])
            source_axis = waveform.layout[
                "yaxis" if axis_number == 1 else f"yaxis{axis_number}"
            ]
            if source_axis.title and source_axis.title.text:
                figure.update_yaxes(
                    title_text=source_axis.title.text, row=row, col=1,
                )

    figure.update_layout(
        title=f"{symbol} 가격 및 보조지표",
        height=560 + 210 * indicator_rows,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark",
        uirevision=f"market-{symbol}-{horizon}-{'|'.join(series_map)}",
    )
    figure.update_xaxes(rangeslider_visible=False)
    return figure


def _scale_of(values: pd.Series) -> float:
    finite = np.abs(values.to_numpy(dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return float("nan")
    return float(np.log10(np.median(finite)))


def _same_scale(a: float, b: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)
    return abs(a - b) <= 1.0


def build_account_figure(
    equity: pd.Series,
    *,
    initial_capital: float,
    symbol: str,
    benchmark_price: pd.Series | None = None,
) -> go.Figure:
    account = account_value_series(equity, initial_capital)
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    strategy_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=account.index, y=account, mode="lines",
        name=f"전략 계좌 ({strategy_return:+.2%})",
        line={"color": "#66bb6a", "width": 1.8},
    ), secondary_y=False)
    if benchmark_price is not None:
        benchmark = pd.to_numeric(benchmark_price, errors="coerce").reindex(
            account.index
        ).ffill().bfill()
        benchmark = benchmark.where(benchmark > 0)
        if benchmark.notna().any():
            first_price = float(benchmark.dropna().iloc[0])
            buy_hold_equity = benchmark / first_price
            buy_hold_account = buy_hold_equity * float(initial_capital)
            buy_hold_return = float(buy_hold_equity.dropna().iloc[-1] - 1.0)
            figure.add_trace(go.Scatter(
                x=buy_hold_account.index,
                y=buy_hold_account,
                mode="lines",
                name=f"Buy & Hold ({buy_hold_return:+.2%})",
                line={"color": "#42a5f5", "width": 1.6, "dash": "dash"},
            ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, mode="lines", name="낙폭 %",
        line={"color": "#ef5350", "width": 1.0},
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
    ), secondary_y=True)
    figure.update_layout(
        title=f"{symbol} 전략 vs Buy & Hold", height=430,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 45, "r": 45, "t": 85, "b": 40},
        template="plotly_dark", uirevision=f"account-{symbol}",
    )
    figure.update_yaxes(title_text="계좌 금액", tickprefix="$", secondary_y=False)
    figure.update_yaxes(title_text="낙폭 %", ticksuffix="%", secondary_y=True)
    return figure


# --- 범용 보조 패널 (전략별 extras 선언적 렌더) ------------------------------
# 결과 화면이 전용 위젯으로 그리는 표준 아티팩트. 이외의 아티팩트는 전략이
# config.dashboard.panels 로 선언하거나 자동 발견되어 "보조 패널"로 노출된다.
# 이렇게 두면 전략마다 app.py 를 고치지 않고 config 만으로 add/delete 할 수 있다.
_CORE_ARTIFACT_KINDS = frozenset({
    "manifest", "config", "forecast", "forecast_metadata", "trades",
    "equity", "account_value", "trade_report", "metrics", "report",
    "equity_chart", "error",
})


def available_extra_kinds(run: dict[str, Any]) -> list[str]:
    """표준 아티팩트가 아닌(=보조 패널 후보) 아티팩트 kind 목록(등록 순서)."""
    seen: list[str] = []
    for artifact in run.get("artifacts", []) or []:
        kind = str(artifact.get("kind"))
        if kind not in _CORE_ARTIFACT_KINDS and kind not in seen:
            seen.append(kind)
    return seen


def resolve_extra_panels(
    dashboard_config: dict[str, Any], available_kinds: list[str],
) -> list[dict[str, Any]]:
    """config 선언과 실제 존재하는 아티팩트를 합쳐 패널 스펙 목록을 만든다.

    선언(``dashboard.panels``)에 ``label``/``type``(table|scatter|bar|line)/축
    (``x``/``y``)/``default`` 를 줄 수 있다. 선언이 없는 아티팩트는 표(table)로
    자동 노출돼, 새 extras를 추가해도 별도 코드 없이 화면에 뜬다.
    """
    declared = {
        str(panel.get("kind")): dict(panel)
        for panel in (dashboard_config.get("panels") or [])
        if panel.get("kind")
    }
    panels: list[dict[str, Any]] = []
    for kind in available_kinds:
        spec = declared.get(kind, {})
        panels.append({
            "kind": kind,
            "type": str(spec.get("type", "table")),
            "label": str(spec.get("label", kind)),
            "x": spec.get("x"),
            "y": spec.get("y"),
            "default": bool(spec.get("default", True)),
        })
    return panels


def build_scatter_figure(
    frame: pd.DataFrame, x: str, y: str, *, label: str,
) -> go.Figure:
    """예측 vs 실제 같은 산점도 + y=x 기준선(가능 시)."""
    fx = pd.to_numeric(frame[x], errors="coerce")
    fy = pd.to_numeric(frame[y], errors="coerce")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=fx, y=fy, mode="markers", name=label,
        marker=dict(size=7, color=WAVEFORM_PALETTE[1], opacity=0.75),
    ))
    finite = fx.replace([np.inf, -np.inf], np.nan).dropna()
    finite_y = fy.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) and len(finite_y):
        lo = float(min(finite.min(), finite_y.min()))
        hi = float(max(finite.max(), finite_y.max()))
        figure.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
            line=dict(color="#888", dash="dash"),
        ))
    figure.update_layout(
        title=label, height=420, template="plotly_dark",
        xaxis_title=x, yaxis_title=y, showlegend=True,
    )
    return figure


def build_bar_figure(
    frame: pd.DataFrame, x: str, y: str, *, label: str,
) -> go.Figure:
    """팩터 민감도 같은 막대 그래프(부호별 색)."""
    fy = pd.to_numeric(frame[y], errors="coerce")
    colors = [
        WAVEFORM_PALETTE[6] if v >= 0 else WAVEFORM_PALETTE[5]
        for v in fy.fillna(0.0)
    ]
    figure = go.Figure(go.Bar(x=frame[x].astype(str), y=fy, marker_color=colors))
    figure.update_layout(
        title=label, height=420, template="plotly_dark",
        xaxis_title=x, yaxis_title=y,
    )
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
