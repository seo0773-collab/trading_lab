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
    "hedge_off": "헤지 해제",
    "trailing_take_profit": "트레일링 익절",
    "poc_target": "POC 도달",
    "va_stop": "VA 경계 손절",
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

# ── 다크 미니멀 공통 테마 (모든 패널 차트에 일관 적용) ──────────────────
ACCENT = "#4C9AFF"
_THEME_BG = "rgba(0,0,0,0)"          # 투명 → Streamlit 다크 캔버스에 녹아듦
_THEME_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
_THEME_GRID = "rgba(255,255,255,0.06)"
_THEME_ZERO = "rgba(255,255,255,0.18)"
_THEME_COLORWAY = (
    "#4C9AFF", "#26c6da", "#a78bfa", "#ffa726",
    "#66bb6a", "#ef5350", "#ec407a", "#7e57c2",
)


def _apply_theme(figure: go.Figure) -> go.Figure:
    """다크 미니멀 톤을 모든 차트에 통일 적용한다. 투명 배경으로 앱 캔버스에
    녹아들고 폰트·그리드·팔레트·호버를 일관시킨다. 개별 빌더가 설정한
    title·axis title·height·legend 위치 등 명시값은 부분 업데이트로 보존된다."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=_THEME_BG,
        plot_bgcolor=_THEME_BG,
        colorway=list(_THEME_COLORWAY),
        font=dict(family=_THEME_FONT, color="#C9D1D9", size=13),
        title_font=dict(family=_THEME_FONT, color="#FAFAFA", size=16),
        hoverlabel=dict(
            bgcolor="#1A1D26", bordercolor="rgba(255,255,255,0.12)",
            font=dict(family=_THEME_FONT, size=12, color="#FAFAFA"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    figure.update_xaxes(gridcolor=_THEME_GRID, zerolinecolor=_THEME_ZERO,
                        linecolor=_THEME_GRID, showline=False)
    figure.update_yaxes(gridcolor=_THEME_GRID, zerolinecolor=_THEME_ZERO,
                        linecolor=_THEME_GRID, showline=False)
    return figure


def account_value_series(
    equity: pd.Series, initial_capital: float, *,
    monthly_contribution: float = 0.0,
) -> pd.Series:
    """정규화 NAV(1.0 기준)를 실제 계좌 평가액 시계열로 환산한다.

    거치식(기본, monthly_contribution=0): 초기 일시금 전액을 t0에 투입 →
    equity×capital. 적립식(monthly_contribution>0): 초기 일시금 + 매월 첫
    거래일마다 정액을 추가 투입하고, 각 투입분이 그 시점 이후 NAV 성장률로
    복리 성장한다(money-weighted). NaN NAV 구간(벤치 워밍업 등)은 투입 무효."""
    eq = pd.Series(equity).astype(float)
    if monthly_contribution <= 0 or eq.empty:
        return (eq * float(initial_capital)).rename("account_value")
    contrib = pd.Series(0.0, index=eq.index)
    contrib.iloc[0] += float(initial_capital)
    monthly_first = ~eq.index.to_period("M").duplicated()
    contrib.loc[monthly_first] += float(monthly_contribution)
    # NaN NAV 시점(벤치 워밍업)의 투입은 무효(0)로 두어 cumsum 오염을 막는다.
    added = (contrib / eq.replace(0.0, np.nan)).fillna(0.0)
    return (added.cumsum() * eq).rename("account_value")


def total_invested(
    equity: pd.Series, initial_capital: float, monthly_contribution: float
) -> float:
    """적립식 누적 투입원금(거치식이면 초기 일시금)."""
    if monthly_contribution <= 0 or len(equity) == 0:
        return float(initial_capital)
    months = int((~pd.Series(equity).index.to_period("M").duplicated()).sum())
    return float(initial_capital) + float(monthly_contribution) * months


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
    if "symbol" in trades:
        columns.insert(1, "symbol")
    if trades.empty:
        return pd.DataFrame(columns=columns)

    report = pd.DataFrame(index=trades.index)
    report["trade_number"] = np.arange(1, len(trades) + 1)
    if "symbol" in trades:
        report["symbol"] = trades["symbol"].astype(str).to_numpy()
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
    heatmap_overlay: go.Heatmap | None = None,
) -> go.Figure:
    portfolio_view = forecast_is_portfolio(forecast)
    close_label = "포트폴리오 NAV" if portfolio_view else "종가"
    y_title = "NAV" if portfolio_view else "가격"
    figure = go.Figure()
    # 청산 히트맵은 가장 먼저(맨 아래 레이어) 깔아 가격선·마커가 위로 오게 한다.
    if heatmap_overlay is not None:
        figure.add_trace(heatmap_overlay)
    figure.add_trace(go.Scatter(
        x=forecast.index, y=forecast["close"], mode="lines", name=close_label,
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
            entry_y = (
                _values_at_times(forecast["close"], selected["entry_time"])
                if portfolio_view else selected["entry_price"]
            )
            if portfolio_view:
                customdata = np.column_stack([
                    selected["trade_number"],
                    selected.get("symbol", pd.Series("", index=selected.index)).astype(str),
                    selected["entry_price"].astype(float),
                ])
                hovertemplate = (
                    f"{label} #%{{customdata[0]}}<br>%{{x|%Y-%m-%d %H:%M}}"
                    "<br>%{customdata[1]} 진입가 %{customdata[2]:,.4f}"
                    "<br>NAV %{y:,.2f}<extra></extra>"
                )
            else:
                customdata = selected[["trade_number"]].to_numpy()
                hovertemplate = (
                    f"{label} #%{{customdata[0]}}<br>%{{x|%Y-%m-%d %H:%M}}"
                    "<br>%{y:,.4f}<extra></extra>"
                )
            figure.add_trace(go.Scatter(
                x=selected["entry_time"], y=entry_y,
                mode="markers", name=label,
                marker={"color": color, "size": 10, "symbol": marker_symbol},
                customdata=customdata,
                hovertemplate=hovertemplate,
            ))
        exit_y = (
            _values_at_times(forecast["close"], numbered["exit_time"])
            if portfolio_view else numbered["exit_price"]
        )
        if portfolio_view:
            exit_customdata = np.column_stack([
                numbered["trade_number"],
                numbered.get("symbol", pd.Series("", index=numbered.index)).astype(str),
                numbered["exit_price"].astype(float),
                numbered["net_return"] * 100.0,
            ])
            exit_hovertemplate = (
                "청산 #%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M}"
                "<br>%{customdata[1]} 청산가 %{customdata[2]:,.4f}"
                "<br>NAV %{y:,.2f}<br>손익 %{customdata[3]:+.2f}%<extra></extra>"
            )
        else:
            exit_customdata = np.column_stack([
                numbered["trade_number"], numbered["net_return"] * 100.0,
            ])
            exit_hovertemplate = (
                "청산 #%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M}"
                "<br>%{y:,.4f}<br>손익 %{customdata[1]:+.2f}%<extra></extra>"
            )
        figure.add_trace(go.Scatter(
            x=numbered["exit_time"], y=exit_y,
            mode="markers", name="청산",
            marker={"color": "#ef5350", "size": 9, "symbol": "x"},
            customdata=exit_customdata,
            hovertemplate=exit_hovertemplate,
        ))

    figure.update_layout(
        title=f"{symbol} {'포트폴리오' if portfolio_view else '테스트 가격'} 및 예측 파형",
        height=560,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark", uirevision=f"price-{symbol}-{horizon}",
    )
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_yaxes(title_text=y_title)
    return _apply_theme(figure)


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
    return _apply_theme(figure)


def build_price_indicator_figure(
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
    series_map: dict[str, pd.Series],
    labels: dict[str, str] | None = None,
    heatmap_overlay: go.Heatmap | None = None,
) -> go.Figure:
    """가격과 보조지표를 TradingView 형태의 공유 X축 패널로 구성합니다."""
    price = build_price_figure(
        forecast, trades, symbol=symbol, horizon=horizon,
        heatmap_overlay=heatmap_overlay,
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
    figure.update_yaxes(
        title_text="NAV" if forecast_is_portfolio(forecast) else "가격",
        row=1,
        col=1,
    )

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
        title=f"{symbol} {'포트폴리오' if forecast_is_portfolio(forecast) else '가격'} 및 보조지표",
        height=560 + 210 * indicator_rows,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark",
        uirevision=f"market-{symbol}-{horizon}-{'|'.join(series_map)}",
    )
    figure.update_xaxes(rangeslider_visible=False)
    return _apply_theme(figure)


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
    benchmark_equity: pd.Series | None = None,
    monthly_contribution: float = 0.0,
    currency_symbol: str = "$",
) -> go.Figure:
    account = account_value_series(
        equity, initial_capital, monthly_contribution=monthly_contribution)
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    # 적립식은 단순 NAV 배수가 아니라 평가액/누적투입원금 기준으로 손익을 본다.
    invested = total_invested(equity, initial_capital, monthly_contribution)
    dca = monthly_contribution > 0
    strategy_return = (
        float(account.iloc[-1] / invested - 1.0) if dca and invested
        else float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    )
    name_suffix = " · 적립식" if dca else ""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=account.index, y=account, mode="lines",
        name=f"전략 계좌{name_suffix} ({strategy_return:+.2%})",
        line={"color": "#66bb6a", "width": 1.8},
    ), secondary_y=False)
    buy_hold_equity = _benchmark_equity(
        account.index,
        benchmark_price=benchmark_price,
        benchmark_equity=benchmark_equity,
    )
    if buy_hold_equity is not None and buy_hold_equity.notna().any():
        buy_hold_account = account_value_series(
            buy_hold_equity, initial_capital,
            monthly_contribution=monthly_contribution)
        buy_hold_return = (
            float(buy_hold_account.dropna().iloc[-1] / invested - 1.0)
            if dca and invested
            else float(buy_hold_equity.dropna().iloc[-1] - 1.0)
        )
        figure.add_trace(go.Scatter(
            x=buy_hold_account.index,
            y=buy_hold_account,
            mode="lines",
            name=f"Buy & Hold ({buy_hold_return:+.2%})",
            line={"color": "#42a5f5", "width": 1.6, "dash": "dash"},
        ), secondary_y=False)
    if dca:  # 누적 투입원금 기준선(계단형)
        contrib = pd.Series(0.0, index=account.index)
        contrib.iloc[0] += float(initial_capital)
        contrib.loc[~account.index.to_period("M").duplicated()] += float(
            monthly_contribution)
        figure.add_trace(go.Scatter(
            x=account.index, y=contrib.cumsum(), mode="lines",
            name="누적 투입원금", line={"color": "#90a4ae", "width": 1.0, "dash": "dot"},
        ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, mode="lines", name="낙폭 %",
        line={"color": "#ef5350", "width": 1.0},
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
    ), secondary_y=True)
    has_benchmark = buy_hold_equity is not None and buy_hold_equity.notna().any()
    title = (
        f"{symbol} 전략 vs Buy & Hold"
        if has_benchmark else f"{symbol} 전략 계좌"
    )
    figure.update_layout(
        title=title, height=430,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 45, "r": 45, "t": 85, "b": 40},
        template="plotly_dark", uirevision=f"account-{symbol}",
    )
    figure.update_yaxes(
        title_text="계좌 금액", tickprefix=currency_symbol, secondary_y=False)
    figure.update_yaxes(title_text="낙폭 %", ticksuffix="%", secondary_y=True)
    return _apply_theme(figure)


def _benchmark_equity(
    index: pd.Index,
    *,
    benchmark_price: pd.Series | None,
    benchmark_equity: pd.Series | None,
) -> pd.Series | None:
    if benchmark_equity is not None:
        benchmark = pd.to_numeric(benchmark_equity, errors="coerce").reindex(
            index
        ).ffill().bfill()
        benchmark = benchmark.where(benchmark > 0)
        if benchmark.notna().any():
            first_value = float(benchmark.dropna().iloc[0])
            return benchmark / first_value
    if benchmark_price is None:
        return None
    benchmark = pd.to_numeric(benchmark_price, errors="coerce").reindex(
        index
    ).ffill().bfill()
    benchmark = benchmark.where(benchmark > 0)
    if not benchmark.notna().any():
        return None
    first_price = float(benchmark.dropna().iloc[0])
    return benchmark / first_price


def forecast_is_portfolio(forecast: pd.DataFrame) -> bool:
    """포트폴리오 전략은 close에 종목 가격 대신 NAV를 담아 공통 계약을 맞춘다."""
    portfolio_columns = {"stock_exposure", "cash_ratio", "n_holdings"}
    return portfolio_columns.issubset(set(forecast.columns))


def _values_at_times(series: pd.Series, times: pd.Series) -> np.ndarray:
    source = pd.Series(series).copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index))
    lookup = pd.DatetimeIndex(pd.to_datetime(times))
    return source.reindex(lookup, method="ffill").to_numpy(dtype=float)


# --- 범용 보조 패널 (전략별 extras 선언적 렌더) ------------------------------
# 결과 화면이 전용 위젯으로 그리는 표준 아티팩트. 이외의 아티팩트는 전략이
# config.dashboard.panels 로 선언하거나 자동 발견되어 "보조 패널"로 노출된다.
# 이렇게 두면 전략마다 app.py 를 고치지 않고 config 만으로 add/delete 할 수 있다.
_CORE_ARTIFACT_KINDS = frozenset({
    "manifest", "config", "forecast", "forecast_metadata", "trades",
    "equity", "benchmark", "account_value", "trade_report", "metrics", "report",
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

    선언(``dashboard.panels``)에 ``label``/``type``(table|scatter|bar|stacked_area)/축
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
        # 선언 spec을 베이스로 보존(렌더 옵션 등 임의 키 통과) 후 핵심 필드만 정규화
        panel = dict(spec)
        panel.update({
            "kind": kind,
            "type": str(spec.get("type", "table")),
            "label": str(spec.get("label", kind)),
            "x": spec.get("x"),
            "y": spec.get("y"),
            "default": bool(spec.get("default", True)),
        })
        panels.append(panel)
    return panels


def build_heatmap_trace(
    frame: pd.DataFrame,
    *,
    x: str = "time",
    colorscale: str = "Viridis",
    percentile_clip: float = 99.0,
    gamma: float = 0.5,
    column_normalize: bool = False,
    opacity: float = 1.0,
    showscale: bool = True,
    overlay: bool = False,
) -> go.Heatmap | None:
    """가격×시간 격자(wide 프레임)를 ``go.Heatmap`` trace 하나로 정규화한다.

    ``x``(기본 'time') 컬럼이 시간축, 나머지 숫자 컬럼명은 가격(y)으로 본다.
    값은 분위수 클립 + gamma 보정 후 [0,1]로 정규화해 밴드 대비를 살린다.
    ``column_normalize=True``면 시간 열마다 그 시점의 분위수로 독립 정규화해,
    누적(prefix-sum) 히트맵에서 우측 열이 값을 독식해 좌측이 까맣게 죽는 현상을
    막고 가격대 밴드를 전구간 일정한 대비로 드러낸다(코인글래스 청산 히트맵 느낌).

    ``overlay=True``면 가격 그래프 뒤에 깔 용도로 hover를 끄고 0값을 투명 처리한다.
    프레임이 비었거나 가격 컬럼이 없으면 ``None``을 반환한다(겹쳐 그릴 게 없음).
    """
    time_col = x if x in frame else None
    price_cols = [c for c in frame.columns if c != time_col]
    if frame.empty or not price_cols:
        return None
    y = [float(str(c)) for c in price_cols]
    xs = frame[time_col].tolist() if time_col else list(range(len(frame)))
    z = frame[price_cols].apply(pd.to_numeric, errors="coerce").to_numpy().T
    masked = np.where(np.isfinite(z), z, np.nan)
    if column_normalize:
        # 시간 열별(축 0=가격) 독립 클립 — 누적 히트맵에서 전구간 밴드 대비 유지
        with np.errstate(invalid="ignore"):
            clip_axis = np.nanpercentile(masked, percentile_clip, axis=0, keepdims=True)
        clip_axis = np.where(np.isfinite(clip_axis) & (clip_axis > 0), clip_axis, 1.0)
        norm = np.clip(np.nan_to_num(z) / clip_axis, 0.0, 1.0) ** gamma
    else:
        finite = z[np.isfinite(z)]
        clip = float(np.percentile(finite, percentile_clip)) if finite.size and finite.max() > 0 else 1.0
        norm = np.clip(z / clip, 0.0, 1.0) ** gamma
    if overlay:
        # 0(거래 없는 가격대)은 가격선이 비치도록 투명 처리
        norm = np.where(norm > 0.0, norm, np.nan)
    return go.Heatmap(
        z=norm, x=xs, y=y, colorscale=colorscale,
        opacity=opacity, showscale=showscale,
        colorbar=(dict(title="volume") if showscale else None),
        hoverinfo=("skip" if overlay else None),
        name="청산 히트맵",
    )


def build_heatmap_figure(
    frame: pd.DataFrame,
    *,
    x: str = "time",
    label: str,
    colorscale: str = "Viridis",
    percentile_clip: float = 99.0,
    gamma: float = 0.5,
    column_normalize: bool = False,
    yaxis_title: str = "가격",
) -> go.Figure:
    """가격×시간 격자(wide 프레임)를 단독 히트맵 패널로 렌더.

    볼륨 프로파일 등 2D 격자형 extras를 코드 수정 없이 노출하는 범용 패널.
    ``yaxis_title``로 세로축 의미(절대가격/상대위치 등)를 패널별로 바꿀 수 있다.
    """
    trace = build_heatmap_trace(
        frame, x=x, colorscale=colorscale, percentile_clip=percentile_clip,
        gamma=gamma, column_normalize=column_normalize,
    )
    figure = go.Figure() if trace is None else go.Figure(trace)
    figure.update_layout(
        title=label, height=480, template="plotly_dark",
        xaxis_title="시간", yaxis_title=yaxis_title,
    )
    return _apply_theme(figure)


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
    return _apply_theme(figure)


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
    return _apply_theme(figure)


def build_stacked_area_figure(
    frame: pd.DataFrame,
    *,
    x: str = "time",
    label: str,
) -> go.Figure:
    """현금/종목 비중처럼 합이 100%인 시계열 구성을 누적 면적으로 그립니다."""
    if x not in frame:
        plot_x = frame.index
        value_columns = list(frame.columns)
    else:
        try:
            plot_x = pd.to_datetime(frame[x])
        except (TypeError, ValueError):
            plot_x = frame[x]
        value_columns = [column for column in frame.columns if column != x]
    numeric = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any(axis=0)].fillna(0.0)

    ordered = (
        ["cash"] + [column for column in numeric.columns if column != "cash"]
        if "cash" in numeric else list(numeric.columns)
    )
    figure = go.Figure()
    for i, column in enumerate(ordered):
        values = numeric[column].clip(lower=0.0)
        is_cash = column == "cash"
        figure.add_trace(go.Scatter(
            x=plot_x,
            y=values,
            mode="lines",
            name="현금" if is_cash else str(column),
            stackgroup="portfolio",
            line={
                "width": 0.8,
                "color": (
                    "#90a4ae" if is_cash
                    else WAVEFORM_PALETTE[(i - 1) % len(WAVEFORM_PALETTE)]
                ),
            },
            hovertemplate="%{x}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
        ))
    figure.update_layout(
        title=label,
        height=460,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 45, "r": 25, "t": 85, "b": 40},
        template="plotly_dark",
        uirevision=f"stacked-area-{label}",
    )
    figure.update_yaxes(title_text="비중", tickformat=".0%", range=[0, 1])
    return _apply_theme(figure)


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
