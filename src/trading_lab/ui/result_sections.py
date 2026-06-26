from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_lab.ui import portfolio_report
from trading_lab.ui.artifact_io import artifact_path, load_json, load_json_frame
from trading_lab.ui.formatting import currency_spec, metric_text, money_text
from trading_lab.ui.presentation import (
    DERIVED_LABELS,
    available_extra_kinds,
    build_account_figure,
    build_bar_figure,
    build_heatmap_figure,
    build_heatmap_trace,
    build_price_indicator_figure,
    build_scatter_figure,
    build_stacked_area_figure,
    build_trade_overview,
    build_trade_report,
    forecast_is_portfolio,
    indicator_series,
    resolve_extra_panels,
)


def render_backtest_detail(
    run: dict[str, Any],
    config: dict[str, Any],
    initial_capital: float,
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
) -> None:
    horizon = int(config.get("horizon", 72))
    confidence_quantile = float(config.get("confidence_quantile", 0.85))
    quantile_window = int(config.get("quantile_window", 2000))

    indicators = indicator_series(
        forecast,
        horizon=horizon,
        confidence_quantile=confidence_quantile,
        quantile_window=quantile_window,
    )
    dashboard_config = config.get("dashboard") or {}
    indicator_labels = {
        **DERIVED_LABELS,
        **{
            str(key): str(value)
            for key, value in (dashboard_config.get("indicator_labels") or {}).items()
        },
    }
    default_indicators = [
        name for name in dashboard_config.get("default_indicators", [])
        if name in indicators
    ] or list(indicators)[:4]
    portfolio_view = forecast_is_portfolio(forecast)
    # 포트폴리오형(NAV+구성)은 가격·웨이브폼보다 표준 리포트가 메인이므로
    # 가격/보조지표 차트는 접이식(기본 접힘)으로 둔다. 단일자산 전략은 그대로 노출.
    price_section = (
        st.expander("구성 자산 NAV · 보조지표 차트", expanded=False)
        if portfolio_view else contextlib.nullcontext()
    )
    with price_section:
        selected_indicators = st.multiselect(
            "표시할 보조지표 (스케일이 비슷한 지표는 같은 패널에 겹쳐 표시)",
            list(indicators),
            default=default_indicators,
            format_func=lambda name: indicator_labels.get(name, name),
            key=f"indicators-{run['run_id']}",
        )
        heatmap_spec, heatmap_frame = _resolve_heatmap_panel(run, dashboard_config)
        heatmap_overlay = None
        # overlay=false 패널(예: 세로축이 절대가격이 아닌 상대위치)은 가격 y축과
        # 정렬되지 않으므로 가격 차트에 겹치지 않고 독립 패널로만 노출한다.
        if heatmap_frame is not None and bool(heatmap_spec.get("overlay", True)):
            show_heatmap = st.checkbox(
                "가격 그래프에 청산 히트맵 겹쳐 표시",
                value=True,
                key=f"heatmap-overlay-{run['run_id']}",
            )
            if show_heatmap:
                heatmap_overlay = build_heatmap_trace(
                    heatmap_frame,
                    x=str(heatmap_spec.get("x") or "time"),
                    column_normalize=bool(heatmap_spec.get("column_normalize", False)),
                    opacity=0.55,
                    showscale=False,
                    overlay=True,
                )
        st.plotly_chart(
            build_price_indicator_figure(
                forecast,
                trades,
                symbol=run["symbol"],
                horizon=horizon,
                series_map={name: indicators[name] for name in selected_indicators},
                labels=indicator_labels,
                heatmap_overlay=heatmap_overlay,
            ),
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
        )
    currency_sym, currency_dec, currency_step, _ = currency_spec(
        config.get("base_currency"))
    invest_mode = st.radio(
        "투자 방식", ["거치식", "적립식"], horizontal=True,
        key=f"invest-mode-{run['run_id']}",
        help="거치식=초기 일시금 전액 투입. 적립식=초기 일시금 + 매월 정액 추가 투입.",
    )
    monthly_contribution = 0.0
    if invest_mode == "적립식":
        monthly_contribution = float(st.number_input(
            f"월 적립액 ({currency_sym})", min_value=0.0,
            value=float(currency_step), step=float(currency_step),
            key=f"monthly-contrib-{run['run_id']}",
        ))
    st.plotly_chart(
        build_account_figure(
            equity,
            initial_capital=initial_capital,
            symbol=run["symbol"],
            benchmark_price=(None if portfolio_view else forecast["close"]),
            benchmark_equity=benchmark_equity,
            monthly_contribution=monthly_contribution,
            currency_symbol=currency_sym,
        ),
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )

    # 포트폴리오형(NAV 기반) 전략은 가격·웨이브폼이 의미없으므로 표준 포트폴리오
    # 리포트로 대체 보강한다. config dashboard.portfolio_report 로 강제할 수도 있다.
    if portfolio_view or dashboard_config.get("portfolio_report"):
        render_portfolio_report(run, equity, benchmark_equity)

    st.subheader("트레이드 리포트")
    overview = build_trade_overview(trades)
    overview_values = [
        ("롱 횟수", str(overview["long_trades"])),
        ("숏 횟수", str(overview["short_trades"])),
        ("롱 평균 수익률", metric_text(overview["long_avg_return"], percent=True)),
        ("숏 평균 수익률", metric_text(overview["short_avg_return"], percent=True)),
        ("롱 청산율", metric_text(overview["long_close_rate"], percent=True)),
        ("숏 청산율", metric_text(overview["short_close_rate"], percent=True)),
    ]
    for column, (label, value) in zip(st.columns(6), overview_values):
        column.metric(label, value)
    st.caption("청산율은 해당 방향의 진입 거래 중 청산 시각이 기록된 거래 비율입니다.")

    report = build_trade_report(
        trades,
        equity,
        initial_capital=initial_capital,
        horizon=horizon,
        execution=str(config.get("execution", "next_open")),
    )
    if report.empty:
        st.info("선택한 구간에서 체결된 거래가 없습니다.")
    else:
        display = report.rename(columns={
            "trade_number": "번호",
            "symbol": "종목",
            "side": "방향",
            "entry_time": "진입 시각",
            "entry_price": "진입가",
            "exit_time": "청산 시각",
            "exit_price": "청산가",
            "stop_loss_price": "손절가",
            "take_profit_price": "익절가",
            "exit_reason": "청산 사유",
            "net_return_pct": "결과 손익 %",
            "account_value_after": "거래 후 계좌 금액",
            "entry_reason": "진입 근거",
        })
        for column in ("손절가", "익절가"):
            display[column] = display[column].map(
                lambda value: "미사용" if pd.isna(value) else f"{value:,.4f}"
            )
        display["결과 손익 %"] = display["결과 손익 %"].map(
            lambda value: f"{value:+.2f}%"
        )
        display["거래 후 계좌 금액"] = display["거래 후 계좌 금액"].map(
            lambda value: money_text(value, currency_sym, currency_dec)
        )
        st.dataframe(display, width="stretch", hide_index=True, height=520)
        if report[["stop_loss_price", "take_profit_price"]].isna().all().all():
            st.caption(
                "이 전략은 고정 손절/익절 주문을 사용하지 않습니다. "
                "해당 가격은 임의 추정하지 않고 '미사용'으로 표시합니다."
            )

    render_extra_panels(run, dashboard_config)


def _resolve_heatmap_panel(
    run: dict[str, Any], dashboard_config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """heatmap 타입 보조 패널이 있으면 그 스펙과 프레임을 돌려준다(없으면 None).

    전략명을 가리지 않는 범용 탐지: ``dashboard.panels`` 중 ``type == "heatmap"``
    선언이 있고 실제 아티팩트가 존재하면 가격 그래프 오버레이용으로 로드한다.
    """
    available = available_extra_kinds(run)
    for panel in resolve_extra_panels(dashboard_config, available):
        if panel.get("type") != "heatmap":
            continue
        path = artifact_path(run, panel["kind"])
        records = load_json(path) if path and Path(path).exists() else None
        if not records:
            continue
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        return panel, frame
    return {}, None


def render_portfolio_report(run: dict[str, Any], equity, benchmark_equity) -> None:
    """포트폴리오형 전략(NAV 기반)에 QuantStats 표준 리포트를 노출한다 —
    멀티에셋/계층 전략용 '플랫폼 표준 형식'(지표표 + 누적수익·낙폭·월별
    히트맵·롤링 샤프). equity/benchmark는 StrategyArtifacts가 이미 만든다."""
    st.subheader("포트폴리오 표준 리포트 (QuantStats)")
    if not portfolio_report.is_available():
        st.info(
            "QuantStats 미설치 — `.venv/bin/pip install quantstats` 후 표준 "
            "리포트가 표시됩니다."
        )
        return
    try:
        table = portfolio_report.metrics_table(equity, benchmark_equity)
        st.dataframe(table, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"지표표 생성 실패: {exc}")

    figures = portfolio_report.report_figures(equity, benchmark_equity)
    columns = st.columns(2)
    for i, name in enumerate(portfolio_report.REPORT_FIGURES):
        figure = figures.get(name)
        if figure is None:
            continue
        with columns[i % 2]:
            st.caption(portfolio_report.FIGURE_LABELS.get(name, name))
            st.pyplot(figure, clear_figure=True)


def _panel_scatter(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    x, y = spec.get("x"), spec.get("y")
    if x in frame and y in frame:
        return build_scatter_figure(frame, x, y, label=spec["label"])
    return None


def _panel_bar(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    x, y = spec.get("x"), spec.get("y")
    if x in frame and y in frame:
        return build_bar_figure(frame, x, y, label=spec["label"])
    return None


def _panel_stacked_area(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    return build_stacked_area_figure(
        frame, x=str(spec.get("x") or "time"), label=spec["label"])


def _panel_heatmap(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    return build_heatmap_figure(
        frame, x=str(spec.get("x") or "time"), label=spec["label"],
        column_normalize=bool(spec.get("column_normalize", False)),
        yaxis_title=str(spec.get("yaxis_title", "가격")))


# 패널 type → figure 빌더 레지스트리. 새 시각화를 추가할 때는 presentation.py에
# 빌더를 만들고 여기 한 줄만 등록하면 된다 — render_extra_panels 디스패치 본체는
# 더 이상 수정하지 않는다(Open-Closed). 미등록 타입은 원본 테이블로 폴백한다.
PANEL_RENDERERS: dict[str, Any] = {
    "scatter": _panel_scatter,
    "bar": _panel_bar,
    "stacked_area": _panel_stacked_area,
    "heatmap": _panel_heatmap,
}


def _render_panel_figure(spec: dict[str, Any], frame: pd.DataFrame) -> None:
    """spec["type"]을 PANEL_RENDERERS로 디스패치. 미등록 타입이거나 필요한
    컬럼이 없어 빌더가 None을 주면 원본 테이블로 폴백한다(전략명·타입 하드코딩 없음)."""
    renderer = PANEL_RENDERERS.get(str(spec.get("type")))
    figure = renderer(frame, spec) if renderer is not None else None
    if figure is None:
        st.dataframe(frame, width="stretch", hide_index=True)
        return
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_extra_panels(run: dict[str, Any], dashboard_config: dict[str, Any]) -> None:
    learning = dashboard_config.get("learning_tab") or {}
    learning_kinds = {
        str(learning[key])
        for key in (
            "summary_kind", "predictions_kind",
            "sensitivity_kind", "events_kind",
        )
        if learning.get(key)
    }
    available = [
        kind for kind in available_extra_kinds(run)
        if kind not in learning_kinds
    ]
    if not available:
        return
    panels = resolve_extra_panels(dashboard_config, available)
    spec_by_kind = {panel["kind"]: panel for panel in panels}
    labels = {panel["kind"]: panel["label"] for panel in panels}

    st.subheader("보조 패널")
    chosen = st.multiselect(
        "표시할 보조 패널 (전략별 추가 자료 - 선택해서 추가/삭제)",
        [panel["kind"] for panel in panels],
        default=[panel["kind"] for panel in panels if panel["default"]],
        format_func=lambda kind: labels.get(kind, kind),
        key=f"panels-{run['run_id']}",
    )
    for kind in chosen:
        spec = spec_by_kind[kind]
        path = artifact_path(run, kind)
        records = load_json(path) if path and Path(path).exists() else None
        if not records:
            continue
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        st.markdown(f"**{spec['label']}**")
        _render_panel_figure(spec, frame)


def render_learning_result(
    run: dict[str, Any], learning_config: dict[str, Any]
) -> None:
    st.subheader("재무 변화로 학습한 주가 반응")
    st.caption(
        "각 재무 발표 시점에 과거에 이미 실현된 사례만 사용해 모델을 다시 "
        "학습합니다. 실제 수익률은 진입 판단이 아니라 사후 예측력 평가에 사용됩니다."
    )

    summary = load_json_frame(run, learning_config.get("summary_kind"))
    if summary.empty:
        st.info("표본이 부족해 학습 진단값이 생성되지 않았습니다.")
    else:
        for _, row in summary.sort_values("horizon_days").iterrows():
            horizon = int(row["horizon_days"])
            st.markdown(f"**{horizon}일 예측 성능**")
            values = [
                ("평가 표본", f"{int(row['samples'])}건"),
                ("Spearman IC", metric_text(row.get("spearman_ic"))),
                ("평균 절대오차", metric_text(row.get("mae"), percent=True)),
                (
                    "방향 적중률",
                    metric_text(row.get("direction_accuracy"), percent=True),
                ),
                (
                    "평균 예상수익률",
                    metric_text(row.get("mean_predicted_return"), percent=True),
                ),
                (
                    "평균 실제수익률",
                    metric_text(row.get("mean_actual_return"), percent=True),
                ),
            ]
            for column, (label, value) in zip(st.columns(6), values):
                column.metric(label, value)

    predictions = load_json_frame(
        run, learning_config.get("predictions_kind")
    )
    if not predictions.empty:
        st.markdown("**예상수익률과 실제수익률**")
        columns = st.columns(2)
        for column, horizon in zip(columns, (20, 60)):
            x, y = f"pred_ret_{horizon}d", f"ret_{horizon}d"
            if x in predictions and y in predictions:
                column.plotly_chart(
                    build_scatter_figure(
                        predictions, x, y,
                        label=f"{horizon}일 예상 vs 실제",
                    ),
                    width="stretch",
                    config={"displaylogo": False},
                )

    sensitivity = load_json_frame(
        run, learning_config.get("sensitivity_kind")
    )
    if (
        not sensitivity.empty
        and "factor" in sensitivity
        and "sensitivity_mean" in sensitivity
    ):
        st.markdown("**재무 팩터별 평균 민감도**")
        st.plotly_chart(
            build_bar_figure(
                sensitivity,
                "factor",
                "sensitivity_mean",
                label="20일 수익률에 대한 평균 민감도",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "양수는 해당 재무 변화가 클수록 예상수익률이 높아지는 관계, "
            "음수는 반대 관계를 뜻합니다. 인과관계를 증명하는 값은 아닙니다."
        )

    events = load_json_frame(run, learning_config.get("events_kind"))
    if not events.empty:
        st.markdown("**발표 이벤트별 학습 데이터**")
        st.dataframe(events, width="stretch", hide_index=True, height=520)
