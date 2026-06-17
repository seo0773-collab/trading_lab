from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_lab.ui.artifact_io import artifact_path, load_json, load_json_frame
from trading_lab.ui.formatting import metric_text, money_text
from trading_lab.ui.presentation import (
    DERIVED_LABELS,
    available_extra_kinds,
    build_account_figure,
    build_bar_figure,
    build_price_indicator_figure,
    build_scatter_figure,
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
    selected_indicators = st.multiselect(
        "표시할 보조지표 (스케일이 비슷한 지표는 같은 패널에 겹쳐 표시)",
        list(indicators),
        default=default_indicators,
        format_func=lambda name: indicator_labels.get(name, name),
        key=f"indicators-{run['run_id']}",
    )
    st.plotly_chart(
        build_price_indicator_figure(
            forecast,
            trades,
            symbol=run["symbol"],
            horizon=horizon,
            series_map={name: indicators[name] for name in selected_indicators},
            labels=indicator_labels,
        ),
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.plotly_chart(
        build_account_figure(
            equity,
            initial_capital=initial_capital,
            symbol=run["symbol"],
            benchmark_price=(
                None if forecast_is_portfolio(forecast) else forecast["close"]
            ),
            benchmark_equity=benchmark_equity,
        ),
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )

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
            money_text
        )
        st.dataframe(display, width="stretch", hide_index=True, height=520)
        if report[["stop_loss_price", "take_profit_price"]].isna().all().all():
            st.caption(
                "이 전략은 고정 손절/익절 주문을 사용하지 않습니다. "
                "해당 가격은 임의 추정하지 않고 '미사용'으로 표시합니다."
            )

    render_extra_panels(run, dashboard_config)


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
        ptype, x, y = spec["type"], spec.get("x"), spec.get("y")
        st.markdown(f"**{spec['label']}**")
        if ptype == "scatter" and x in frame and y in frame:
            st.plotly_chart(
                build_scatter_figure(frame, x, y, label=spec["label"]),
                width="stretch", config={"displaylogo": False},
            )
        elif ptype == "bar" and x in frame and y in frame:
            st.plotly_chart(
                build_bar_figure(frame, x, y, label=spec["label"]),
                width="stretch", config={"displaylogo": False},
            )
        else:
            st.dataframe(frame, width="stretch", hide_index=True)


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
