from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_lab.execution import DisabledBrokerAdapter
from trading_lab.market_catalog import (
    filter_market_options,
    load_market_options,
    option_label,
)
from trading_lab.paths import database_path, var_dir
from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.storage import RunStore
from trading_lab.strategies import get_strategy, list_strategies
from trading_lab.ui.presentation import (
    DERIVED_LABELS,
    available_extra_kinds,
    build_account_figure,
    build_bar_figure,
    build_price_indicator_figure,
    build_scatter_figure,
    build_trade_overview,
    build_trade_report,
    indicator_series,
    resolve_extra_panels,
)
from trading_lab.ui.research import render_research_page


st.set_page_config(page_title="Trading Lab", layout="wide")
store = RunStore()
service = BacktestService(store)

# yfinance가 받는 타임프레임/기간 후보. 전략 config의 기본값이 목록에 없으면
# render 시점에 맨 앞에 끼워 넣어 선택 상태를 유지한다.
TF_OPTIONS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
PERIOD_OPTIONS = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


def metric_text(value: Any, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%" if percent else f"{float(value):.2f}"


def money_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.2f}"


def load_frame(path: str) -> pd.DataFrame:
    artifact = Path(path)
    if artifact.suffix == ".parquet":
        return pd.read_parquet(artifact)
    return pd.read_csv(artifact, index_col=0, parse_dates=True)


def load_json(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_path(run: dict[str, Any], kind: str) -> str | None:
    return next(
        (item["path"] for item in run.get("artifacts", []) if item["kind"] == kind),
        None,
    )


def run_label(run: dict[str, Any]) -> str:
    if run.get("run_name"):
        return f"{run['run_name']} · {run['status']}"
    return (
        f"{run['created_at']} | {run['symbol']} | {run['status']} | "
        f"{run['run_id'][:8]}"
    )


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_market_options(chart_type: str):
    return load_market_options(chart_type)


def strategy_config_dict(strategy_id: str) -> dict[str, Any]:
    path = get_strategy(strategy_id).config_path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_strategy_tunables(strategy_id: str) -> dict[str, Any]:
    """config의 tunables 스키마로 파라미터 위젯을 자동 렌더하고 override를 모읍니다."""
    tunables = strategy_config_dict(strategy_id).get("tunables") or []
    overrides: dict[str, Any] = {}
    if not tunables:
        return overrides
    with st.expander("전략 파라미터", expanded=False):
        for spec in tunables:
            name = spec["name"]
            label = spec.get("label", name)
            kind = spec.get("type", "number")
            key = f"tunable-{strategy_id}-{name}"
            if kind == "select":
                options = spec["options"]
                default = spec.get("default", options[0])
                index = options.index(default) if default in options else 0
                overrides[name] = st.selectbox(label, options, index=index, key=key)
            elif kind == "int":
                overrides[name] = int(st.number_input(
                    label,
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    value=int(spec["default"]),
                    step=int(spec.get("step", 1)),
                    key=key,
                ))
            else:
                overrides[name] = float(st.number_input(
                    label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=float(spec.get("step", 0.01)),
                    format="%.4f",
                    key=key,
                ))
    return overrides


def run_inputs(run: dict[str, Any]) -> tuple[dict[str, Any], float]:
    config = load_json(artifact_path(run, "config"))
    manifest = load_json(artifact_path(run, "manifest"))
    metrics = run.get("metrics") or {}
    initial_capital = float(
        metrics.get("initial_capital", manifest.get("initial_capital", 10_000.0))
    )
    return config, initial_capital


def render_extra_panels(run: dict[str, Any], dashboard_config: dict[str, Any]) -> None:
    """전략별 보조 패널을 config 선언으로 렌더하고, multiselect로 add/delete.

    전략은 ``dashboard.panels``(kind/type/label/x/y/default)만 선언하면 되고
    app.py 는 더 이상 전략별로 수정하지 않는다(공통 인프라). 선언 없는 extras도
    표로 자동 노출된다.
    """
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
        "표시할 보조 패널 (전략별 추가 자료 — 선택해서 추가/삭제)",
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


def load_json_frame(run: dict[str, Any], kind: str | None) -> pd.DataFrame:
    path = artifact_path(run, kind) if kind else None
    records = load_json(path) if path and Path(path).exists() else []
    return pd.DataFrame(records or [])


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


def render_backtest_detail(
    run: dict[str, Any],
    config: dict[str, Any],
    initial_capital: float,
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.Series,
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
            series_map={
                name: indicators[name] for name in selected_indicators
            },
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
            benchmark_price=forecast["close"],
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


def render_run_result(run: dict[str, Any]) -> None:
    st.title("백테스트 결과")
    st.caption(
        f"{run.get('run_name') or run['run_id']} · {run['symbol']} · "
        f"{run['strategy_id']} · {run['phase']}"
    )
    if run["status"] != "succeeded":
        message = run.get("error") or "완료된 결과가 없습니다."
        st.error(message)
        return

    metrics = run.get("metrics") or {}
    config, initial_capital = run_inputs(run)
    final_account = metrics.get(
        "final_account_value",
        initial_capital * (1.0 + float(metrics.get("total_return", 0.0))),
    )
    values = [
        ("거래 수", str(metrics.get("trades", 0))),
        ("승률", metric_text(metrics.get("hit_rate"), percent=True)),
        ("누적 수익률", metric_text(metrics.get("total_return"), percent=True)),
        ("최종 계좌", money_text(final_account)),
        ("순손익", money_text(float(final_account) - initial_capital)),
        ("Sharpe", metric_text(metrics.get("sharpe"))),
        ("최대 낙폭", metric_text(metrics.get("max_drawdown"), percent=True)),
    ]
    for column, (label, value) in zip(st.columns(len(values)), values):
        column.metric(label, value)

    required = {
        kind: artifact_path(run, kind) for kind in ("forecast", "trades", "equity")
    }
    missing = [kind for kind, path in required.items() if not path or not Path(path).exists()]
    if missing:
        st.error(f"결과 아티팩트 누락: {', '.join(missing)}")
        return

    forecast = load_frame(required["forecast"])
    trades = load_frame(required["trades"])
    equity_frame = load_frame(required["equity"])
    equity = equity_frame["equity"].astype(float)
    dashboard_config = config.get("dashboard") or {}
    learning_config = dashboard_config.get("learning_tab") or {}
    if learning_config:
        performance_tab, learning_tab = st.tabs([
            "성과 및 거래",
            str(learning_config.get("label", "학습 결과")),
        ])
        with performance_tab:
            render_backtest_detail(
                run, config, initial_capital, forecast, trades, equity
            )
        with learning_tab:
            render_learning_result(run, learning_config)
    else:
        render_backtest_detail(
            run, config, initial_capital, forecast, trades, equity
        )

    with st.expander("실행 정보 및 원본 아티팩트"):
        st.json({
            "상태": run["status"],
            "전략": run["strategy_id"],
            "심볼": run["symbol"],
            "평가 구간": run["phase"],
            "초기 자본": initial_capital,
            "설정": config,
        })
        st.dataframe(pd.DataFrame(run["artifacts"]), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(run["events"]), width="stretch", hide_index=True)


st.sidebar.title("Trading Lab")
page = st.sidebar.radio("메뉴", ["새 백테스트", "연구", "결과", "시스템"])
runs = store.list_runs(200)

if page == "새 백테스트":
    st.title("새 백테스트")
    st.caption("전략 실행 후 같은 화면 아래에 전체 결과를 표시합니다.")
    strategies = [item for item in list_strategies() if item.enabled]
    strategy_id = st.selectbox("전략", [item.strategy_id for item in strategies])
    config_overrides = render_strategy_tunables(strategy_id)
    chart_label = st.selectbox("차트 타입", ["크립토", "주식", "랜덤"])
    chart_type = {"크립토": "crypto", "주식": "stock", "랜덤": "random"}[
        chart_label
    ]

    selected_option = None
    if chart_type == "random":
        symbol = "RANDOM"
        chart_detail = "랜덤"
        bars_per_year = 8760
        synthetic = True
        st.info("랜덤 차트는 재현 가능한 합성 데이터를 사용합니다.")
    else:
        search = st.text_input(
            "종목 검색", placeholder="심볼 또는 종목명 검색 (예: BTC, Bitcoin, SPY)"
        )
        market_options = cached_market_options(chart_type)
        filtered_options = filter_market_options(market_options, search)
        if filtered_options:
            selected_option = st.selectbox(
                "종목 (시가총액 높은 순)",
                filtered_options,
                format_func=option_label,
            )
            symbol = selected_option.symbol
            chart_detail = selected_option.detail
            bars_per_year = selected_option.bars_per_year
        else:
            st.warning("검색 조건에 맞는 종목이 없습니다.")
            symbol = ""
            chart_detail = ""
            bars_per_year = 8760 if chart_type == "crypto" else 1638
        synthetic = False
        st.caption(
            "시가총액은 Yahoo Finance 조회값으로 정렬하며, 조회 실패 시 내장 순서를 사용합니다."
        )

    base_config = strategy_config_dict(strategy_id)
    default_tf = str(base_config.get("interval", "1d"))
    default_period = str(base_config.get("period", "max"))
    tf_choices = list(dict.fromkeys([default_tf, *TF_OPTIONS]))
    period_choices = list(dict.fromkeys([default_period, *PERIOD_OPTIONS]))

    st.subheader("데이터 설정")
    tf_col, period_col = st.columns(2)
    interval = tf_col.selectbox(
        "타임프레임 (TF)", tf_choices, index=tf_choices.index(default_tf),
        key=f"tf-{strategy_id}",
    )
    if chart_type == "random":
        period = default_period
        period_col.caption("합성 차트는 기간 대신 합성 봉 수를 사용합니다.")
    else:
        period = period_col.selectbox(
            "데이터 기간", period_choices,
            index=period_choices.index(default_period),
            key=f"period-{strategy_id}",
        )
    st.caption(
        "yfinance는 짧은 TF에서 받을 수 있는 기간이 제한됩니다 "
        "(예: 1h는 약 730일, 1m은 약 7일)."
    )
    config_overrides["interval"] = interval
    if chart_type != "random":
        config_overrides["period"] = period

    phase = st.selectbox("평가 구간", ["validation", "all"])
    initial_capital = st.number_input(
        "초기 계좌 금액", min_value=100.0, value=10_000.0, step=1_000.0,
    )
    st.info(
        "현재 전략은 연구 전용이며 live 주문과 holdout test 실행은 잠겨 있습니다."
    )
    if st.button("백테스트 실행", type="primary", disabled=not symbol):
        with st.spinner("데이터, 예측, 체결, 리포트 파이프라인 실행 중..."):
            run_id = service.run(BacktestRequest(
                strategy_id=strategy_id,
                symbol=symbol,
                phase=phase,
                chart_type=chart_type,
                chart_detail=chart_detail,
                bars_per_year=bars_per_year,
                initial_capital=float(initial_capital),
                synthetic=synthetic,
                config_overrides=config_overrides or None,
            ))
        st.session_state["last_backtest_run_id"] = run_id

    # 결과는 버튼 블록 밖에서 렌더링해야 결과 화면의 위젯(보조지표 선택 등)을
    # 조작해도 Streamlit 재실행 후 결과가 유지됩니다.
    last_run_id = st.session_state.get("last_backtest_run_id")
    if last_run_id:
        result = store.get_run(last_run_id)
        if result and result["status"] == "succeeded":
            st.success(f"실행 완료: {result.get('run_name') or last_run_id}")
            render_run_result(result)
        elif result:
            st.error(result.get("error") or "실행이 실패했습니다.")
        else:
            st.error("실행 기록을 찾을 수 없습니다.")

elif page == "연구":
    render_research_page()

elif page == "결과":
    if not runs:
        st.title("백테스트 결과")
        st.info("저장된 실행이 없습니다. 새 백테스트를 먼저 실행하세요.")
    else:
        labels = {run_label(run): run["run_id"] for run in runs}
        selected = st.selectbox("실행 선택", list(labels))
        run = store.get_run(labels[selected])
        if run is None:
            st.error("선택한 실행을 찾을 수 없습니다.")
        else:
            render_run_result(run)

else:
    st.title("시스템")
    succeeded = sum(run["status"] == "succeeded" for run in runs)
    failed = sum(run["status"] == "failed" for run in runs)
    columns = st.columns(4)
    columns[0].metric("전체 실행", len(runs))
    columns[1].metric("성공", succeeded)
    columns[2].metric("실패", failed)
    columns[3].metric("Live 거래", "비활성")
    st.write({
        "database": str(database_path()),
        "runtime_directory": str(var_dir()),
        "broker_adapter": DisabledBrokerAdapter.__name__,
        "live_order_submission": "disabled",
    })
    st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
