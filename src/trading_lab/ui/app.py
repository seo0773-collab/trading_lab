from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
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
from trading_lab.strategies import list_strategies
from trading_lab.ui.presentation import (
    build_account_figure,
    build_indicator_figure,
    build_price_figure,
    build_trade_overview,
    build_trade_report,
)


st.set_page_config(page_title="Trading Lab", layout="wide")
store = RunStore()
service = BacktestService(store)


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


def run_inputs(run: dict[str, Any]) -> tuple[dict[str, Any], float]:
    config = load_json(artifact_path(run, "config"))
    manifest = load_json(artifact_path(run, "manifest"))
    metrics = run.get("metrics") or {}
    initial_capital = float(
        metrics.get("initial_capital", manifest.get("initial_capital", 10_000.0))
    )
    return config, initial_capital


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
    horizon = int(config.get("horizon", 72))
    confidence_quantile = float(config.get("confidence_quantile", 0.85))
    quantile_window = int(config.get("quantile_window", 2000))

    st.plotly_chart(
        build_price_figure(
            forecast, trades, symbol=run["symbol"], horizon=horizon,
        ),
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.plotly_chart(
        build_indicator_figure(
            forecast,
            horizon=horizon,
            confidence_quantile=confidence_quantile,
            quantile_window=quantile_window,
        ),
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.plotly_chart(
        build_account_figure(
            equity, initial_capital=initial_capital, symbol=run["symbol"],
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
    st.caption(
        "현재 h72-price-v1은 고정 손절/익절 주문을 사용하지 않습니다. "
        "해당 가격은 임의 추정하지 않고 '미사용'으로 표시합니다."
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


def render_compare(comparable: list[dict[str, Any]]) -> None:
    st.title("실행 결과 비교")
    st.caption("성과 지표와 동일 기준 누적 수익률 파형을 함께 비교합니다.")
    if len(comparable) < 2:
        st.info("비교하려면 성공한 실행이 두 개 이상 필요합니다.")
        return

    labels = {run_label(run): run for run in comparable}
    selected = st.multiselect(
        "비교할 실행",
        list(labels),
        default=list(labels)[:2],
        max_selections=6,
    )
    rows: list[dict[str, Any]] = []
    figure = go.Figure()
    for label in selected:
        run = labels[label]
        metrics = run["metrics"]
        _, initial_capital = run_inputs(run)
        final_account = metrics.get(
            "final_account_value",
            initial_capital * (1.0 + metrics["total_return"]),
        )
        rows.append({
            "실행": run.get("run_name") or run["run_id"][:8],
            "심볼": run["symbol"],
            "구간": run["phase"],
            "거래 수": metrics["trades"],
            "승률 %": metrics["hit_rate"] * 100.0,
            "수익률 %": metrics["total_return"] * 100.0,
            "Sharpe": metrics["sharpe"],
            "MDD %": metrics["max_drawdown"] * 100.0,
            "초기 계좌": initial_capital,
            "최종 계좌": final_account,
        })
        equity_path = artifact_path(run, "equity")
        if equity_path and Path(equity_path).exists():
            equity = load_frame(equity_path)["equity"]
            figure.add_trace(go.Scatter(
                x=equity.index,
                y=(equity - 1.0) * 100.0,
                mode="lines",
                name=run.get("run_name") or f"{run['symbol']} {run['run_id'][:8]}",
            ))

    if not rows:
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    figure.update_layout(
        title="누적 수익률 비교",
        height=520,
        hovermode="x unified",
        yaxis_title="누적 수익률 %",
        template="plotly_dark",
        legend={"orientation": "h"},
    )
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True})


st.sidebar.title("Trading Lab")
page = st.sidebar.radio("메뉴", ["새 백테스트", "결과", "비교", "시스템"])
runs = store.list_runs(200)

if page == "새 백테스트":
    st.title("새 백테스트")
    st.caption("전략 실행 후 같은 화면 아래에 전체 결과를 표시합니다.")
    strategies = [item for item in list_strategies() if item.enabled]
    strategy_id = st.selectbox("전략", [item.strategy_id for item in strategies])
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
            ))
        result = store.get_run(run_id)
        if result and result["status"] == "succeeded":
            st.success(f"실행 완료: {result.get('run_name') or run_id}")
            render_run_result(result)
        else:
            st.error(result["error"] if result else "실행 기록을 찾을 수 없습니다.")

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

elif page == "비교":
    render_compare([run for run in runs if run.get("metrics")])

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
