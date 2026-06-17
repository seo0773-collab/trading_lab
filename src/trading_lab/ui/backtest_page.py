from __future__ import annotations

import streamlit as st

from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.storage import RunStore
from trading_lab.ui.backtest_controls import (
    PORTFOLIO_STRATEGY_ID,
    MarketSelection,
    render_data_controls,
    render_market_selector,
    render_multi_portfolio_selector,
    render_portfolio_mode_selector,
    render_strategy_selector,
    render_strategy_tunables,
)
from trading_lab.ui.config import strategy_config_dict
from trading_lab.ui.results_page import render_run_result


def render_new_backtest_page(store: RunStore, service: BacktestService) -> None:
    st.title("새 백테스트")
    st.caption("전략 실행 후 같은 화면 아래에 전체 결과를 표시합니다.")

    portfolio_mode = render_portfolio_mode_selector()
    if portfolio_mode == "멀티 포트폴리오":
        strategy_id = PORTFOLIO_STRATEGY_ID
        selection, config_overrides = render_multi_backtest_setup(strategy_id)
    else:
        strategy_id = render_strategy_selector()
        selection, config_overrides = render_single_backtest_setup(strategy_id)

    phase = st.selectbox("평가 구간", ["validation", "all"])
    initial_capital = st.number_input(
        "초기 계좌 금액", min_value=100.0, value=10_000.0, step=1_000.0,
    )
    st.info(
        "현재 전략은 연구 전용이며 live 주문과 holdout test 실행은 잠겨 있습니다."
    )
    if st.button("백테스트 실행", type="primary", disabled=not selection.symbol):
        run_backtest(
            service,
            strategy_id=strategy_id,
            selection=selection,
            phase=phase,
            initial_capital=float(initial_capital),
            config_overrides=config_overrides,
        )

    render_last_backtest_result(store)


def render_single_backtest_setup(
    strategy_id: str,
) -> tuple[MarketSelection, dict]:
    config_overrides = render_strategy_tunables(strategy_id)
    selection = render_market_selector()
    base_config = strategy_config_dict(strategy_id)
    config_overrides.update(render_data_controls(strategy_id, base_config, selection))
    return selection, config_overrides


def render_multi_backtest_setup(
    strategy_id: str,
) -> tuple[MarketSelection, dict]:
    st.selectbox("전략", [strategy_id], format_func=lambda _: "yoon1")
    config_overrides = render_strategy_tunables(strategy_id)
    portfolio_selection = render_multi_portfolio_selector()
    selection = portfolio_selection.market
    config_overrides.update(portfolio_selection.config_overrides)
    base_config = strategy_config_dict(strategy_id)
    config_overrides.update(render_data_controls(strategy_id, base_config, selection))
    return selection, config_overrides


def run_backtest(
    service: BacktestService,
    *,
    strategy_id: str,
    selection: MarketSelection,
    phase: str,
    initial_capital: float,
    config_overrides: dict,
) -> None:
    request = BacktestRequest(
        strategy_id=strategy_id,
        symbol=selection.symbol,
        phase=phase,
        chart_type=selection.chart_type,
        chart_detail=selection.chart_detail,
        bars_per_year=selection.bars_per_year,
        initial_capital=initial_capital,
        synthetic=selection.synthetic,
        config_overrides=config_overrides or None,
    )
    with st.spinner("데이터, 예측, 체결, 리포트 파이프라인 실행 중..."):
        run_id = service.run(request)
    st.session_state["last_backtest_run_id"] = run_id


def render_last_backtest_result(store: RunStore) -> None:
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
