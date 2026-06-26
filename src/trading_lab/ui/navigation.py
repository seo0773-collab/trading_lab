from __future__ import annotations

import streamlit as st

from trading_lab.service import BacktestService
from trading_lab.storage import RunStore
from trading_lab.ui.artifact_io import run_label
from trading_lab.ui.backtest_page import render_new_backtest_page
from trading_lab.ui.research import render_research_page, research_available
from trading_lab.ui.results_page import render_run_result
from trading_lab.ui.system_page import render_system_page


PAGES = ["새 백테스트", "연구", "결과", "시스템"]


def render_dashboard(store: RunStore, service: BacktestService) -> None:
    page = render_sidebar()
    runs = store.list_runs(200)

    if page == "새 백테스트":
        render_new_backtest_page(store, service)
    elif page == "연구":
        render_research_page()
    elif page == "결과":
        render_results_selector(store, runs)
    else:
        render_system_page(runs)


def render_sidebar() -> str:
    st.sidebar.title("Trading Lab")
    pages = [p for p in PAGES if p != "연구" or research_available()]
    return st.sidebar.radio("메뉴", pages)


def render_results_selector(store: RunStore, runs: list[dict]) -> None:
    if not runs:
        st.title("백테스트 결과")
        st.info("저장된 실행이 없습니다. 새 백테스트를 먼저 실행하세요.")
        return

    labels = {run_label(run): run["run_id"] for run in runs}
    selected = st.selectbox("실행 선택", list(labels))
    run = store.get_run(labels[selected])
    if run is None:
        st.error("선택한 실행을 찾을 수 없습니다.")
    else:
        render_run_result(run)
