from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_lab.ui.artifact_io import (
    artifact_path,
    load_frame,
    run_inputs,
)
from trading_lab.ui.formatting import metric_text, money_text
from trading_lab.ui.result_sections import (
    render_backtest_detail,
    render_learning_result,
)


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
    missing = [
        kind for kind, path in required.items()
        if not path or not Path(path).exists()
    ]
    if missing:
        st.error(f"결과 아티팩트 누락: {', '.join(missing)}")
        return

    forecast = load_frame(required["forecast"])
    trades = load_frame(required["trades"])
    equity_frame = load_frame(required["equity"])
    equity = equity_frame["equity"].astype(float)
    benchmark_equity = load_benchmark_equity(run)
    dashboard_config = config.get("dashboard") or {}
    learning_config = dashboard_config.get("learning_tab") or {}
    if learning_config:
        performance_tab, learning_tab = st.tabs([
            "성과 및 거래",
            str(learning_config.get("label", "학습 결과")),
        ])
        with performance_tab:
            render_backtest_detail(
                run, config, initial_capital, forecast, trades, equity,
                benchmark_equity,
            )
        with learning_tab:
            render_learning_result(run, learning_config)
    else:
        render_backtest_detail(
            run, config, initial_capital, forecast, trades, equity,
            benchmark_equity,
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


def load_benchmark_equity(run: dict[str, Any]) -> pd.Series | None:
    path = artifact_path(run, "benchmark")
    if not path or not Path(path).exists():
        return None
    frame = load_frame(path)
    if "benchmark" not in frame:
        return None
    return frame["benchmark"].astype(float)
