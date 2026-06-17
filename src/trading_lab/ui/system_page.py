from __future__ import annotations

import pandas as pd
import streamlit as st

from trading_lab.execution import DisabledBrokerAdapter
from trading_lab.paths import database_path, var_dir


def render_system_page(runs: list[dict]) -> None:
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
