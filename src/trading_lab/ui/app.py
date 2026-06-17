from __future__ import annotations

import streamlit as st

from trading_lab.service import BacktestService
from trading_lab.storage import RunStore
from trading_lab.ui.navigation import render_dashboard


st.set_page_config(page_title="Trading Lab", layout="wide")
store = RunStore()
service = BacktestService(store)
render_dashboard(store, service)
