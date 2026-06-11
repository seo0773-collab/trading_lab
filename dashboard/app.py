import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trading Lab Wave Viewer", layout="wide")
st.title("Trading Lab - Wave Viewer")
st.caption("Flat-chart waves, strategy timing, profile levels, and backtest diagnostics")

try:
    import yfinance as yf
except ImportError as exc:
    st.error(f"Required dashboard package is not installed: {exc}")
    st.code("python -m pip install -r requirements-backtest.txt")
    st.stop()

try:
    from dashboard.wave_viewer import (
        build_profile_figure,
        build_signal_table,
        build_wave_figure,
    )
    from strategies.cycle_reversion import run_cycle_reversion_backtest
except ImportError as exc:
    st.error(f"Dashboard module import failed: {exc}")
    st.stop()


@st.cache_data(ttl=900, show_spinner=False)
def download_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if getattr(data.columns, "nlevels", 1) > 1:
        data.columns = data.columns.get_level_values(0)
    return data


def finite_metric(value: float, suffix: str = "", digits: int = 2) -> str:
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "n/a"


with st.sidebar:
    st.header("Market data")
    ticker = st.text_input("Ticker", "SPY").strip().upper()
    start = st.date_input("Start date", date(2020, 1, 1))
    end = st.date_input("End date", date.today())

    st.header("Flat chart")
    mode = st.selectbox("Base-cycle mode", ["kalman", "sma"])
    length = st.slider("Base length", 20, 500, 100, 10)
    bins = st.slider("Profile bins", 50, 400, 200, 10)

    st.header("Backtest")
    init_cash = st.number_input("Initial cash", min_value=1000.0, value=10000.0, step=1000.0)
    fees_pct = st.number_input("Fee per order (%)", min_value=0.0, value=0.10, step=0.01, format="%.2f")
    slippage_pct = st.number_input("Slippage per order (%)", min_value=0.0, value=0.10, step=0.01, format="%.2f")
    partial_target_pct = st.slider("Position after partial exit (%)", 0, 90, 50, 5)
    refresh = st.button("Refresh market data", width="stretch")

if refresh:
    download_ohlcv.clear()
if not ticker:
    st.error("Ticker is required.")
    st.stop()
if start >= end:
    st.error("Start date must be earlier than end date.")
    st.stop()

try:
    with st.spinner(f"Downloading {ticker} OHLCV and running strategy..."):
        raw = download_ohlcv(ticker, start, end)
except Exception as exc:
    st.error(f"OHLCV download failed: {exc}")
    st.stop()

if raw.empty:
    st.error("No data returned. Check the ticker, date range, and network connection.")
    st.stop()

missing = [column for column in ["Open", "High", "Low", "Close"] if column not in raw]
if missing:
    st.error(f"Downloaded data is missing required columns: {missing}")
    st.stop()
raw = raw.dropna(subset=["Open", "High", "Low", "Close"])

try:
    portfolio, flat, profile, summary, targets = run_cycle_reversion_backtest(
        raw,
        mode=mode,
        length=length,
        bins=bins,
        fees=fees_pct / 100.0,
        slippage=slippage_pct / 100.0,
        init_cash=init_cash,
        partial_target=partial_target_pct / 100.0,
    )
    orders = portfolio.orders.records_readable.copy()
    trades = portfolio.trades.records_readable.copy()
    signals = build_signal_table(flat, targets, orders)
    stats = portfolio.stats()
except Exception as exc:
    st.error(f"Strategy calculation failed: {exc}")
    st.stop()

end_value = float(portfolio.value().iloc[-1])
total_return = (end_value / init_cash - 1.0) * 100.0
buy_hold_return = (float(raw["Close"].iloc[-1]) / float(raw["Close"].iloc[0]) - 1.0) * 100.0

metric_columns = st.columns(6)
metric_values = [
    ("Ending value", f"${end_value:,.2f}"),
    ("Strategy return", finite_metric(total_return, "%")),
    ("Buy & hold", finite_metric(buy_hold_return, "%")),
    ("Max drawdown", finite_metric(float(stats["Max Drawdown [%]"]), "%")),
    ("Win rate", finite_metric(float(stats["Win Rate [%]"]), "%") if pd.notna(stats["Win Rate [%]"]) else "n/a"),
    ("Orders", str(len(orders))),
]
for column, (label, value) in zip(metric_columns, metric_values):
    column.metric(label, value)

st.warning(
    "Research baseline: profile thresholds currently use the full selected interval. "
    "Signals therefore contain look-ahead bias and must not be treated as live trading signals."
)

wave_tab, profile_tab, trades_tab, data_tab = st.tabs(
    ["Wave Viewer", "Volume Profile", "Signals & Trades", "Calculated Data"]
)

with wave_tab:
    if signals.empty:
        st.info("No strategy signals were generated for the selected settings.")
    wave_figure = build_wave_figure(flat, summary, targets, portfolio, signals)
    st.plotly_chart(wave_figure, width="stretch", config={"scrollZoom": True})
    st.caption(
        "Green triangle: buy, yellow diamond: partial exit, red triangle: full exit. "
        "Drag to zoom, double-click to reset, and use the range buttons below the chart."
    )

with profile_tab:
    profile_metrics = st.columns(5)
    for column, (label, value) in zip(
        profile_metrics,
        [
            ("Lower", summary["lower_percentile"]),
            ("POC", summary["poc"]),
            ("Upper", summary["upper_percentile"]),
            ("Gaussian mu", summary["mu"]),
            ("Gaussian sigma", summary["sigma"]),
        ],
    ):
        column.metric(label, finite_metric(value, digits=4))
    st.plotly_chart(build_profile_figure(profile, summary), width="stretch")
    st.dataframe(profile, width="stretch", hide_index=True)

with trades_tab:
    st.subheader("Strategy signal events")
    if signals.empty:
        st.info("No signal events.")
    else:
        display_signals = signals.copy()
        display_signals["Target"] = display_signals["Target"] * 100.0
        display_signals = display_signals.rename(columns={"Target": "Target %"})
        st.dataframe(display_signals.sort_values("Timestamp", ascending=False), width="stretch", hide_index=True)

    st.subheader("Executed orders")
    st.dataframe(orders.sort_values("Timestamp", ascending=False), width="stretch", hide_index=True)
    st.subheader("Trade legs")
    st.dataframe(trades.sort_values("Entry Timestamp", ascending=False), width="stretch", hide_index=True)

with data_tab:
    calculated_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "base_cycle",
        "cm_open",
        "cm_high",
        "cm_low",
        "cm_close",
    ]
    available_columns = [column for column in calculated_columns if column in flat]
    st.dataframe(flat[available_columns].sort_index(ascending=False), width="stretch")
