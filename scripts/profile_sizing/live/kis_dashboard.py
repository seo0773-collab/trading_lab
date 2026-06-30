"""KIS 모의투자 · 한투 계좌 모니터 대시보드 (standalone Streamlit).

백테스트 대시보드(run_dashboard.sh)와 분리된 **라이브/모의 모니터**. 공통 인프라
(app.py/navigation.py)를 건드리지 않는다. 다크 미니멀 테마는 repo 루트
.streamlit/config.toml 이 적용되므로 **반드시 repo 루트에서 실행**:

    streamlit run scripts/profile_sizing/live/kis_dashboard.py

읽기 전용(잔고/시세 조회만, 주문 없음). 시세=실키, 잔고=모의키(.env).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

from profile_sizing.live.execution_layer import integer_execution_plan  # noqa: E402
from profile_sizing.live.kis_client import KISClient, KISConfigError  # noqa: E402
from profile_sizing.live.rebalance import load_latest_target  # noqa: E402

st.set_page_config(page_title="KIS 모의 모니터", layout="wide")
JDIR = ROOT / "var" / "paper_trading"
CFG_DIR = ROOT / "configs" / "strategies"


# --------------------------------------------------------------------- clients
@st.cache_resource
def get_clients():
    data = KISClient.from_env("KIS", env="real")
    try:
        order = KISClient.from_env("KIS_MOCK", env="mock")
    except KISConfigError:
        order = None
    return data, order


@st.cache_data(ttl=120)
def fx_rate() -> float:
    from trading_lab.market_data import load_cumulative_yfinance
    try:
        return float(load_cumulative_yfinance("KRW=X", "1d", "5d")["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return float("nan")


@st.cache_data(ttl=60)
def mock_domestic_balance() -> dict | None:
    _, order = get_clients()
    if order is None:
        return None
    try:
        return order.domestic_balance()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


@st.cache_data(ttl=300)
def last_close(symbol: str, domestic: bool, exch: str | None) -> float:
    data, _ = get_clients()
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    try:
        df = data.daily_history(symbol, interval="1d", start=start,
                                exchange=exch or "NASD")
        return float(df["close"].iloc[-1]) if not df.empty else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


@st.cache_data(ttl=3600)
def spy_history(period: str) -> pd.Series:
    from trading_lab.market_data import load_cumulative_yfinance

    df = load_cumulative_yfinance("SPY", "1d", period)
    close = df["close"].dropna().astype(float)
    close.index = pd.DatetimeIndex(close.index).normalize()
    return close[~close.index.duplicated(keep="last")].sort_index()


def load_journal(strategy: str) -> pd.DataFrame:
    import json
    path = JDIR / f"{strategy}_journal.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["run_at"] = pd.to_datetime(df["run_at"])
    return df.sort_values("run_at")


# --------------------------------------------------------------------- sections
def section_status():
    data, order = get_clients()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("시세 도메인", "실데이터" if "openapi.korea" in data.base else data.base)
    c2.metric("주문 도메인", "모의" if order and "openapivts" in order.base else "미설정")
    fx = fx_rate()
    c3.metric("USD/KRW", f"{fx:,.1f}" if fx == fx else "n/a")
    c4.metric("모의 계좌", order.cano if order else "—")


def _holdings_table(raw: dict) -> pd.DataFrame:
    rows = []
    for r in (raw.get("output1") or []):
        qty = float(r.get("hldg_qty", 0) or 0)
        if qty <= 0:
            continue
        rows.append({
            "종목": r.get("prdt_name") or r.get("pdno"),
            "코드": r.get("pdno"),
            "보유": qty,
            "현재가": float(r.get("prpr", 0) or 0),
            "평가금액": float(r.get("evlu_amt", 0) or 0),
            "매입금액": float(r.get("pchs_amt", 0) or 0),
            "평가손익": float(r.get("evlu_pfls_amt", 0) or 0),
            "수익률%": float(r.get("evlu_pfls_rt", 0) or 0),
        })
    return pd.DataFrame(rows)


def _account_nav(bal: dict | None) -> float | None:
    if not bal or "error" in bal:
        return None
    cash = float(bal["cash"])
    tbl = _holdings_table(bal["raw"])
    stock_val = float(tbl["평가금액"].sum()) if not tbl.empty else 0.0
    return cash + stock_val


def sidebar_investment_inputs(bal: dict | None) -> tuple[float, float]:
    default_principal = _account_nav(bal) or 10_000_000.0
    st.sidebar.header("투자 입력")
    principal = st.sidebar.number_input(
        "투자 원금(원)",
        min_value=0.0,
        value=float(default_principal),
        step=1_000_000.0,
        format="%.0f",
    )
    monthly = st.sidebar.number_input(
        "월 적립식 금액(원)",
        min_value=0.0,
        value=0.0,
        step=100_000.0,
        format="%.0f",
    )
    return float(principal), float(monthly)


def section_account(bal: dict | None):
    st.subheader("모의계좌 현황 (국내)")
    if bal is None:
        st.warning("모의 자격증명 없음 — .env 에 KIS_MOCK_APP_KEY/SECRET/ACCOUNT 필요.")
        return None
    if "error" in bal:
        st.error(f"잔고 조회 실패: {bal['error']}")
        return None
    cash = float(bal["cash"])
    tbl = _holdings_table(bal["raw"])
    stock_val = float(tbl["평가금액"].sum()) if not tbl.empty else 0.0
    nav = cash + stock_val
    pl = float(tbl["평가손익"].sum()) if not tbl.empty else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 평가(NAV)", f"{nav:,.0f}원")
    c2.metric("예수금", f"{cash:,.0f}원")
    c3.metric("주식 평가", f"{stock_val:,.0f}원")
    c4.metric("평가손익", f"{pl:,.0f}원", f"{(pl/stock_val*100 if stock_val else 0):.2f}%")
    if tbl.empty:
        st.info("보유 종목 없음(현금 100%).")
    else:
        st.dataframe(tbl, width="stretch", hide_index=True)
    return bal


def section_rebalance(bal: dict | None, total_krw: float):
    st.subheader("yoon1k 목표 vs 현재 · 리밸런스 주문표")
    strat = st.selectbox("전략", ["yoon2", "yoon1k_lp", "yoon1k"], index=0)
    cfg_path = CFG_DIR / f"{strat}.json"
    if not cfg_path.exists():
        st.error(f"config 없음: {strat}")
        return
    import json
    cfg = json.loads(cfg_path.read_text("utf-8"))
    fx = fx_rate()
    wsum = sum(float(s.get("weight", 0)) for s in cfg["sub_strategies"]) or 1.0

    for sub in cfg["sub_strategies"]:
        cid, cur = sub["config"], str(sub.get("currency", "KRW"))
        domestic = cur == "KRW"
        exch = None if domestic else "AMEX"
        jpath = JDIR / f"{cid}_journal.jsonl"
        if not jpath.exists():
            st.warning(f"[{cid}] 저널 없음 → paper_trade.py --strategy {cid} 먼저 실행")
            continue
        snap = load_latest_target(jpath)
        targets = {(s.split('.')[0] if domestic else s.upper()): w
                   for s, w in snap["targets"].items()}
        budget_krw = total_krw * float(sub.get("weight", 0)) / wsum
        cash = budget_krw if domestic else (budget_krw / fx if fx == fx else 0)
        # 현재 보유. 국내/해외 모두 같은 KIS_MOCK 모의 계좌를 사용한다.
        holdings = {}
        if domestic and bal:
            for r in (bal["raw"].get("output1") or []):
                q = float(r.get("hldg_qty", 0) or 0)
                if q:
                    holdings[str(r.get("pdno"))] = q
        elif not domestic:
            _, order = get_clients()
            if order is not None:
                try:
                    holdings = {str(k).upper(): float(v)
                                for k, v in order.balance(exchange=exch or "NASD")["holdings"].items()}
                except Exception:  # noqa: BLE001
                    holdings = {}
        prices = {s: last_close(s, domestic, exch) for s in set(targets) | set(holdings)}
        plan = integer_execution_plan(
            targets, holdings, prices, cash,
            min_trade_value=(50000.0 if domestic else 50.0),
        )
        unit = "원" if domestic else "$"
        tag = "모의 주문가능 ✅"
        st.markdown(f"**{cid}** [{cur}] · 배정 {cash:,.0f}{unit} · {tag} "
                    f"· 목표 {snap['data_through']} · 레짐 {snap.get('market_regime','-')}")
        st.caption(
            f"정수주 실행 예상: 현금 {plan['expected_cash']:,.0f}{unit} · "
            f"목표 재현율 {plan['target_fit']*100:.1f}% · "
            f"비중오차합 {plan['tracking_error_abs']*100:.2f}%p"
        )
        if plan["orders"]:
            od = pd.DataFrame(plan["orders"])[
                ["symbol", "side", "qty", "price", "est_value", "target_w"]]
            st.dataframe(od, width="stretch", hide_index=True)
        else:
            st.caption("조정 주문 없음.")
        rows = pd.DataFrame(plan.get("execution_rows", []))
        if not rows.empty:
            show_cols = ["symbol", "target_w", "expected_w", "error_w",
                         "expected_qty", "expected_value"]
            st.dataframe(rows[show_cols], width="stretch", hide_index=True)


def section_journal():
    st.subheader("페이퍼 저널 추이")
    jfiles = sorted(p.stem.replace("_journal", "") for p in JDIR.glob("*_journal.jsonl"))
    if not jfiles:
        st.info("저널 없음.")
        return
    pick = st.multiselect("전략(저널)", jfiles,
                          default=[s for s in ("yoon1j_lp", "yoon1j_kr") if s in jfiles]
                          or jfiles[:2])
    for s in pick:
        df = load_journal(s)
        if df.empty:
            continue
        st.markdown(f"**{s}** — 스냅샷 {len(df)}개 "
                    f"({df['run_at'].min():%Y-%m-%d}~{df['run_at'].max():%Y-%m-%d})")
        cols = [c for c in ("exposure", "cash", "n_holdings") if c in df.columns]
        if cols:
            chart = df.set_index("run_at")[cols]
            st.line_chart(chart, height=180)
        if "market_regime" in df.columns:
            st.caption("레짐: " + " → ".join(df["market_regime"].astype(str).tail(8)))


def _comparison_frame(
    close: pd.Series,
    initial_krw: float,
    monthly_krw: float,
    annual_rate: float = 0.08,
) -> pd.DataFrame:
    contrib = pd.Series(0.0, index=close.index)
    contrib.iloc[0] = initial_krw

    if monthly_krw > 0:
        month_starts = close.groupby(close.index.to_period("M")).head(1).index
        month_starts = month_starts[month_starts > close.index[0]]
        contrib.loc[month_starts] += monthly_krw

    principal = contrib.cumsum()
    spy_value = (contrib / close).cumsum() * close

    fixed_values = []
    fixed_value = 0.0
    prev_date = None
    for current_date, contribution in contrib.items():
        if prev_date is not None:
            days = max((current_date - prev_date).days, 0)
            fixed_value *= (1.0 + annual_rate) ** (days / 365.25)
        fixed_value += float(contribution)
        fixed_values.append(fixed_value)
        prev_date = current_date

    return pd.DataFrame(
        {
            "투자원금": principal,
            "SPY B&H": spy_value,
            "연 8% 증가": fixed_values,
        },
        index=close.index,
    )


def section_graph(initial_krw: float, monthly_krw: float):
    st.subheader("투자원금 · SPY B&H · 연 8% 비교")
    period = st.selectbox("SPY 데이터 범위", ["1y", "3y", "5y", "10y", "max"], index=2)

    try:
        close = spy_history(period)
    except Exception as e:  # noqa: BLE001
        st.error(f"SPY 데이터 로드 실패: {type(e).__name__}: {e}")
        return
    if close.empty:
        st.info("SPY 데이터가 없습니다.")
        return

    min_date = close.index.min().date()
    max_date = close.index.max().date()
    start_date = st.date_input(
        "그래프 시작일",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
    )
    close = close.loc[close.index >= pd.Timestamp(start_date)]
    if close.empty:
        st.info("선택한 시작일 이후 SPY 데이터가 없습니다.")
        return

    frame = _comparison_frame(close, initial_krw, monthly_krw)
    latest = frame.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("투자원금", f"{latest['투자원금']:,.0f}원")
    c2.metric("SPY B&H", f"{latest['SPY B&H']:,.0f}원")
    c3.metric("연 8% 증가", f"{latest['연 8% 증가']:,.0f}원")

    fig = go.Figure()
    for name in frame.columns:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[name], mode="lines", name=name))
    fig.update_layout(
        height=520,
        hovermode="x unified",
        margin=dict(l=16, r=16, t=24, b=16),
        yaxis=dict(title="금액(원)", tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📈 KIS 모의투자 · 한투 계좌 모니터")
    if st.button("🔄 새로고침(캐시 비움)"):
        st.cache_data.clear()
    st.caption("읽기 전용 — 잔고·시세 조회만(주문 없음). 시세=실키 / 잔고=모의키.")
    try:
        get_clients()
    except KISConfigError as e:
        st.error(f"자격증명 오류: {e}")
        st.stop()
    bal = mock_domestic_balance()
    initial_krw, monthly_krw = sidebar_investment_inputs(bal)

    account_tab, rebalance_tab, journal_tab, graph_tab = st.tabs(
        ["계좌", "리밸런스", "저널", "그래프"]
    )
    with account_tab:
        section_status()
        st.divider()
        section_account(bal)
    with rebalance_tab:
        section_rebalance(bal, initial_krw)
    with journal_tab:
        section_journal()
    with graph_tab:
        section_graph(initial_krw, monthly_krw)


if __name__ in ("__main__", "__page__"):
    main()
