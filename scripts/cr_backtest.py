"""cr_plan.txt 전략 백테스트: SPY 70:30 + 낙폭 단계별 현금 투입 + 반등 이익실현.

규칙 (cr_plan.txt):
- 기본 배분 주식:현금 = 70:30 (시작 시점에 70% 매수).
- 역대 고점 대비 종가 기준 -10% 도달 시 현금의 15% 투입.
- -20% 도달 시 남은 현금의 30% 투입.
- -30% 도달 시 남은 현금 전부 투입.
- 각 티어는 신고점 갱신으로 리셋되기 전까지 1회만 발동.
- 주식 비중이 70%를 초과할 때, 낙폭 매수 트랜치가 매입가 대비 +10% 이상이면
  해당 트랜치의 50% 한도 내에서 70:30 복원에 필요한 만큼 매도.
  트랜치당 이익실현은 1회이며, 매도 후 잔여분은 코어 보유분으로 편입된다.

가정: 종가 시그널·종가 체결, 수수료/세금/배당/현금이자 0, 가격 데이터만 사용.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIERS = [(-0.10, 0.15), (-0.20, 0.30), (-0.30, 1.00)]  # (낙폭, 현금 투입 비율)
TARGET_STOCK_WEIGHT = 0.70
TAKE_PROFIT = 0.10
MAX_TRANCHE_SELL = 0.50


@dataclass
class Tranche:
    price: float
    shares: float


def run_strategy(close: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    cash = 0.30
    shares = TARGET_STOCK_WEIGHT / close.iloc[0]
    peak = close.iloc[0]
    fired = set()  # 현재 낙폭 에피소드에서 이미 발동한 티어 인덱스
    tranches: list[Tranche] = []
    equity = np.empty(len(close))
    events = []

    for i, (ts, px) in enumerate(close.items()):
        if px > peak:
            peak = px
            fired.clear()

        dd = px / peak - 1.0
        for idx, (level, frac) in enumerate(TIERS):
            if dd <= level and idx not in fired and cash > 1e-12:
                fired.add(idx)
                spend = cash * frac
                cash -= spend
                bought = spend / px
                shares += bought
                tranches.append(Tranche(px, bought))
                events.append((ts, f"buy_dd{int(-level*100)}", px, spend))

        nav = cash + shares * px
        stock_weight = shares * px / nav
        if stock_weight > TARGET_STOCK_WEIGHT and tranches:
            excess_value = shares * px - TARGET_STOCK_WEIGHT * nav
            done = []
            for tr in reversed(tranches):  # 최근 매입분부터
                if excess_value <= 1e-12:
                    break
                if px / tr.price - 1.0 < TAKE_PROFIT:
                    continue
                sellable = tr.shares * MAX_TRANCHE_SELL
                sell = min(sellable, excess_value / px)
                shares -= sell
                cash += sell * px
                excess_value -= sell * px
                done.append(tr)  # 트랜치당 1회 실현 후 잔여분은 코어로 편입
                events.append((ts, "sell_tp", px, sell * px))
            tranches = [t for t in tranches if t not in done]

        equity[i] = cash + shares * px

    eq = pd.Series(equity, index=close.index, name="cr")
    ev = pd.DataFrame(events, columns=["time", "action", "price", "value"])
    return eq, ev


def metrics(eq: pd.Series) -> dict:
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    return {
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
    }


def report_window(close: pd.Series, label: str) -> tuple[str, pd.DataFrame]:
    eq_cr, ev = run_strategy(close)
    eq_bh = close / close.iloc[0]
    static = 0.70 * close / close.iloc[0] + 0.30  # 70:30 방치(리밸런스 없음)
    rows = {
        "CR 전략": metrics(eq_cr),
        "B&H (SPY 100%)": metrics(eq_bh),
        "70:30 방치": metrics(static),
    }
    df = pd.DataFrame(rows).T
    buys = ev[ev.action.str.startswith("buy")]
    sells = ev[ev.action == "sell_tp"]
    lines = [f"### {label} ({close.index[0].date()} ~ {close.index[-1].date()})", ""]
    lines.append("| 전략 | 총수익 | CAGR | MDD | Sharpe |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, m in df.iterrows():
        lines.append(
            f"| {name} | {m.total_return*100:,.1f}% | {m.cagr*100:.2f}% "
            f"| {m.mdd*100:.1f}% | {m.sharpe:.3f} |"
        )
    lines.append("")
    lines.append(
        f"낙폭 매수 {len(buys)}회 (-10%: {len(ev[ev.action=='buy_dd10'])}, "
        f"-20%: {len(ev[ev.action=='buy_dd20'])}, -30%: {len(ev[ev.action=='buy_dd30'])}), "
        f"이익실현 매도 {len(sells)}회"
    )
    lines.append("")
    return "\n".join(lines), ev


def main() -> None:
    df = pd.read_parquet("var/market_data/1d/SPY.parquet")
    close = df["close"].dropna()

    windows = [
        ("전체 기간", close),
        ("2000년 이후 (닷컴+GFC 포함)", close.loc["2000-01-01":]),
        ("2010년 이후 (강세장 위주)", close.loc["2010-01-01":]),
        ("최근 10년", close.loc["2016-07-01":]),
    ]

    out = ["# CR 전략 (cr_plan.txt) 백테스트 — SPY 일봉", ""]
    out.append(
        "규칙: 70:30 기본, 고점대비 -10%/-20%/-30%에서 현금 15%/30%/100% 단계 투입, "
        "주식비중>70% & 매수 트랜치 +10% 시 트랜치 50% 한도 매도로 현금 복원. "
        "종가 체결, 비용·배당·현금이자 0."
    )
    out.append("")
    for label, series in windows:
        section, ev = report_window(series, label)
        out.append(section)
        if label == "전체 기간":
            full_events = ev

    out.append("### 전체 기간 이벤트 로그 (최근 20건)")
    out.append("")
    out.append(full_events.tail(20).to_string(index=False))
    text = "\n".join(out)
    print(text)
    with open("reports/cr_plan_backtest.md", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
