"""2026-01-01 시작 QQQ 단일종목 정수주 DCA 시뮬레이션 (벤치마크 비교용).

yoon2_final_fx / yoon2m과 동일한 조건(초기 200만원 + 월 20만원, 정수주,
US 편도비용·최소주문)으로 QQQ를 매수만 하는 단순 적립식을 시뮬레이션한다.
기간/FX 캘린더는 yoon2_final_fx 시뮬레이션과 동일하게 맞춘다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/qqq_dca_2026.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_kalman_pipeline import load_yfinance  # noqa: E402
from yoon2_final_fx_integer_dca import (  # noqa: E402
    COST_US,
    INITIAL_KRW,
    MIN_US_ORDER_USD,
    MONTHLY_KRW,
    account_mdd,
    build_leaf_targets,
    metric_nav,
    money,
    pct,
)

START_DATE = "2026-01-01"
SYMBOL = "QQQ"


def simulate() -> dict:
    data = build_leaf_targets("yoon2_final_fx")
    idx_full: pd.DatetimeIndex = data["idx"]
    idx = idx_full[idx_full >= pd.Timestamp(START_DATE)]
    fx = data["fx"].reindex(idx).ffill()
    price_usd = load_yfinance(SYMBOL, "1d", "max")["close"].reindex(idx).ffill()
    price_krw = price_usd * fx

    month_first = ~pd.Series(idx, index=idx).index.to_period("M").duplicated()

    shares = 0.0
    cash = 0.0
    invested = 0.0
    cost_paid = 0.0
    order_count = 0
    account_values, invested_values, twr_returns = [], [], []
    prev_account = None

    for i, t in enumerate(idx):
        contribution = 0.0
        if i == 0:
            contribution += INITIAL_KRW
        if bool(month_first[i]):
            contribution += MONTHLY_KRW
        cash += contribution
        invested += contribution

        px = float(price_krw.loc[t])
        fx_t = float(fx.loc[t])
        min_order_krw = MIN_US_ORDER_USD * fx_t
        qty = math.floor(cash / (px * (1.0 + COST_US)))
        if qty > 0 and qty * px >= min_order_krw:
            cost = qty * px * COST_US
            cash -= qty * px + cost
            cost_paid += cost
            shares += qty
            order_count += 1

        account = shares * px + cash
        if prev_account is None or prev_account <= 0.0:
            twr_returns.append(0.0)
        else:
            twr_returns.append((account - contribution) / prev_account - 1.0)
        prev_account = account
        account_values.append(account)
        invested_values.append(invested)

    account_series = pd.Series(account_values, index=idx, name="account")
    invested_series = pd.Series(invested_values, index=idx, name="invested")
    twr = pd.Series(twr_returns, index=idx, name="twr_ret")
    nav = (1.0 + twr).cumprod()
    m = metric_nav(nav)

    final_px = float(price_krw.loc[idx[-1]])
    final_value = shares * final_px + cash

    return {
        "idx": idx,
        "metrics": m,
        "account_mdd": account_mdd(account_series),
        "final_value": float(final_value),
        "invested_principal": float(invested_series.iloc[-1]),
        "profit": float(final_value - invested_series.iloc[-1]),
        "return_on_principal": float(final_value / invested_series.iloc[-1] - 1.0),
        "shares": shares,
        "price_krw": final_px,
        "value_krw": shares * final_px,
        "cash": float(cash),
        "cash_weight": float(cash / final_value) if final_value > 0 else 0.0,
        "order_count": order_count,
        "cost_paid": float(cost_paid),
    }


def write_outputs(sim: dict) -> tuple[Path, Path]:
    idx = sim["idx"]
    period = f"{idx[0].date()}~{idx[-1].date()}"
    summary = {
        "period": period,
        "bars": len(idx),
        "initial_krw": INITIAL_KRW,
        "monthly_krw": MONTHLY_KRW,
        "invested_principal": sim["invested_principal"],
        "final_value": sim["final_value"],
        "profit": sim["profit"],
        "return_on_principal": sim["return_on_principal"],
        "cagr": sim["metrics"]["cagr"],
        "twr_mdd": sim["metrics"]["mdd"],
        "twr_sharpe": sim["metrics"]["sharpe"],
        "account_mdd": sim["account_mdd"],
        "shares": sim["shares"],
        "price_krw": sim["price_krw"],
        "value_krw": sim["value_krw"],
        "cash_krw": sim["cash"],
        "cash_weight": sim["cash_weight"],
        "order_count": sim["order_count"],
        "cost_paid": sim["cost_paid"],
        "cost_us": COST_US,
        "min_us_order_usd": MIN_US_ORDER_USD,
    }

    out_json = ROOT / "reports" / "profile_sizing" / "qqq_dca_2026.json"
    out_md = ROOT / "reports" / "profile_sizing" / "qqq_dca_2026.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# QQQ 2026년 적립식 DCA 시뮬레이션 (벤치마크)",
        "",
        "## 가정",
        "",
        f"- 기간: `{period}` ({summary['bars']} bars)",
        f"- 초기 원금: `{money(INITIAL_KRW)}` (첫 거래일 납입)",
        f"- 월 적립금: `{money(MONTHLY_KRW)}` (매월 첫 거래일)",
        f"- 누적 투입원금: `{money(summary['invested_principal'])}`",
        f"- 정수주, 편도 비용: US `{pct(COST_US)}`, 최소 주문금액: `${MIN_US_ORDER_USD:.0f}`",
        "",
        "## 평가자산 수익률",
        "",
        "| 누적원금 | 최종평가액 | 손익 | 원금대비 수익률 | 계좌 MDD | TWR CAGR | TWR Sharpe |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {money(summary['invested_principal'])} | {money(summary['final_value'])} | "
            f"{money(summary['profit'])} | {pct(summary['return_on_principal'])} | "
            f"{pct(summary['account_mdd'])} | {pct(summary['cagr'])} | {summary['twr_sharpe']:.3f} |"
        ),
        "",
        f"- 주문 수: `{summary['order_count']}`",
        f"- 총 비용: `{money(summary['cost_paid'])}`",
        f"- 현금 보유: `{money(summary['cash_krw'])}` (`{pct(summary['cash_weight'])}`)",
        "",
        "## 계좌 종목 현황 (최신 거래일 기준)",
        "",
        "| 종목 | 통화 | 보유수량 | 현재가(KRW) | 평가액(KRW) | 비중 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| {SYMBOL} | USD | {summary['shares']:.0f} | {money(summary['price_krw'])} | "
            f"{money(summary['value_krw'])} | {pct(1.0 - summary['cash_weight'])} |"
        ),
        f"| **현금** | KRW | - | - | {money(summary['cash_krw'])} | {pct(summary['cash_weight'])} |",
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def main() -> None:
    sim = simulate()
    out_json, out_md = write_outputs(sim)
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
