"""2026-01-01 시작 정수주 DCA 시뮬레이션 (yoon1k 계열: yoon2_final_fx, yoon2m 등).

기존 yoon2_final_fx_integer_dca.py는 전체 히스토리(2007~)를 시뮬레이션한다.
이 스크립트는 동일한 실행 로직(정수주·KIS 실비용·최소주문)을 재사용하되,
기간을 2026-01-01~최신 거래일로 한정하고 최종 보유 종목 현황을 함께 출력한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_dca_2026.py [strategy_id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from yoon2_final_fx_integer_dca import (  # noqa: E402
    COST_KR,
    COST_US,
    INITIAL_KRW,
    MIN_KR_ORDER,
    MIN_US_ORDER_USD,
    MONTHLY_KRW,
    account_mdd,
    build_leaf_targets,
    current_account,
    current_weight_frame,
    metric_nav,
    money,
    pct,
    rebalance_integer,
)

START_DATE = "2026-01-01"


def simulate(data: dict, start_date: str) -> dict:
    idx_full: pd.DatetimeIndex = data["idx"]
    idx = idx_full[idx_full >= pd.Timestamp(start_date)]
    target = data["target"].reindex(idx)
    price = data["price"].reindex(idx)
    fx = data["fx"].reindex(idx)
    asset_currency = data["asset_currency"]
    symbols = sorted(asset_currency)

    month_first = ~pd.Series(idx, index=idx).index.to_period("M").duplicated()
    rebal = rebalance_dates(idx, "monthly")

    shares = {s: 0.0 for s in symbols}
    cash = 0.0
    account_values, invested_values, twr_returns = [], [], []
    prev_account = None
    invested = 0.0
    rebalance_count = 0
    order_count = 0
    cost_paid = 0.0
    turnover = 0.0

    for i, t in enumerate(idx):
        contribution = 0.0
        if i == 0:
            contribution += INITIAL_KRW
        if bool(month_first[i]):
            contribution += MONTHLY_KRW
        cash += contribution
        invested += contribution

        px = price.loc[t]
        account_before = current_account(shares, cash, px, symbols)
        if i == 0 or t in rebal:
            shares, cash, st = rebalance_integer(
                shares=shares,
                cash=cash,
                account=account_before,
                target=target.loc[t],
                prices=px,
                fx_value=float(fx.loc[t]),
                asset_currency=asset_currency,
            )
            turnover += st["traded"] / max(account_before, 1.0)
            cost_paid += st["cost_paid"]
            order_count += st["orders"]
            rebalance_count += 1

        account = current_account(shares, cash, px, symbols)
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

    last_px = price.loc[idx[-1]]
    final_account = current_account(shares, cash, last_px, symbols)
    weights = current_weight_frame(shares, cash, last_px, symbols, final_account)

    holdings = []
    for s in symbols:
        qty = shares[s]
        if qty <= 0.0:
            continue
        px_local = None
        raw_price = data.get("raw_local_price", {}).get(s)
        cur = asset_currency[s]
        value_krw = qty * float(last_px.get(s, np.nan)) if pd.notna(last_px.get(s)) else 0.0
        holdings.append({
            "symbol": s,
            "currency": cur,
            "shares": qty,
            "price_krw": float(last_px.get(s, np.nan)) if pd.notna(last_px.get(s)) else None,
            "value_krw": value_krw,
            "weight": float(weights.get(s, 0.0)),
        })
    holdings.sort(key=lambda h: -h["value_krw"])

    m = metric_nav(nav)
    return {
        "idx": idx,
        "account_series": account_series,
        "invested_series": invested_series,
        "nav": nav,
        "metrics": m,
        "account_mdd": account_mdd(account_series),
        "final_value": float(account_series.iloc[-1]),
        "invested_principal": float(invested_series.iloc[-1]),
        "profit": float(account_series.iloc[-1] - invested_series.iloc[-1]),
        "return_on_principal": float(account_series.iloc[-1] / invested_series.iloc[-1] - 1.0),
        "cash": float(cash),
        "cash_weight": float(weights.get("__CASH__", 0.0)),
        "holdings": holdings,
        "rebalance_count": rebalance_count,
        "order_count": order_count,
        "turnover_per_year": turnover / max(len(idx) / 252.0, 1e-9),
        "cost_paid": float(cost_paid),
    }


def write_outputs(sim: dict, strategy_id: str) -> tuple[Path, Path]:
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
        "cash_krw": sim["cash"],
        "cash_weight": sim["cash_weight"],
        "rebalance_count": sim["rebalance_count"],
        "order_count": sim["order_count"],
        "turnover_per_year": sim["turnover_per_year"],
        "cost_paid": sim["cost_paid"],
        "holdings": sim["holdings"],
        "cost_kr": COST_KR,
        "cost_us": COST_US,
        "min_kr_order": MIN_KR_ORDER,
        "min_us_order_usd": MIN_US_ORDER_USD,
    }

    out_json = ROOT / "reports" / "profile_sizing" / f"{strategy_id}_dca_2026.json"
    out_md = ROOT / "reports" / "profile_sizing" / f"{strategy_id}_dca_2026.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        f"# {strategy_id} 2026년 적립식 DCA 시뮬레이션",
        "",
        "## 가정",
        "",
        f"- 기간: `{period}` ({summary['bars']} bars, 리밸런스: 월말)",
        f"- 초기 원금: `{money(INITIAL_KRW)}` (첫 거래일 납입)",
        f"- 월 적립금: `{money(MONTHLY_KRW)}` (매월 첫 거래일)",
        f"- 누적 투입원금: `{money(summary['invested_principal'])}`",
        f"- 정수주, 편도 비용: KR `{pct(COST_KR)}`, US `{pct(COST_US)}`",
        f"- 최소 주문금액: KR `{money(MIN_KR_ORDER)}`, US `${MIN_US_ORDER_USD:.0f}`",
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
        f"- 리밸런스 횟수/주문 수: `{summary['rebalance_count']}` / `{summary['order_count']}`",
        f"- 총 비용: `{money(summary['cost_paid'])}`",
        f"- 현금 보유: `{money(summary['cash_krw'])}` (`{pct(summary['cash_weight'])}`)",
        "",
        "## 계좌 종목 현황 (최신 거래일 기준)",
        "",
        "| 종목 | 통화 | 보유수량 | 현재가(KRW) | 평가액(KRW) | 비중 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for h in summary["holdings"]:
        lines.append(
            f"| {h['symbol']} | {h['currency']} | {h['shares']:.0f} | "
            f"{money(h['price_krw'])} | {money(h['value_krw'])} | {pct(h['weight'])} |"
        )
    lines.append(
        f"| **현금** | KRW | - | - | {money(summary['cash_krw'])} | {pct(summary['cash_weight'])} |"
    )
    lines.extend([
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def main() -> None:
    strategy_id = sys.argv[1] if len(sys.argv) > 1 else "yoon2_final_fx"
    data = build_leaf_targets(strategy_id)
    sim = simulate(data, START_DATE)
    out_json, out_md = write_outputs(sim, strategy_id)
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
