"""Integer-share DCA execution check for yoon2_final_fx.

Assumptions intentionally match the existing small-account execution probes:

- initial capital: KRW 2,000,000
- monthly contribution: KRW 200,000 on the first trading day of each month
- rebalance: month-end, same as the strategy
- integer shares only
- one-way cost: KR 0.15%, US 0.60%
- minimum order value: KRW 50,000, US $50 converted to KRW

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_integer_dca.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from trading_lab.ui.presentation import account_value_series, total_invested  # noqa: E402


INITIAL_KRW = 2_000_000.0
MONTHLY_KRW = 200_000.0
COST_KR = 0.000646527  # 0.0647%/side = KIS 국내 수수료 0.0147% + 슬리피지 0.05% (계좌 실조건 2026-07-05)
COST_US = 0.0025103    # 0.2510%/side = 수수료 0.1%우대+제비용 + 환전 0.1%(90%우대) + 슬리피지 0.05%
MIN_KR_ORDER = 50_000.0
MIN_US_ORDER_USD = 50.0


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def money(v: float | None) -> str:
    return "-" if v is None else f"{v:,.0f}원"


def metric_nav(nav: pd.Series) -> dict[str, float]:
    nav = pd.Series(nav).dropna().astype(float)
    eq = nav / nav.iloc[0]
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {
        "cagr": float(p["cagr"]),
        "mdd": float(p["max_drawdown"]),
        "sharpe": float(p["sharpe"]),
        "total_return": float(p["total_return"]),
        "volatility": float(p["volatility"]),
    }


def account_mdd(account: pd.Series) -> float:
    account = pd.Series(account).dropna().astype(float)
    return float((account / account.cummax() - 1.0).min())


def real_symbol(col: str) -> str:
    return col[:-5] if col.endswith("_SAFE") else col


def load_sub_wave(
    handler: Yoon1kHandler,
    raw: pd.DataFrame,
    sub: dict[str, Any],
) -> pd.DataFrame:
    sub_id = str(sub["config"])
    cfg = handler._sub_config(sub_id, sub.get("overrides"))
    art = ProfilePortfolioHandler().build_artifacts(
        raw[sub_id], cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252
    )
    wave = art.extras["portfolio_wave"]
    if "time" in wave.columns:
        wave = wave.set_index("time")
    wave.index = pd.to_datetime(wave.index)
    return wave.astype(float)


def build_leaf_targets(strategy_id: str = "yoon2_final_fx") -> dict[str, Any]:
    config = json.loads((ROOT / "configs" / "strategies" / f"{strategy_id}.json").read_text("utf-8"))
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)
    ideal_art = handler.build_artifacts(raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252)

    sub_ids = [str(s["config"]) for s in config["sub_strategies"]]
    idx = ideal_art.equity.index
    top_weights = handler._target_weight_frame(
        config,
        raw,
        pd.DatetimeIndex(idx),
        [float(s.get("weight", 1.0 / len(config["sub_strategies"]))) for s in config["sub_strategies"]],
    ).reindex(idx).ffill().bfill()

    leaf = pd.DataFrame(index=idx)
    cash = pd.Series(0.0, index=idx)
    asset_currency: dict[str, str] = {}
    for sub in config["sub_strategies"]:
        sid = str(sub["config"])
        cur = str(sub.get("currency", config.get("base_currency", "KRW")))
        wave = load_sub_wave(handler, raw, sub).reindex(idx).ffill().fillna(0.0)
        sleeve_w = top_weights[sid].reindex(idx).ffill().fillna(0.0)
        for col in wave.columns:
            weighted = sleeve_w * wave[col]
            if col == "cash":
                cash = cash.add(weighted, fill_value=0.0)
                continue
            sym = real_symbol(str(col))
            leaf[sym] = leaf.get(sym, pd.Series(0.0, index=idx)).add(weighted, fill_value=0.0)
            asset_currency.setdefault(sym, cur)

    leaf = leaf.fillna(0.0)
    cash = cash.add((1.0 - leaf.sum(axis=1) - cash).clip(lower=0.0), fill_value=0.0)
    fx = raw["__FX__"][str(config["fx_symbol"])]["close"].reindex(idx).ffill()

    prices = {}
    for sym, cur in asset_currency.items():
        try:
            px = load_yfinance(sym, "1d", "max")["close"].reindex(idx).ffill()
        except Exception:  # noqa: BLE001
            px = pd.Series(np.nan, index=idx)
        prices[sym] = px * fx if cur == "USD" else px
    price = pd.DataFrame(prices, index=idx)

    # Keep the full strategy period, but zero out impossible non-cash targets until
    # their price exists. This reflects an executable account holding residual cash.
    valid = price.notna() & (price > 0.0)
    leaf = leaf.where(valid, 0.0)
    leaf["__CASH__"] = (1.0 - leaf.sum(axis=1)).clip(lower=0.0)

    return {
        "config": config,
        "idx": pd.DatetimeIndex(idx),
        "ideal_equity": ideal_art.equity.reindex(idx).ffill(),
        "target": leaf,
        "price": price.reindex(idx).ffill(),
        "fx": fx,
        "asset_currency": asset_currency,
    }


def current_account(
    shares: dict[str, float],
    cash: float,
    prices: pd.Series,
    symbols: list[str],
) -> float:
    total = float(cash)
    for s in symbols:
        px = prices.get(s, np.nan)
        price = 0.0 if pd.isna(px) else float(px)
        total += shares[s] * price
    return float(total)


def current_weight_frame(
    shares: dict[str, float],
    cash: float,
    prices: pd.Series,
    symbols: list[str],
    account: float,
) -> pd.Series:
    if account <= 0.0:
        return pd.Series(0.0, index=[*symbols, "__CASH__"])
    vals = {}
    for s in symbols:
        px = prices.get(s, np.nan)
        price = 0.0 if pd.isna(px) else float(px)
        vals[s] = shares[s] * price / account
    vals["__CASH__"] = cash / account
    return pd.Series(vals)


def min_order_value(sym: str, cur: str, fx_value: float) -> float:
    return MIN_US_ORDER_USD * fx_value if cur == "USD" else MIN_KR_ORDER


def rebalance_integer(
    *,
    shares: dict[str, float],
    cash: float,
    account: float,
    target: pd.Series,
    prices: pd.Series,
    fx_value: float,
    asset_currency: dict[str, str],
) -> tuple[dict[str, float], float, dict[str, Any]]:
    symbols = sorted(asset_currency)
    avail = {
        s: float(prices[s])
        for s in symbols
        if pd.notna(prices.get(s)) and float(prices[s]) > 0.0
    }
    target = target.reindex([*symbols, "__CASH__"]).fillna(0.0)
    desired = {
        s: math.floor(max(account * float(target[s]), 0.0) / avail[s])
        for s in avail
    }

    def cost_rate(s: str) -> float:
        return COST_US if asset_currency[s] == "USD" else COST_KR

    def order_value_ok(s: str, from_qty: float, to_qty: float) -> bool:
        delta = abs(to_qty - from_qty) * avail[s]
        if delta <= 0.0:
            return False
        # Allow full liquidation of small residual positions; apply minimums to
        # buys and partial sells.
        if to_qty == 0.0 and from_qty > 0.0:
            return True
        return delta >= min_order_value(s, asset_currency[s], fx_value)

    for s in symbols:
        if s not in avail:
            desired[s] = shares[s]
            continue
        if not order_value_ok(s, shares[s], desired.get(s, 0.0)):
            desired[s] = shares[s]

    def cash_delta(next_shares: dict[str, float]) -> float:
        buy = sum(
            max(next_shares.get(s, 0.0) - shares[s], 0.0) * avail[s] * (1.0 + cost_rate(s))
            for s in avail
        )
        sell = sum(
            max(shares[s] - next_shares.get(s, 0.0), 0.0) * avail[s] * (1.0 - cost_rate(s))
            for s in avail
        )
        return buy - sell

    guard = 0
    while cash_delta(desired) > cash + 1e-6 and guard < 50_000:
        over = max(
            avail,
            key=lambda s: (
                desired.get(s, 0.0) * avail[s] / max(account, 1.0) - float(target[s])
                if desired.get(s, 0.0) > 0.0 else -9.0
            ),
        )
        if desired.get(over, 0.0) <= 0.0:
            break
        desired[over] -= 1.0
        guard += 1

    guard = 0
    while guard < 50_000:
        candidates = []
        for s in avail:
            nxt = desired.get(s, 0.0) + 1.0
            if not order_value_ok(s, desired.get(s, 0.0), nxt):
                continue
            if cash_delta({**desired, s: nxt}) <= cash + 1e-6:
                under = float(target[s]) - desired.get(s, 0.0) * avail[s] / max(account, 1.0)
                if under > 0.0:
                    candidates.append((under, s))
        if not candidates:
            break
        _, pick = max(candidates)
        desired[pick] = desired.get(pick, 0.0) + 1.0
        guard += 1

    traded = 0.0
    cost_paid = 0.0
    orders = 0
    skipped = []
    for s in avail:
        delta_qty = desired.get(s, 0.0) - shares[s]
        if abs(delta_qty) <= 0.0:
            if float(target[s]) > 0.0 and shares[s] * avail[s] < target[s] * account:
                one_share = avail[s] * (1.0 + cost_rate(s))
                if one_share > cash or avail[s] < min_order_value(s, asset_currency[s], fx_value):
                    skipped.append(s)
            continue
        value = abs(delta_qty) * avail[s]
        traded += value
        cost_paid += value * cost_rate(s)
        orders += 1

    new_cash = cash - cash_delta(desired)
    next_shares = {s: float(desired.get(s, 0.0)) for s in symbols}
    return next_shares, float(new_cash), {
        "traded": traded,
        "cost_paid": cost_paid,
        "orders": orders,
        "skipped": skipped,
    }


def simulate_integer_dca(data: dict[str, Any]) -> dict[str, Any]:
    idx: pd.DatetimeIndex = data["idx"]
    target: pd.DataFrame = data["target"]
    price: pd.DataFrame = data["price"]
    fx: pd.Series = data["fx"]
    asset_currency: dict[str, str] = data["asset_currency"]
    symbols = sorted(asset_currency)
    month_first = ~pd.Series(idx, index=idx).index.to_period("M").duplicated()
    rebal = rebalance_dates(idx, "monthly")

    shares = {s: 0.0 for s in symbols}
    cash = 0.0
    prev_account: float | None = None
    account_values = []
    invested_values = []
    twr_returns = []
    target_fits = []
    tracking_errors = []
    cash_weights = []
    skipped_counter: Counter[str] = Counter()
    turnover = 0.0
    cost_paid = 0.0
    order_count = 0
    rebalance_count = 0
    invested = 0.0

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
            skipped_counter.update(st["skipped"])
            rebalance_count += 1

        account = current_account(shares, cash, px, symbols)
        cw = current_weight_frame(shares, cash, px, symbols, account)
        tw = target.loc[t].reindex([*symbols, "__CASH__"]).fillna(0.0)
        tracking_error = float((cw - tw).abs().sum())
        tracking_errors.append(tracking_error)
        target_fits.append(max(0.0, 1.0 - tracking_error / 2.0))
        cash_weights.append(float(cw["__CASH__"]))

        if prev_account is None or prev_account <= 0.0:
            twr_returns.append(0.0)
        else:
            twr_returns.append((account - contribution) / prev_account - 1.0)
        prev_account = account
        account_values.append(account)
        invested_values.append(invested)

    account_series = pd.Series(account_values, index=idx, name="integer_account")
    invested_series = pd.Series(invested_values, index=idx, name="invested_principal")
    twr = pd.Series(twr_returns, index=idx, name="integer_twr_ret")
    integer_nav = (1.0 + twr).cumprod().rename("integer_twr_nav")
    m = metric_nav(integer_nav)
    return {
        "account": account_series,
        "invested": invested_series,
        "integer_nav": integer_nav,
        "metrics": m,
        "account_mdd": account_mdd(account_series),
        "final_value": float(account_series.iloc[-1]),
        "invested_principal": float(invested_series.iloc[-1]),
        "profit": float(account_series.iloc[-1] - invested_series.iloc[-1]),
        "return_on_principal": float(account_series.iloc[-1] / invested_series.iloc[-1] - 1.0),
        "rebalance_count": rebalance_count,
        "order_count": order_count,
        "turnover_sum": turnover,
        "turnover_per_year": turnover / (len(idx) / 252.0),
        "cost_paid": cost_paid,
        "cost_on_principal": cost_paid / float(invested_series.iloc[-1]),
        "avg_target_fit": float(np.mean(target_fits)),
        "min_target_fit": float(np.min(target_fits)),
        "avg_tracking_error": float(np.mean(tracking_errors)),
        "avg_cash_weight": float(np.mean(cash_weights)),
        "max_cash_weight": float(np.max(cash_weights)),
        "skipped_high_price": skipped_counter.most_common(12),
    }


def simulate_with_assumptions(
    data: dict[str, Any],
    *,
    cost_kr: float,
    cost_us: float,
    min_kr_order: float,
    min_us_order_usd: float,
) -> dict[str, Any]:
    global COST_KR, COST_US, MIN_KR_ORDER, MIN_US_ORDER_USD

    old = (COST_KR, COST_US, MIN_KR_ORDER, MIN_US_ORDER_USD)
    try:
        COST_KR = cost_kr
        COST_US = cost_us
        MIN_KR_ORDER = min_kr_order
        MIN_US_ORDER_USD = min_us_order_usd
        sim = simulate_integer_dca(data)
    finally:
        COST_KR, COST_US, MIN_KR_ORDER, MIN_US_ORDER_USD = old
    return {
        "final_value": sim["final_value"],
        "cagr": sim["metrics"]["cagr"],
        "mdd": sim["metrics"]["mdd"],
        "sharpe": sim["metrics"]["sharpe"],
        "account_mdd": sim["account_mdd"],
        "avg_target_fit": sim["avg_target_fit"],
        "avg_cash_weight": sim["avg_cash_weight"],
        "cost_paid": sim["cost_paid"],
    }


def write_outputs(data: dict[str, Any], sim: dict[str, Any]) -> tuple[Path, Path]:
    idx = data["idx"]
    ideal_eq = data["ideal_equity"].reindex(idx).ffill()
    ideal_account = account_value_series(
        ideal_eq / ideal_eq.iloc[0],
        INITIAL_KRW,
        monthly_contribution=MONTHLY_KRW,
    ).reindex(idx).ffill()
    ideal_metrics = metric_nav(ideal_eq / ideal_eq.iloc[0])
    ideal_final = float(ideal_account.iloc[-1])
    invested = total_invested(ideal_eq, INITIAL_KRW, MONTHLY_KRW)
    integer_metrics = sim["metrics"]
    corr = float(
        sim["integer_nav"].pct_change().fillna(0.0).corr(
            (ideal_eq / ideal_eq.iloc[0]).pct_change().fillna(0.0)
        )
    )
    account_diff = sim["account"] / ideal_account - 1.0
    sensitivity = {
        "base": {
            "cost_kr": COST_KR,
            "cost_us": COST_US,
            "min_kr_order": MIN_KR_ORDER,
            "min_us_order_usd": MIN_US_ORDER_USD,
            "result": {
                "final_value": sim["final_value"],
                "cagr": sim["metrics"]["cagr"],
                "mdd": sim["metrics"]["mdd"],
                "sharpe": sim["metrics"]["sharpe"],
                "account_mdd": sim["account_mdd"],
                "avg_target_fit": sim["avg_target_fit"],
                "avg_cash_weight": sim["avg_cash_weight"],
                "cost_paid": sim["cost_paid"],
            },
        },
        "no_cost": {
            "cost_kr": 0.0,
            "cost_us": 0.0,
            "min_kr_order": MIN_KR_ORDER,
            "min_us_order_usd": MIN_US_ORDER_USD,
            "result": simulate_with_assumptions(
                data,
                cost_kr=0.0,
                cost_us=0.0,
                min_kr_order=MIN_KR_ORDER,
                min_us_order_usd=MIN_US_ORDER_USD,
            ),
        },
        "no_min_order": {
            "cost_kr": COST_KR,
            "cost_us": COST_US,
            "min_kr_order": 0.0,
            "min_us_order_usd": 0.0,
            "result": simulate_with_assumptions(
                data,
                cost_kr=COST_KR,
                cost_us=COST_US,
                min_kr_order=0.0,
                min_us_order_usd=0.0,
            ),
        },
        "no_cost_no_min_order": {
            "cost_kr": 0.0,
            "cost_us": 0.0,
            "min_kr_order": 0.0,
            "min_us_order_usd": 0.0,
            "result": simulate_with_assumptions(
                data,
                cost_kr=0.0,
                cost_us=0.0,
                min_kr_order=0.0,
                min_us_order_usd=0.0,
            ),
        },
    }
    for item in sensitivity.values():
        item["result"]["final_vs_ideal"] = item["result"]["final_value"] / ideal_final - 1.0

    summary = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "initial_krw": INITIAL_KRW,
        "monthly_krw": MONTHLY_KRW,
        "invested_principal": float(invested),
        "cost_kr": COST_KR,
        "cost_us": COST_US,
        "min_kr_order": MIN_KR_ORDER,
        "min_us_order_usd": MIN_US_ORDER_USD,
        "ideal": {
            **ideal_metrics,
            "account_mdd": account_mdd(ideal_account),
            "final_value": ideal_final,
            "profit": ideal_final - invested,
            "return_on_principal": ideal_final / invested - 1.0,
        },
        "integer": {
            **integer_metrics,
            "account_mdd": sim["account_mdd"],
            "final_value": sim["final_value"],
            "profit": sim["profit"],
            "return_on_principal": sim["return_on_principal"],
            "rebalance_count": sim["rebalance_count"],
            "order_count": sim["order_count"],
            "turnover_per_year": sim["turnover_per_year"],
            "cost_paid": sim["cost_paid"],
            "cost_on_principal": sim["cost_on_principal"],
            "avg_target_fit": sim["avg_target_fit"],
            "min_target_fit": sim["min_target_fit"],
            "avg_tracking_error": sim["avg_tracking_error"],
            "avg_cash_weight": sim["avg_cash_weight"],
            "max_cash_weight": sim["max_cash_weight"],
        },
        "tracking": {
            "daily_return_corr": corr,
            "final_integer_vs_ideal": sim["final_value"] / ideal_final - 1.0,
            "avg_account_diff": float(account_diff.mean()),
            "min_account_diff": float(account_diff.min()),
            "max_account_diff": float(account_diff.max()),
        },
        "sensitivity": sensitivity,
        "skipped_high_price": sim["skipped_high_price"],
    }

    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_integer_dca_2m.json"
    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_integer_dca_2m.md"
    out_csv = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_integer_dca_2m.csv"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.DataFrame({
        "ideal_account": ideal_account,
        "integer_account": sim["account"],
        "invested_principal": sim["invested"],
        "ideal_nav": ideal_eq / ideal_eq.iloc[0],
        "integer_twr_nav": sim["integer_nav"],
        "account_diff_integer_vs_ideal": account_diff,
    }).to_csv(out_csv, index_label="date")

    lines = [
        "# yoon2_final_fx 정수주 DCA 실거래 검증",
        "",
        "## 가정",
        "",
        f"- 기간: `{summary['period']}` ({summary['bars']} bars)",
        f"- 초기 원금: `{money(INITIAL_KRW)}`",
        f"- 월 적립금: `{money(MONTHLY_KRW)}` (매월 첫 거래일, 첫 달도 납입)",
        f"- 누적 투입원금: `{money(summary['invested_principal'])}`",
        f"- 리밸런스: 월말, `yoon2_final_fx` 종목 목표비중 기준",
        f"- 정수주, 편도 비용: KR `{pct(COST_KR)}`, US `{pct(COST_US)}`",
        f"- 최소 주문금액: KR `{money(MIN_KR_ORDER)}`, US `${MIN_US_ORDER_USD:.0f}`",
        "",
        "## 결과",
        "",
        "| 구분 | 누적원금 | 최종평가액 | 손익 | 원금대비 수익률 | 계좌 MDD | TWR CAGR | TWR MDD | TWR Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 이상적 yoon2_final_fx | {money(summary['invested_principal'])} | "
            f"{money(summary['ideal']['final_value'])} | {money(summary['ideal']['profit'])} | "
            f"{pct(summary['ideal']['return_on_principal'])} | {pct(summary['ideal']['account_mdd'])} | "
            f"{pct(summary['ideal']['cagr'])} | {pct(summary['ideal']['mdd'])} | "
            f"{summary['ideal']['sharpe']:.3f} |"
        ),
        (
            f"| 정수주+비용 실행 | {money(summary['invested_principal'])} | "
            f"{money(summary['integer']['final_value'])} | {money(summary['integer']['profit'])} | "
            f"{pct(summary['integer']['return_on_principal'])} | {pct(summary['integer']['account_mdd'])} | "
            f"{pct(summary['integer']['cagr'])} | {pct(summary['integer']['mdd'])} | "
            f"{summary['integer']['sharpe']:.3f} |"
        ),
        "",
        "## 실행 재현성",
        "",
        f"- 일간 TWR 수익률 상관(이상적 vs 정수주): `{summary['tracking']['daily_return_corr']:.4f}`",
        f"- 최종평가액 차이(정수주/이상적 - 1): `{pct(summary['tracking']['final_integer_vs_ideal'])}`",
        f"- 평균 계좌 차이(정수주/이상적 - 1): `{pct(summary['tracking']['avg_account_diff'])}`",
        f"- 평균 목표 재현율: `{pct(summary['integer']['avg_target_fit'])}`",
        f"- 최저 목표 재현율: `{pct(summary['integer']['min_target_fit'])}`",
        f"- 평균 비중오차합: `{pct(summary['integer']['avg_tracking_error'])}p`",
        f"- 평균/최대 현금비중: `{pct(summary['integer']['avg_cash_weight'])}` / `{pct(summary['integer']['max_cash_weight'])}`",
        f"- 리밸런스 횟수/주문 수: `{summary['integer']['rebalance_count']}` / `{summary['integer']['order_count']}`",
        f"- 연환산 회전율: `{summary['integer']['turnover_per_year']:.2f}x`",
        f"- 총 비용: `{money(summary['integer']['cost_paid'])}` (`{pct(summary['integer']['cost_on_principal'])}` of principal)",
        "- 주당가격/현금 제약 빈도 상위: "
        + (", ".join(f"{s} {n}회" for s, n in summary["skipped_high_price"]) or "없음"),
        "",
        "## 비용/최소주문 민감도",
        "",
        "| 시나리오 | 최종평가액 | 이상적 대비 | TWR CAGR | TWR MDD | TWR Sharpe | 평균 재현율 | 평균 현금 | 비용 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "base": "기본(비용+최소주문)",
        "no_cost": "무비용",
        "no_min_order": "최소주문 제거",
        "no_cost_no_min_order": "무비용+최소주문 제거",
    }
    for key in ["base", "no_cost", "no_min_order", "no_cost_no_min_order"]:
        r = summary["sensitivity"][key]["result"]
        lines.append(
            f"| {labels[key]} | {money(r['final_value'])} | {pct(r['final_vs_ideal'])} | "
            f"{pct(r['cagr'])} | {pct(r['mdd'])} | {r['sharpe']:.3f} | "
            f"{pct(r['avg_target_fit'])} | {pct(r['avg_cash_weight'])} | {money(r['cost_paid'])} |"
        )
    lines.extend([
        "",
        "## 판정",
        "",
    ])
    if summary["tracking"]["daily_return_corr"] >= 0.95 and summary["integer"]["avg_target_fit"] >= 0.95:
        lines.append(
            "200만원 시작 + 월 20만원 적립 조건에서도 정수주 실행은 이상적 `yoon2_final_fx`와 "
            "대체로 같은 경로를 재현한다. 다만 최종 성과 차이는 비용과 현금 드래그로 별도 감안해야 한다."
        )
    else:
        lines.append(
            "200만원 시작 + 월 20만원 적립 조건에서는 정수주/최소주문/비용 때문에 이상적 "
            "`yoon2_final_fx`와 동일한 환경이라고 보기 어렵다. 평균 목표 재현율은 95%대지만 "
            "일간 수익률 상관이 낮고, 일부 구간의 재현율 급락과 높은 현금/비용 드래그가 커서 "
            "계좌 규모 확대, 월 적립금 누적 후 주문, 비용 가정 재확인, 또는 보유 종목 수 축소가 필요하다."
        )
    lines.extend([
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
        f"- CSV: `{out_csv.relative_to(ROOT)}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def main() -> None:
    data = build_leaf_targets()
    sim = simulate_integer_dca(data)
    out_json, out_md = write_outputs(data, sim)
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
