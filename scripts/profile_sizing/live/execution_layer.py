"""Execution-layer portfolio shaping under integer-share constraints.

The strategy layer emits target weights. This module converts those weights into
orders and an expected post-trade portfolio without changing strategy signals.
"""
from __future__ import annotations

from math import floor
from typing import Any


def integer_execution_plan(
    targets: dict[str, float],
    holdings: dict[str, float],
    prices: dict[str, float],
    cash: float,
    *,
    min_trade_value: float = 0.0,
    max_iterations: int = 100_000,
) -> dict[str, Any]:
    """Convert target weights into an executable integer-share order plan.

    The optimizer is intentionally broker-neutral and fee-free. It first sells
    integer shares above target, then greedily buys one share at a time only when
    doing so reduces total absolute weight error including cash weight.
    """
    symbols = sorted(set(targets) | set(holdings))
    clean_prices = {s: float(prices.get(s, 0.0) or 0.0) for s in symbols}
    target_w = {s: max(0.0, float(targets.get(s, 0.0) or 0.0)) for s in symbols}
    shares = {s: float(holdings.get(s, 0.0) or 0.0) for s in symbols}
    start_cash = float(cash)
    work_cash = start_cash
    account = start_cash + sum(
        shares[s] * clean_prices[s] for s in symbols if clean_prices[s] > 0
    )

    orders: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_no_price: list[str] = []
    skipped_high_price: list[str] = []

    if account <= 0:
        return {
            "account_value": round(account, 2),
            "cash": round(start_cash, 2),
            "expected_cash": round(work_cash, 2),
            "orders": [],
            "skipped_no_price": [
                s for s in symbols if target_w.get(s, 0.0) > 0 or shares.get(s, 0.0) > 0
            ],
            "skipped_high_price": [],
            "execution_rows": [],
            "tracking_error_abs": 0.0,
            "target_fit": 0.0,
        }

    target_cash_w = max(0.0, 1.0 - sum(target_w.values()))

    def add_order(symbol: str, side: str, qty: float) -> None:
        if qty <= 0:
            return
        price = clean_prices[symbol]
        key = (symbol, side)
        if key not in orders:
            orders[key] = {
                "symbol": symbol,
                "side": side,
                "qty": 0.0,
                "price": round(price, 4),
                "est_value": 0.0,
                "target_w": round(float(target_w.get(symbol, 0.0)), 4),
            }
        orders[key]["qty"] += float(qty)
        orders[key]["est_value"] = round(orders[key]["qty"] * price, 2)

    def total_abs_error(next_shares: dict[str, float], next_cash: float) -> float:
        err = abs((next_cash / account) - target_cash_w)
        for sym in symbols:
            price = clean_prices.get(sym, 0.0)
            expected_w = (next_shares.get(sym, 0.0) * price / account) if price > 0 else 0.0
            err += abs(expected_w - target_w.get(sym, 0.0))
        return err

    for sym in symbols:
        price = clean_prices[sym]
        if price <= 0:
            if target_w.get(sym, 0.0) > 0 or shares.get(sym, 0.0) > 0:
                skipped_no_price.append(sym)
            continue
        current_value = shares[sym] * price
        target_value = target_w[sym] * account
        over_value = current_value - target_value
        if over_value < max(float(min_trade_value), price):
            continue
        qty = min(float(floor(over_value / price)), float(floor(shares[sym])))
        if qty <= 0 or qty * price < min_trade_value:
            continue
        shares[sym] -= qty
        work_cash += qty * price
        add_order(sym, "SELL", qty)

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        current_error = total_abs_error(shares, work_cash)
        best_sym = None
        best_error = current_error
        for sym in symbols:
            price = clean_prices[sym]
            if target_w.get(sym, 0.0) <= 0 or price <= 0 or price > work_cash:
                continue
            trial_shares = dict(shares)
            trial_shares[sym] += 1.0
            trial_error = total_abs_error(trial_shares, work_cash - price)
            if trial_error + 1e-12 < best_error:
                best_error = trial_error
                best_sym = sym
        if best_sym is None:
            break
        price = clean_prices[best_sym]
        shares[best_sym] += 1.0
        work_cash -= price
        add_order(best_sym, "BUY", 1.0)

    for (sym, side), order in list(orders.items()):
        if side == "BUY" and order["est_value"] < min_trade_value:
            qty = float(order["qty"])
            shares[sym] -= qty
            work_cash += qty * clean_prices[sym]
            del orders[(sym, side)]

    for sym in symbols:
        price = clean_prices[sym]
        if (
            target_w.get(sym, 0.0) > 0
            and price > 0
            and price > work_cash
            and shares.get(sym, 0.0) * price < target_w[sym] * account
        ):
            skipped_high_price.append(sym)

    rows = []
    for sym in symbols:
        price = clean_prices[sym]
        current_qty = float(holdings.get(sym, 0.0) or 0.0)
        current_value = current_qty * price if price > 0 else 0.0
        expected_qty = shares.get(sym, 0.0)
        expected_value = expected_qty * price if price > 0 else 0.0
        rows.append({
            "symbol": sym,
            "price": round(price, 4),
            "current_qty": current_qty,
            "expected_qty": expected_qty,
            "target_w": round(target_w.get(sym, 0.0), 6),
            "current_w": round(current_value / account, 6),
            "expected_w": round(expected_value / account, 6),
            "error_w": round((expected_value / account) - target_w.get(sym, 0.0), 6),
            "target_value": round(target_w.get(sym, 0.0) * account, 2),
            "expected_value": round(expected_value, 2),
        })

    cash_row = {
        "symbol": "CASH",
        "price": 1.0,
        "current_qty": start_cash,
        "expected_qty": work_cash,
        "target_w": round(target_cash_w, 6),
        "current_w": round(start_cash / account, 6),
        "expected_w": round(work_cash / account, 6),
        "error_w": round((work_cash / account) - target_cash_w, 6),
        "target_value": round(target_cash_w * account, 2),
        "expected_value": round(work_cash, 2),
    }
    rows.append(cash_row)

    order_list = list(orders.values())
    order_list.sort(key=lambda o: (0 if o["side"] == "SELL" else 1, -o["est_value"]))
    tracking_error = sum(abs(float(row["error_w"])) for row in rows)
    return {
        "account_value": round(account, 2),
        "cash": round(start_cash, 2),
        "expected_cash": round(work_cash, 2),
        "orders": order_list,
        "skipped_no_price": skipped_no_price,
        "skipped_high_price": sorted(set(skipped_high_price)),
        "execution_rows": rows,
        "tracking_error_abs": round(tracking_error, 6),
        "target_fit": round(max(0.0, 1.0 - tracking_error / 2.0), 6),
    }
