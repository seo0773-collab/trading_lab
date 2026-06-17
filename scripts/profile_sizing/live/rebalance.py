"""브로커 무관 리밸런스 주문 계산 (순수 함수 — 단위 테스트 대상).

목표 비중(yoon1b의 last_target) + 현재 보유 + 현재가 + 현금 → 구체 주문 리스트.
네트워크·브로커 의존 없음. KIS/IBKR/수동 어디에나 동일하게 쓰인다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_latest_target(journal_path: str | Path) -> dict[str, Any]:
    """페이퍼 저널의 가장 최근 스냅샷(현재 목표 비중 포함)을 읽는다."""
    lines = [ln for ln in Path(journal_path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"저널이 비었습니다: {journal_path}")
    return json.loads(lines[-1])


def rebalance_plan(
    targets: dict[str, float],
    holdings: dict[str, float],
    prices: dict[str, float],
    cash: float,
    *,
    min_trade_value: float = 0.0,
    allow_fractional: bool = False,
) -> dict[str, Any]:
    """목표 비중으로 이동하기 위한 주문 리스트.

    targets: 종목→목표비중(0~1, 합≤1, 나머지 현금). holdings: 종목→보유 주수.
    prices: 종목→현재가. cash: 현금(계좌통화). min_trade_value: 이보다 작은 금액
    변화는 거래 안 함(잔잔한 회전 억제). allow_fractional: 소수주 허용(기본 정수주).

    반환: {"account_value", "orders":[{symbol,side,qty,price,est_value,target_w}], ...}
    주문은 매도 먼저(현금 확보) → 매수, 금액 큰 순.
    """
    account = float(cash) + sum(
        float(holdings.get(s, 0.0)) * float(prices[s])
        for s in holdings if s in prices and prices[s]
    )
    orders: list[dict[str, Any]] = []
    skipped: list[str] = []
    for s in sorted(set(targets) | set(holdings)):
        price = float(prices.get(s, 0.0) or 0.0)
        cur_sh = float(holdings.get(s, 0.0))
        if price <= 0:
            if cur_sh > 0 or targets.get(s, 0.0) > 0:
                skipped.append(s)  # 가격 없으면 거래 불가
            continue
        tgt_val = float(targets.get(s, 0.0)) * account
        delta_val = tgt_val - cur_sh * price
        if abs(delta_val) < min_trade_value:
            continue
        raw_sh = abs(delta_val) / price
        qty = round(raw_sh, 4) if allow_fractional else float(int(raw_sh))
        if qty <= 0:
            continue
        side = "BUY" if delta_val > 0 else "SELL"
        if side == "SELL":
            qty = min(qty, cur_sh if allow_fractional else float(int(cur_sh)))
            if qty <= 0:
                continue
        orders.append({
            "symbol": s, "side": side, "qty": qty, "price": round(price, 4),
            "est_value": round(qty * price, 2),
            "target_w": round(float(targets.get(s, 0.0)), 4),
        })
    orders.sort(key=lambda o: (0 if o["side"] == "SELL" else 1, -o["est_value"]))
    return {
        "account_value": round(account, 2),
        "cash": round(float(cash), 2),
        "orders": orders,
        "skipped_no_price": skipped,
    }
