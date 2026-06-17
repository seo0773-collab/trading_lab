#!/usr/bin/env python
"""yoon1b 실거래 리밸런스 오케스트레이터 (KIS).

흐름: 페이퍼 저널의 현재 목표비중 → (KIS 잔고·현재가) → 주문 리스트 계산 → 출력.
기본은 **dry-run(주문 미제출)**. 제출은 --execute, 실계좌는 추가로 --env real +
--confirm-real 까지 있어야 한다(다단 안전장치).

오프라인(키 없이 로직만): --holdings-file 로 {"cash":.., "holdings":{...}, "prices":{...}}
JSON을 주면 KIS 없이 주문표만 계산한다(수동/토스 체결용).

Usage:
  # 오프라인 주문표(키 불필요):
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_rebalance.py \
      --holdings-file my_account.json
  # KIS 모의투자 잔고로 주문표(키 필요, 제출 안 함):
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_rebalance.py
  # 모의투자 제출:
  ... --execute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.live.rebalance import load_latest_target, rebalance_plan  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="yoon1b")
    ap.add_argument("--env", default=None, help="mock|real (기본 env KIS_ENV 또는 mock)")
    ap.add_argument("--exchange", default="NASD")
    ap.add_argument("--min-trade-value", type=float, default=50.0,
                    help="이보다 작은 금액 변화는 거래 안 함($)")
    ap.add_argument("--holdings-file", type=Path, default=None,
                    help="오프라인 계좌 JSON(cash/holdings/prices) — KIS 미사용")
    ap.add_argument("--execute", action="store_true", help="주문 실제 제출")
    ap.add_argument("--confirm-real", action="store_true", help="실계좌 제출 확인")
    args = ap.parse_args(argv)

    journal = ROOT / "var" / "paper_trading" / f"{args.strategy}_journal.jsonl"
    snap = load_latest_target(journal)
    targets = snap["targets"]
    print(f"[{args.strategy}] 목표 기준일 {snap['data_through']} · "
          f"레짐 {snap.get('market_regime')} · 보유목표 {len(targets)}종")

    client = None
    if args.holdings_file:
        acct = json.loads(args.holdings_file.read_text())
        cash = float(acct.get("cash", 0.0))
        holdings = {k: float(v) for k, v in acct.get("holdings", {}).items()}
        prices = {k: float(v) for k, v in acct.get("prices", {}).items()}
        # 목표 종목 중 가격 누락분은 사용자가 채워야 함.
        miss = [s for s in targets if s not in prices]
        if miss:
            print(f"  ⚠️ 가격 누락(holdings-file에 추가 필요): {', '.join(miss)}")
    else:
        from profile_sizing.live.kis_client import KISClient  # noqa: E402
        client = KISClient(args.env)
        print(f"  KIS {client.env} 도메인 잔고 조회 ...")
        bal = client.balance(exchange=args.exchange)
        cash, holdings = bal["cash"], bal["holdings"]
        syms = sorted(set(targets) | set(holdings))
        prices = {s: client.price(s, args.exchange) for s in syms}

    plan = rebalance_plan(targets, holdings, prices, cash,
                          min_trade_value=args.min_trade_value)
    _print_plan(plan, snap, args)

    if args.execute:
        if client is None:
            print("\n--execute는 KIS 연동에서만 가능(--holdings-file은 계산 전용).")
            return 1
        return _execute(client, plan, args)
    print("\n[dry-run] 주문 미제출. 제출하려면 --execute (모의투자) 추가.")
    return 0


def _print_plan(plan, snap, args) -> None:
    print(f"\n  계좌평가액 ${plan['account_value']:,.2f} (현금 ${plan['cash']:,.2f})")
    if plan["skipped_no_price"]:
        print(f"  ⚠️ 가격없어 제외: {', '.join(plan['skipped_no_price'])}")
    orders = plan["orders"]
    if not orders:
        print("  → 리밸런스 주문 없음(임계치 내).")
        return
    print(f"  주문 {len(orders)}건 (매도 우선):")
    print(f"  {'종목':<8}{'구분':<6}{'수량':>8}{'현재가':>10}{'예상금액':>12}{'목표%':>8}")
    for o in orders:
        print(f"  {o['symbol']:<8}{o['side']:<6}{o['qty']:>8.0f}"
              f"{o['price']:>10.2f}{o['est_value']:>12.2f}{o['target_w']*100:>7.1f}%")


def _execute(client, plan, args) -> int:
    real = client.env == "real"
    if real and not args.confirm_real:
        print("\n⛔ 실계좌(real) 제출은 --confirm-real 필요. 중단.")
        return 1
    print(f"\n{'🔴 실계좌' if real else '🟡 모의투자'} 주문 제출 ...")
    ok = 0
    for o in plan["orders"]:
        try:
            res = client.order(o["symbol"], o["side"], o["qty"],
                               exchange=args.exchange, price=o["price"],
                               confirm_real=args.confirm_real)
            rt = res.get("rt_cd")
            print(f"  {o['symbol']} {o['side']} {o['qty']:.0f}: "
                  f"rt_cd={rt} {res.get('msg1', '')}")
            ok += 1 if rt == "0" else 0
        except Exception as e:  # noqa: BLE001
            print(f"  {o['symbol']} {o['side']} 실패: {e}")
    print(f"  완료: {ok}/{len(plan['orders'])} 성공")
    return 0 if ok == len(plan["orders"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
