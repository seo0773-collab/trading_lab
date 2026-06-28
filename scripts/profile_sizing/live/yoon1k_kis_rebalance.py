#!/usr/bin/env python
"""yoon1k(저가변형) KIS 모의투자 리밸런스 오케스트레이터.

시세=실데이터 도메인(실키), 주문=모의 도메인(모의키)을 **분리**해서 쓴다.

통화 배분(기본=통합증거금, --per-sleeve 로 끔):
  A3 재백테스트 결론 — 통화 50/50 을 안 맞추고 통화별 독립 운용하면(드리프트)
  미국 비중이 평균 ~67% 로 흘러가 분산 알파를 잃는다(Sharpe ~0.07 손실). 반면
  월간 환전비용은 무시 가능(50bp 전량환전도 19년 누적 3.25%). 그래서 **총자본 기준
  50/50 복원이 기본**이다. 다만 ±밴드(기본 5%p) 안이면 환전을 건너뛴다(불필요한 환전
  최소화). 원↔달러 실제 환전은 KIS 자동 미지원이라 **필요 환전액을 출력만** 하고
  수동 환전을 전제로 각 슬리브를 50% 목표로 사이징한다(배정현금 = 배정자본 − 보유평가).

흐름:
  1) 두 슬리브 상태 수집(목표=페이퍼 저널, 보유/예수금=모의잔고, 현재가=시세)
  2) 총자본·통화비중 산출 → ±밴드 판정 → (밴드 밖이면) 50/50 배정현금·환전액 계산
  3) 슬리브별 rebalance_plan(주문표) → 기본 출력만. --execute 시에만 모의 제출.

안전:
  - 기본 **dry-run**(제출 안 함). --execute 라야 모의 주문 제출.
  - 실계좌(real)는 이 스크립트에서 막아둠(모의 전용). 실매매는 별도 검증 후.
  - ⚠️ 주문/잔고 TR_ID·필드명은 실서버로 검증된 적 없음 → 소량으로 먼저 확인.

env: 시세 KIS_APP_KEY/SECRET/ACCOUNT(실), 주문 KIS_MOCK_APP_KEY/SECRET/ACCOUNT(모의).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/yoon1k_kis_rebalance.py
  PYTHONPATH=src .venv/bin/python .../yoon1k_kis_rebalance.py --execute      # 모의 제출
  PYTHONPATH=src .venv/bin/python .../yoon1k_kis_rebalance.py --per-sleeve   # 통화독립(드리프트)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from profile_sizing.live.kis_client import KISClient, KISConfigError  # noqa: E402
from profile_sizing.live.rebalance import load_latest_target, rebalance_plan  # noqa: E402


def to_code6(sym: str) -> str:
    return sym.upper().split(".")[0].zfill(6)


def last_close(data: KISClient, sym: str, domestic: bool, exch: str | None) -> float:
    import pandas as pd
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    df = data.daily_history(sym, interval="1d", start=start,
                            exchange=exch or "NASD")
    return float(df["close"].iloc[-1]) if not df.empty else 0.0


def fx_krw_per_usd() -> float:
    from trading_lab.market_data import load_cumulative_yfinance
    return float(load_cumulative_yfinance("KRW=X", "1d", "5d")["close"].iloc[-1])


def plan_currency_allocation(sleeves, target_w, fx, *, total_krw=None, band=0.05):
    """순수 통화 배분 계산(네트워크 없음 — 단위테스트 대상). A3 의 ±밴드 50/50 복원.

    sleeves: {sid: {"currency","account_value","holdings_value"}}.
    target_w: {sid: 목표 통화비중(합 1)}. fx: 원/달러. total_krw: 총자본 강제(None=평가합산).
    band: 통화비중이 목표±band 안이면 환전 생략.

    반환: {total_krw, cur_w, drift, rebalance(bool),
           budgets:{sid: 배정현금(슬리브통화, 밴드 밖일 때만; 배정자본−보유평가, 음수 가능)},
           transfers:{sid: 환전액(원, +유입/−유출)}}.
    배정현금을 rebalance_plan 의 cash 로 주면 account=배정자본 이 되어 50% 목표로 사이징된다."""
    krw_val = {sid: s["account_value"] * (1.0 if s["currency"] == "KRW" else fx)
               for sid, s in sleeves.items()}
    tot = float(total_krw) if total_krw is not None else sum(krw_val.values())
    cur_w = {sid: (v / tot if tot else 0.0) for sid, v in krw_val.items()}
    drift = max((abs(cur_w[sid] - target_w.get(sid, 0.0)) for sid in sleeves), default=0.0)
    budgets: dict[str, float] = {}
    transfers: dict[str, float] = {}
    rebalance = drift > band
    if rebalance:
        for sid, s in sleeves.items():
            alloc_krw = tot * target_w.get(sid, 0.0)
            alloc_local = alloc_krw if s["currency"] == "KRW" else alloc_krw / fx
            budgets[sid] = alloc_local - s["holdings_value"]
            transfers[sid] = alloc_krw - krw_val[sid]
    return {"total_krw": tot, "cur_w": cur_w, "drift": drift,
            "rebalance": rebalance, "budgets": budgets, "transfers": transfers}


def target_age_days(data_through, today=None) -> int:
    """페이퍼 목표의 데이터 기준일(data_through)로부터 경과 달력일수.

    today=None 이면 오늘. 월간 전략이라 며칠은 정상이나, 수 주 묵으면 paper_trade 미갱신
    = 월말 리밸런스를 놓쳤을 위험 → 신선도 가드의 신호로 쓴다."""
    import pandas as pd
    t0 = pd.Timestamp(data_through).normalize()
    t1 = (pd.Timestamp(today).normalize() if today is not None
          else pd.Timestamp.today().normalize())
    return int((t1 - t0).days)


def gather_sleeve(cfg_id, currency, exch, data, order, *, max_stale_days=None):
    """슬리브 현재 상태(목표·보유·현재가·예수금·계좌평가)를 수집한다. 주문 없음.

    반환 dict 또는 실패 시 None. account_value 는 통화 비중·총자본 산출과
    50/50 배정에 쓰인다(통화 리밸런싱은 두 슬리브 평가액을 모두 알아야 한다).
    max_stale_days 지정 시 목표 기준일이 그보다 오래면 None(주문 거부=stale 가드)."""
    journal = ROOT / "var" / "paper_trading" / f"{cfg_id}_journal.jsonl"
    if not journal.exists():
        print(f"[{cfg_id}] 저널 없음 → paper_trade.py --strategy {cfg_id} 먼저 실행\n")
        return None
    snap = load_latest_target(journal)
    if max_stale_days is not None and snap.get("data_through"):
        age = target_age_days(snap["data_through"])
        if age > max_stale_days:
            print(f"[{cfg_id}] 목표 stale: 기준일 {snap['data_through']}"
                  f"({age}일 경과 > {max_stale_days}일) → paper_trade.py 갱신 필요. 건너뜀"
                  f" (무시: --allow-stale)\n")
            return None
    domestic = currency == "KRW"
    targets = {(to_code6(s) if domestic else s.upper()): w
               for s, w in snap["targets"].items()}
    try:
        bal = order.domestic_balance() if domestic else order.balance(exchange=exch)
    except Exception as e:  # noqa: BLE001
        print(f"[{cfg_id}] 잔고 조회 실패: {type(e).__name__}: {str(e)[:140]}\n")
        return None
    holdings = {(to_code6(k) if domestic else k.upper()): v
                for k, v in bal["holdings"].items()}
    prices = {}
    for s in set(targets) | set(holdings):
        try:
            prices[s] = last_close(data, s, domestic, exch)
        except Exception:  # noqa: BLE001
            prices[s] = 0.0
    holdings_value = sum(holdings.get(s, 0.0) * prices.get(s, 0.0) for s in holdings)
    return {
        "cfg_id": cfg_id, "currency": currency, "exch": exch, "order": order,
        "snap": snap, "domestic": domestic, "targets": targets,
        "holdings": holdings, "prices": prices,
        "cash": float(bal["cash"]), "holdings_value": float(holdings_value),
        "account_value": float(bal["cash"]) + float(holdings_value),
    }


def execute_sleeve(st, min_tv, execute, budget_cash=None):
    """수집된 상태로 rebalance_plan → 출력(+ --execute 시 제출).

    budget_cash=배정 현금(통합증거금: 50/50 배정 달성용 환전후 가정 현금). None=자체 예수금.
    배정현금은 음수일 수 있다(이 슬리브가 줄여서 타통화로 보낼 때) → account=배정자본."""
    cfg_id, currency, domestic = st["cfg_id"], st["currency"], st["domestic"]
    order, exch = st["order"], st["exch"]
    cash = float(budget_cash) if budget_cash is not None else st["cash"]
    src = "배정(통합·환전후)" if budget_cash is not None else "예수금"
    plan = rebalance_plan(st["targets"], st["holdings"], st["prices"], cash,
                          min_trade_value=min_tv)
    unit = "원" if domestic else "$"
    print(f"── {cfg_id} [{currency}] {src} {cash:,.2f}{unit} · 보유 {len(st['holdings'])}종 "
          f"· 계좌평가 {plan['account_value']:,.2f}{unit} (목표 {st['snap']['data_through']})")
    if not plan["orders"]:
        print("   조정 주문 없음(이미 목표 근접)\n")
        return 0
    print(f"   {'종목':10} {'side':4} {'수량':>7} {'현재가':>12} {'예상금액':>13}")
    submitted = 0
    for o in plan["orders"]:
        print(f"   {o['symbol']:10} {o['side']:4} {o['qty']:7.0f} "
              f"{o['price']:12,.2f} {o['est_value']:13,.2f}", end="")
        if execute:
            try:
                if domestic:
                    resp = order.domestic_order(o["symbol"], o["side"], o["qty"], price=0.0)
                else:
                    resp = order.order(o["symbol"], o["side"], o["qty"],
                                       exchange=exch, price=float(o["price"]))
                ok = str(resp.get("rt_cd")) == "0"
                print(f"  → {'제출OK' if ok else '거부'}: {str(resp.get('msg1'))[:40]}")
                submitted += int(ok)
            except Exception as e:  # noqa: BLE001
                print(f"  → 오류 {type(e).__name__}: {str(e)[:60]}")
        else:
            print("  (dry-run)")
    print(f"   주문 {len(plan['orders'])}건"
          + (f" · 제출 {submitted}건" if execute else " (미제출)") + "\n")
    return len(plan["orders"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="yoon1k_lp")
    ap.add_argument("--execute", action="store_true", help="모의 주문 실제 제출")
    ap.add_argument("--per-sleeve", action="store_true",
                    help="통화별 독립 예수금 사이징(통화 리밸런싱 안 함=드리프트). "
                         "기본은 통합증거금(총자본 기준 50/50 복원). A3 재백테스트상 "
                         "드리프트는 Sharpe ~0.07 손실이라 비권장.")
    ap.add_argument("--total-krw", type=float, default=None,
                    help="총 투자원금(원) 강제. 미지정 시 두 슬리브 계좌평가 합산을 총자본으로.")
    ap.add_argument("--rebalance-band", type=float, default=0.05,
                    help="통화비중이 목표±이 값 벗어날 때만 50/50 환전(기본 0.05=±5%%p). "
                         "A3: ±5%%p 밴드가 환전 최소화+분산효과 회수 최적.")
    ap.add_argument("--max-stale-days", type=int, default=7,
                    help="페이퍼 목표 기준일이 이보다 오래면 그 슬리브 주문 거부(저널 미갱신 방지, 기본 7일).")
    ap.add_argument("--allow-stale", action="store_true", help="stale 목표 가드 무시.")
    args = ap.parse_args(argv)
    integrated = not args.per_sleeve
    stale_guard = None if args.allow_stale else args.max_stale_days

    try:
        data = KISClient.from_env("KIS", env="real")          # 시세=실키
        order = KISClient.from_env("KIS_MOCK", env="mock")    # 국내 모의(상시대회)
    except KISConfigError as e:
        print(f"[자격증명 오류] {e}")
        print("필요: 시세 KIS_APP_KEY/SECRET/ACCOUNT(실), 주문 KIS_MOCK_APP_KEY/SECRET/ACCOUNT(모의)")
        return 1

    # 해외주식 모의는 별도 '대회 계좌' — 계좌번호만 다르고 키는 동일(별도 키면 OVS_APP_*).
    ovs_acct = os.environ.get("KIS_MOCK_OVS_ACCOUNT")
    order_ovs = None
    if ovs_acct:
        order_ovs = KISClient(
            env="mock",
            app_key=os.environ.get("KIS_MOCK_OVS_APP_KEY") or order.app_key,
            app_secret=os.environ.get("KIS_MOCK_OVS_APP_SECRET") or order.app_secret,
            account=ovs_acct)

    cfg = json.loads((ROOT / "configs" / "strategies" / f"{args.strategy}.json")
                     .read_text("utf-8"))
    mode = "EXECUTE(모의 제출)" if args.execute else "DRY-RUN(미제출)"
    margin = "·통합증거금(50/50 복원)" if integrated else "·통화독립(드리프트)"
    print(f"=== {args.strategy} · KIS 모의 리밸런스 · {mode}{margin} ===")
    print(f"시세 도메인 {data.base} · 주문 도메인 {order.base}\n")

    # 1) 두 슬리브 현재 상태 수집(통화 비중·총자본 산출에 필요)
    states: dict[str, dict] = {}
    for sub in cfg["sub_strategies"]:
        cur = str(sub.get("currency", "KRW"))
        exch = None if cur == "KRW" else "AMEX"
        if cur == "KRW":
            oc = order
        elif order_ovs is not None:
            oc = order_ovs
        else:
            print(f"── {sub['config']} [{cur}] 건너뜀: 해외 모의계좌 미설정 "
                  f"(.env 에 KIS_MOCK_OVS_ACCOUNT 추가 필요)\n")
            continue
        st = gather_sleeve(sub["config"], cur, exch, data, oc,
                           max_stale_days=stale_guard)
        if st is not None:
            states[sub["config"]] = st

    # 2) 통합증거금: 총자본·통화비중 → ±밴드 판정 → 50/50 배정현금 계산
    budgets: dict[str, float] = {}
    if integrated and states:
        try:
            fx = fx_krw_per_usd()
        except Exception as e:  # noqa: BLE001
            print(f"[환율 조회 실패 → 통화 리밸런싱 불가] {type(e).__name__}: {e}")
            return 1
        wsum = sum(float(s.get("weight", 0)) for s in cfg["sub_strategies"]) or 1.0
        tgt_w = {s["config"]: float(s.get("weight", 0)) / wsum for s in cfg["sub_strategies"]}
        alloc = plan_currency_allocation(states, tgt_w, fx, total_krw=args.total_krw,
                                         band=args.rebalance_band)
        budgets = alloc["budgets"]
        cur_w, drift = alloc["cur_w"], alloc["drift"]
        print(f"총자본 {alloc['total_krw']:,.0f}원 · 환율 {fx:,.1f}원/$ · "
              f"통화비중 " + " / ".join(
                  f"{sid} {cur_w[sid]*100:.1f}%(목표 {tgt_w.get(sid,0)*100:.0f}%)"
                  for sid in states))
        if not alloc["rebalance"]:
            print(f"드리프트 {drift*100:.1f}%p ≤ 밴드 {args.rebalance_band*100:.0f}%p "
                  f"→ 환전 없이 각 슬리브 내부만 리밸런스\n")
        else:
            print(f"드리프트 {drift*100:.1f}%p > 밴드 {args.rebalance_band*100:.0f}%p "
                  f"→ 50/50 복원(환전 필요):")
            for sid, st in states.items():
                cur = st["currency"]
                unit = "원" if cur == "KRW" else "$"
                xfer_krw = alloc["transfers"][sid]
                arrow = "유입" if xfer_krw > 0 else "유출"
                print(f"   {sid} [{cur}] 배정현금 {budgets[sid]:,.2f}{unit} · "
                      f"환전 {arrow} {abs(xfer_krw):,.0f}원 상당")
            print("   ⚠️ 원↔달러 실제 환전은 KIS 자동 미지원 → 수동 환전 후 주문 실행\n")

    # 3) 슬리브별 주문표 산출/제출
    total = 0
    for sid, st in states.items():
        min_tv = 50000.0 if st["currency"] == "KRW" else 50.0
        total += execute_sleeve(st, min_tv, args.execute, budget_cash=budgets.get(sid))
    print(f"=== 총 {total}건 ({mode}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
