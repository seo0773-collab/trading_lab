#!/usr/bin/env python
"""yoon1k 모의투자 전진(forward) NAV 트래커 — 경향성 검증용.

모의계좌 평가액을 주기적으로 적립해 '실행된 전략'의 전진 equity 를 만들고,
같은 기간 50/50 패시브 벤치(SPY원화 + KODEX200)와 낙폭·수익을 비교한다.
백테스트 경향(위험조정 우위 + 낙폭 방어)이 모의 실행에서도 유지되는지 본다.

  snapshot : 두 통화 슬리브 모의잔고 평가액(원화환산) + 벤치 종가를 1행 적립.
  report   : 적립된 스냅샷으로 모의 NAV vs 벤치 전진 비교(≥2 스냅샷 필요).

저널: var/live/yoon1k_track.jsonl (READ-ONLY 조회만, 주문 없음).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_track.py snapshot
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_track.py report
"""
from __future__ import annotations

import argparse
import datetime as dt
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

TRACK = ROOT / "var" / "live" / "yoon1k_track.jsonl"
BENCH = {"spy": "SPY", "kospi200": "069500.KS", "fx": "KRW=X"}


def _bench_closes() -> dict:
    from trading_lab.market_data import load_cumulative_yfinance
    out = {}
    for k, sym in BENCH.items():
        try:
            out[k] = float(load_cumulative_yfinance(sym, "1d", "5d")["close"].iloc[-1])
        except Exception:  # noqa: BLE001
            out[k] = None
    return out


def _sleeve_value(order: KISClient, domestic: bool, exch: str) -> tuple[float, int]:
    """모의잔고 평가액(예수금+보유평가, 슬리브 통화)과 보유종목 수."""
    bal = order.domestic_balance() if domestic else order.balance(exchange=exch)
    hv = 0.0
    if not domestic:
        out2 = bal.get("raw", {}).get("output2") or {}
        if isinstance(out2, list):
            out2 = out2[0] if out2 else {}
        hv = float(out2.get("tot_evlu_pfls_amt", 0) or 0)  # 보유평가는 별도 합산 권장
    # 보유평가는 잔고 raw 의 종목별 평가합이 정확하나, 간이로 cash + (해외=평가합 필드).
    return float(bal["cash"]), len(bal["holdings"])


def snapshot() -> int:
    try:
        order = KISClient.from_env("KIS_MOCK", env="mock")
    except KISConfigError as e:
        print(f"[자격증명 오류] {e}")
        return 1
    bench = _bench_closes()
    fx = bench.get("fx") or 0.0

    # 국내(KRW)
    try:
        kr_bal = order.domestic_balance()
        kr_val = float(kr_bal["cash"]) + _holdings_value_krw(order, kr_bal, domestic=True)
        kr_n = len(kr_bal["holdings"])
    except Exception as e:  # noqa: BLE001
        print(f"[국내 잔고 실패] {type(e).__name__}: {str(e)[:100]}")
        kr_val, kr_n = 0.0, 0

    # 해외(USD) — 국내 모의와 같은 KIS_MOCK 계좌를 사용한다.
    us_val_usd, us_n = 0.0, 0
    try:
        us_bal = order.balance(exchange="NASD")
        us_val_usd = float(us_bal["cash"]) + _holdings_value_usd(order, us_bal)
        us_n = len(us_bal["holdings"])
    except Exception as e:  # noqa: BLE001
        print(f"[해외 잔고 실패] {type(e).__name__}: {str(e)[:100]}")

    us_krw = us_val_usd * fx
    total_krw = kr_val + us_krw
    rec = {
        "date": dt.date.today().isoformat(),
        "total_krw": round(total_krw, 2),
        "kr_krw": round(kr_val, 2), "us_usd": round(us_val_usd, 2),
        "us_krw": round(us_krw, 2), "kr_n": kr_n, "us_n": us_n,
        "fx": fx, "spy": bench.get("spy"), "kospi200": bench.get("kospi200"),
    }
    TRACK.parent.mkdir(parents=True, exist_ok=True)
    with TRACK.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[적립] {rec['date']} · 모의총평가 {total_krw:,.0f}원 "
          f"(KR {kr_val:,.0f} {kr_n}종 / US ${us_val_usd:,.0f} {us_n}종) · 환율 {fx:,.1f}")
    return 0


def _holdings_value_krw(order, bal, *, domestic) -> float:
    """국내 보유 평가액(원). 잔고 raw output1 의 종목별 평가금액 합."""
    out1 = bal.get("raw", {}).get("output1") or []
    tot = 0.0
    for row in out1:
        tot += float(row.get("evlu_amt", 0) or 0)
    return tot


def _holdings_value_usd(oc, bal) -> float:
    out1 = bal.get("raw", {}).get("output1") or []
    tot = 0.0
    for row in out1:
        tot += float(row.get("ovrs_stck_evlu_amt", 0) or row.get("evlu_amt", 0) or 0)
    return tot


def report() -> int:
    if not TRACK.exists():
        print("스냅샷 없음 → 먼저 snapshot 적립")
        return 1
    rows = [json.loads(x) for x in TRACK.read_text().splitlines() if x.strip()]
    if len(rows) < 2:
        print(f"스냅샷 {len(rows)}개 — 전진 비교에는 ≥2개 필요(주기 적립 후 재실행)")
        return 0
    import numpy as np
    nav = np.array([r["total_krw"] for r in rows], float)
    spy = np.array([r.get("spy") or np.nan for r in rows], float)
    ks = np.array([r.get("kospi200") or np.nan for r in rows], float)
    fx = np.array([r.get("fx") or np.nan for r in rows], float)
    # 벤치 50/50 (SPY 원화환산 + KODEX200), 첫 스냅샷 기준 리베이스
    spy_krw = spy * fx
    bench = 0.5 * (spy_krw / spy_krw[0]) + 0.5 * (ks / ks[0])
    navn = nav / nav[0]

    def mdd(s):
        peak = np.maximum.accumulate(s)
        return float(np.min(s / peak - 1))
    print(f"전진 스냅샷 {len(rows)}개: {rows[0]['date']} ~ {rows[-1]['date']}")
    print(f"  모의 전략 NAV: {navn[-1]-1:+.2%} · MDD {mdd(navn):.2%}")
    print(f"  벤치 50/50  : {bench[-1]-1:+.2%} · MDD {mdd(bench):.2%}")
    held = "유지 ✓" if mdd(navn) >= mdd(bench) else "역전(점검)"
    print(f"  → 낙폭 방어 경향 {held} (전략 MDD {mdd(navn):.2%} vs 벤치 {mdd(bench):.2%})")
    print("  주: 전진 표본이 짧으면 노이즈. 분기 이상 누적 후 해석.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["snapshot", "report"])
    args = ap.parse_args(argv)
    return snapshot() if args.cmd == "snapshot" else report()


if __name__ == "__main__":
    raise SystemExit(main())
