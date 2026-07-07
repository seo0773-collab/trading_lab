#!/usr/bin/env python
"""KIS 해외주식 분봉조회(HHDFS76950200) 과거 조회 깊이 실측 프로브 (READ-ONLY, 주문 없음).

yfinance는 5m/15m/30m을 최근 60일로 제한한다. KIS Developers 공개 문서(GitHub
open-trading-api 샘플)에 따르면:
  - 국내(domestic) 분봉조회(inquire-time-itemchartprice, 국내)는 "당일 분봉만
    제공(전일자 미제공)" — 과거 재검증에 아예 못 쓴다.
  - 해외(overseas) 분봉조회(HHDFS76950200)는 pinc=1(전일포함) + keyb + next로
    페이지네이션 가능하다고 문서화돼 있으나, **실제 최대 과거 깊이는 문서에
    명시돼 있지 않다** — 그래서 이 스크립트로 직접 측정한다.

한 심볼(기본 AAPL/NAS)에 대해 최신 시점부터 keyb로 계속 과거로 페이지네이션하며
가장 오래된 타임스탬프가 며칠 전까지 가는지, 언제 데이터가 끊기는지 기록한다.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_overseas_minute_probe.py \
      --symbol AAPL --excd NAS --nmin 5 --max-pages 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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

from profile_sizing.live.kis_client import KISClient  # noqa: E402

TR_ID = "HHDFS76950200"
URL_PATH = "/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"


def fetch_page(client: KISClient, *, excd: str, symb: str, nmin: str,
              pinc: str, next_: str, keyb: str) -> dict:
    import requests
    r = requests.get(
        f"{client.base}{URL_PATH}",
        headers=client._headers(TR_ID),
        params={"AUTH": "", "EXCD": excd, "SYMB": symb, "NMIN": nmin,
               "PINC": pinc, "NEXT": next_, "NREC": "120", "FILL": "", "KEYB": keyb},
        timeout=client.timeout,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--excd", default="NAS")
    ap.add_argument("--nmin", default="5")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--env", default="real", choices=["real", "mock"])
    args = ap.parse_args()

    client = KISClient(env=args.env)
    print(f"env={args.env} base={client.base} symbol={args.symbol} excd={args.excd} nmin={args.nmin}")

    keyb = ""
    next_ = ""
    pinc = "1"
    seen_days = set()
    oldest_ts = None
    total_records = 0
    for page in range(args.max_pages):
        try:
            payload = fetch_page(client, excd=args.excd, symb=args.symbol, nmin=args.nmin,
                                 pinc=pinc, next_=next_, keyb=keyb)
        except Exception as e:  # noqa: BLE001
            print(f"page {page}: 에러 — {type(e).__name__}: {e}")
            break
        out1 = payload.get("output1") or {}
        out2 = payload.get("output2") or []
        rt_cd = payload.get("rt_cd")
        msg = payload.get("msg1")
        if not out2:
            print(f"page {page}: 데이터 없음 (rt_cd={rt_cd}, msg={msg!r}) — 여기서 멈춤")
            break
        # output2는 최신→과거 내림차순. 현지시각(xymd/xhms) 기준으로 페이지네이션.
        oldest_row = out2[-1]
        first_dt = f"{out2[0].get('xymd','')}{out2[0].get('xhms','')}"
        last_dt = f"{oldest_row.get('xymd','')}{oldest_row.get('xhms','')}"
        days = {row.get("xymd") for row in out2}
        seen_days |= {d for d in days if d}
        total_records += len(out2)
        oldest_ts = min(oldest_ts, last_dt) if oldest_ts else last_dt
        print(f"page {page}: {len(out2)}건, xymd/xhms범위 {first_dt}~{last_dt}, rt_cd={rt_cd}, output1.more={out1.get('more')}")
        nxt_keyb = f"{oldest_row.get('xymd','')}{oldest_row.get('xhms','')}"
        if not nxt_keyb or nxt_keyb == keyb:
            print(f"page {page}: keyb 갱신 안 됨 — 페이지네이션 종료로 판단")
            break
        keyb = nxt_keyb
        next_ = "1"
        time.sleep(0.3)

    print("\n=== 요약 ===")
    print(f"총 레코드: {total_records}, 확인된 거래일수: {len(seen_days)}, 가장 오래된 타임스탬프: {oldest_ts}")
    if seen_days:
        print(f"날짜 범위: {min(seen_days)} ~ {max(seen_days)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
