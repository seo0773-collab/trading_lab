#!/usr/bin/env python
"""KIS 잔고 응답 필드 실측 프로브 (READ-ONLY, 주문 절대 없음).

A2 후속: kis_client 의 잔고 파싱은 추정 필드(dnca_tot_amt/frcr_dncl_amt1 등)를 쓴다.
이 스크립트는 모의(+선택적으로 실) 계좌의 잔고를 **조회만** 해서 output1/output2 의
실제 키와 예수금 후보 값을 덤프한다 → 파싱 필드를 확정하는 근거.

기본은 모의 국내(KRW)만. --overseas 면 해외도(키/계좌 있을 때). --real 은 실계좌 조회
(읽기 전용이지만 실서버 접속이라 명시적 동의로만).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_field_probe.py
"""
from __future__ import annotations

import argparse
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

CASH_CANDIDATES = ("dnca_tot_amt", "prvs_rcdl_excc_amt", "nxdy_excc_amt",
                   "frcr_dncl_amt1", "frcr_evlu_amt2", "tot_evlu_amt",
                   "cma_evlu_amt", "frcr_use_psbl_amt")


def dump(label: str, raw: dict) -> None:
    print(f"\n=== {label} ===")
    out1 = raw.get("output1")
    out2 = raw.get("output2")
    if isinstance(out2, list):
        out2 = out2[0] if out2 else {}
    out2 = out2 or {}
    print(f"output1: {len(out1) if isinstance(out1,list) else type(out1).__name__} rows")
    if isinstance(out1, list) and out1:
        print(f"  output1[0] keys: {sorted(out1[0])}")
    print(f"output2 keys ({len(out2)}): {sorted(out2)}")
    hits = {k: out2[k] for k in CASH_CANDIDATES if k in out2}
    if not hits and str(raw.get("rt_cd")) == "0" and str(raw.get("msg_cd")) == "70070000":
        print("예수금 후보 필드 값: (조회 내역 없음 → 해외 현금 0으로 처리)")
    else:
        print(f"예수금 후보 필드 값: {hits if hits else '(후보 키 없음 → 파싱 필드 수정 필요)'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overseas", action="store_true", help="해외 잔고도 조회")
    ap.add_argument("--real", action="store_true", help="실계좌 조회(읽기 전용, 실서버 접속)")
    args = ap.parse_args(argv)
    print("KIS 잔고 필드 프로브 (READ-ONLY · 주문 없음)")
    try:
        if args.real:
            cli = KISClient.from_env("KIS", env="real")
            print(f"실계좌 도메인 {cli.base}")
            dump("실 국내잔고", cli.domestic_balance()["raw"])
            if args.overseas:
                dump("실 해외잔고(NASD)", cli.balance(exchange="NASD", currency="USD")["raw"])
        else:
            cli = KISClient.from_env("KIS_MOCK", env="mock")
            print(f"모의계좌 도메인 {cli.base}")
            dump("모의 국내잔고", cli.domestic_balance()["raw"])
            if args.overseas:
                dump("모의 해외잔고(NASD)", cli.balance(exchange="NASD", currency="USD")["raw"])
    except KISConfigError as e:
        print(f"[자격증명 오류] {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[조회 실패] {type(e).__name__}: {str(e)[:200]}")
        return 2
    print("\n→ 위 output2 키와 예수금 후보로 kis_client 의 _pick_amount 필드를 확정하라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
