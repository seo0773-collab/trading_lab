#!/usr/bin/env python
"""S&P 500 point-in-time(PIT) 멤버십 복원 (생존편향 완화용).

위키피디아 'List of S&P 500 companies'의 현재 구성 + 변경이력(Added/Removed by date)을
역재생해 과거 각 시점의 구성원 집합을 추정한다.

한계(정직):
- 위키 변경이력이 불완전(특히 2017 이전) → 멤버십은 근사(과거일수록 과대추정 경향).
- 편출/상장폐지 종목은 yfinance에 가격이 대부분 없어 실제 백테스트에선 빠진다 →
  '편입일 마스킹'으로 진입 타이밍 편향은 줄여도 상폐 생존편향은 데이터상 남는다.
- 따라서 PIT 결과의 '개선'은 보수적 하한(실제 편향은 더 클 수 있음)으로 해석한다.

캐시: var/pit/ (gitignore). 네트워크 필요(로컬에서 실행 — 클라우드는 위키/야후 차단).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/pit_universe.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "var" / "pit"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HDR = {"User-Agent": "Mozilla/5.0 (research; trading_lab)"}


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-").replace("$", "")


def fetch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    import requests
    html = requests.get(WIKI, headers=HDR, timeout=30).text
    tabs = pd.read_html(io.StringIO(html))
    comp, chg = tabs[0], tabs[1]
    chg.columns = ["date", "add_t", "add_s", "rem_t", "rem_s", "reason"]
    chg = chg[chg["date"] != "Effective Date"].copy()
    chg["date"] = pd.to_datetime(chg["date"], errors="coerce")
    chg = chg.dropna(subset=["date"]).sort_values("date")
    return comp, chg


def build_membership(comp: pd.DataFrame, chg: pd.DataFrame,
                     dates: list[pd.Timestamp]) -> dict[str, list[str]]:
    """각 date 시점의 추정 구성원 집합. 현재 구성에서 이후 변경을 역재생."""
    current = {_norm(s) for s in comp["Symbol"]}
    out = {}
    for d in dates:
        s = set(current)
        future = chg[chg["date"] > d]
        for _, r in future.iterrows():
            if str(r["add_t"]) != "nan":
                s.discard(_norm(r["add_t"]))   # d 이후 편입 → 당시 비회원
            if str(r["rem_t"]) != "nan":
                s.add(_norm(r["rem_t"]))        # d 이후 편출 → 당시 회원
        out[d.strftime("%Y-%m-%d")] = sorted(s)
    return out


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    comp, chg = fetch_tables()
    current = sorted({_norm(s) for s in comp["Symbol"]})
    # 월말 그리드(2000~현재). 백테스트에서 리밸런스 시점 멤버십 조회에 쓴다.
    end = pd.Timestamp.today().normalize()
    dates = list(pd.date_range("2000-01-31", end, freq="ME"))
    membership = build_membership(comp, chg, dates)

    (CACHE / "sp500_current.json").write_text(json.dumps(current, indent=0))
    (CACHE / "sp500_membership.json").write_text(json.dumps(membership))
    print(f"변경이력: {chg['date'].min().date()} ~ {chg['date'].max().date()} ({len(chg)}건)")
    print(f"현재 구성: {len(current)}종 → var/pit/sp500_current.json")
    print(f"멤버십 그리드: {len(membership)}개 월말 → var/pit/sp500_membership.json")
    print(f"샘플 구성수: 2010={len(membership.get('2010-06-30', []))} "
          f"2020={len(membership.get('2020-06-30', []))} "
          f"현재={len(membership[dates[-1].strftime('%Y-%m-%d')])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
