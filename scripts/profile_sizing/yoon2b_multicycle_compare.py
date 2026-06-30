"""Phase 0 확장: EWY/SPY 멀티사이클(다주기) 처리로 원/달러 비중 조정 검증.

yoon2b_core_tilt_compare.py(단일 룩백 momentum)가 게이트 실패한 뒤, 신호를
여러 주기로 분해/결합하면 위험조정 우위가 생기는지 확인한다. 빌딩블록은
기존 스크립트에서 재사용한다(코어 슬리브 로딩·동적 결합·phase 슬라이스·성과).

변형:
  T3_mc_trend  : 3M/6M/12M 모멘텀 z-score 평균 → 추세추종 틸트
  T4_mc_revert : 여러 주기 detrend 오실레이터 z-score 평균 → 역추세(과열 축소) 틸트

판정: train(0.6)에서 k 탐색 → Sharpe↑ 그리고 MDD≤base 동시 충족 시에만 채택,
       validation으로 재확인. 멀티테스팅 보정: 통과해도 validation에서 무너지면 기각.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2b_multicycle_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.yoon2b_core_tilt_compare import (  # noqa: E402
    BAND, GRID_K, Z_WINDOW, combine_dynamic, fmt, load_core_sleeves,
    perf_on, slice_phase,
)
from run_kalman_pipeline import load_yfinance  # noqa: E402

CYCLES = [63, 126, 252]  # 3M / 6M / 12M


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).mean()
    sd = s.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def mc_trend_z(ratio: pd.Series) -> pd.Series:
    """다주기 모멘텀 z-score 평균(추세추종)."""
    zs = [_zscore(ratio / ratio.shift(L) - 1.0) for L in CYCLES]
    return pd.concat(zs, axis=1).mean(axis=1)


def mc_revert_z(ratio: pd.Series) -> pd.Series:
    """다주기 detrend 오실레이터 z-score 평균(>0 = 평균 대비 과열)."""
    zs = []
    for P in CYCLES:
        osc = ratio - ratio.rolling(P, min_periods=P // 2).mean()
        zs.append(_zscore(osc))
    return pd.concat(zs, axis=1).mean(axis=1)


def wkr_from(z: pd.Series, k: float, *, sign: float) -> pd.Series:
    """sign=+1 추세추종(z>0 → KR↑), sign=-1 역추세(z>0=과열 → KR↓). 1일 시프트."""
    return (0.5 + sign * k * np.tanh(z)).clip(*BAND).shift(1)


def main() -> None:
    print("코어 슬리브 로딩...")
    us, kr = load_core_sleeves()
    idx = us.index
    print(f"  기간 {idx[0].date()}~{idx[-1].date()}  {len(idx)}봉")
    ewy = load_yfinance("EWY", "1d", "max")["close"].reindex(idx).ffill()
    spy = load_yfinance("SPY", "1d", "max")["close"].reindex(idx).ffill()
    ratio = (ewy / spy)

    phases = ["train", "validation", "test", "all"]
    base_nav = combine_dynamic(us, kr, None)
    base_perf = {ph: perf_on(base_nav, slice_phase(idx, ph)) for ph in phases}
    print("\n=== 베이스(고정 50/50) ===")
    for ph in phases:
        print(f"  {ph:11s} {fmt(base_perf[ph])}")

    variants = [
        ("T3_mc_trend", mc_trend_z(ratio), +1.0),
        ("T4_mc_revert", mc_revert_z(ratio), -1.0),
    ]
    base_tr = base_perf["train"]
    rows = []
    for name, z, sign in variants:
        print(f"\n=== {name}: train 그리드 (Sharpe↑ & MDD≤base) ===")
        best = None
        for k in GRID_K:
            nav = combine_dynamic(us, kr, wkr_from(z, k, sign=sign))
            p = perf_on(nav, slice_phase(idx, "train"))
            if not p or p.get("sharpe") is None:
                continue
            ok = p["sharpe"] > base_tr["sharpe"] and p["max_drawdown"] >= base_tr["max_drawdown"]
            print(f"  k={k:.2f} [{'OK' if ok else '  '}] {fmt(p)}")
            score = p["sharpe"] if ok else -1e9
            if best is None or score > best[0]:
                best = (score, k, ok)
        _, k, ok = best
        nav = combine_dynamic(us, kr, wkr_from(z, k, sign=sign))
        print(f"  >> 선택 k={k} (train 게이트 {'통과' if ok else '미통과'})")
        for ph in phases:
            p = perf_on(nav, slice_phase(idx, ph))
            print(f"     {ph:11s} {fmt(p)}")
            rows.append({"variant": name, "k": k, "phase": ph,
                         "cagr": p.get("cagr"), "mdd": p.get("max_drawdown"),
                         "sharpe": p.get("sharpe"), "total_return": p.get("total_return")})

    for ph in phases:
        bp = base_perf[ph]
        rows.append({"variant": "base_5050", "k": None, "phase": ph,
                     "cagr": bp.get("cagr"), "mdd": bp.get("max_drawdown"),
                     "sharpe": bp.get("sharpe"), "total_return": bp.get("total_return")})
    out = ROOT / "reports" / "profile_sizing" / "yoon2b_multicycle_compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
