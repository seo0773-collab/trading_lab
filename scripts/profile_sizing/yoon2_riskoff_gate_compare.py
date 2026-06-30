"""EWY/SPY 리스크오프 게이트의 한계기여 검증.

지금까지 죽은 건 'KR vs US 제로섬 재배분'. 여기서는 EWY/SPY를 비중 틸트가 아니라
**총노출 디그로싱 게이트**(글로벌 위험선호 프록시)로 쓴다. 핵심 질문은
"고정 50/50을 이기나"가 아니라 **"VIX/200MA 단독 게이트에 EWY/SPY를 더하면 더 나아지나"**.

게이트 파라미터는 yoon4 레짐 블록에서 사전고정(미적합): SPY 200MA, VIX>25, off_scale=0.4.
EWY/SPY는 200MA 하향이탈을 추가 risk-off 조건으로. PIT(1일 시프트). 안전자산=현금(0%).

  G_none : 게이트 없음 (yoon2 50/50)
  G0     : SPY<200MA OR VIX>25  → 노출 0.4
  G1     : G0  OR  (EWY/SPY < 200MA)  → 노출 0.4   (EWY/SPY 추가)

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2_riskoff_gate_compare.py
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
    combine_dynamic, fmt, load_core_sleeves, perf_on, slice_phase,
)
from run_kalman_pipeline import load_yfinance  # noqa: E402

MA = 200
VIX_HIGH = 25.0
OFF_SCALE = 0.4


def gated(core: pd.Series, riskoff: pd.Series) -> pd.Series:
    """riskoff(bool, PIT)면 노출 OFF_SCALE, 아니면 1.0. 안전자산=현금."""
    idx = core.index
    e = pd.Series(np.where(riskoff.reindex(idx).fillna(False), OFF_SCALE, 1.0), index=idx)
    e = e.shift(1).fillna(1.0)
    r = core.pct_change().fillna(0.0)
    return (1.0 + e * r).cumprod()


def main() -> None:
    print("코어 슬리브 로딩...")
    us, kr = load_core_sleeves()
    core = combine_dynamic(us, kr, None)  # yoon2 50/50
    idx = core.index

    spy = load_yfinance("SPY", "1d", "max")["close"].reindex(idx).ffill()
    vix = load_yfinance("^VIX", "1d", "max")["close"].reindex(idx).ffill()
    ewy = load_yfinance("EWY", "1d", "max")["close"].reindex(idx).ffill()
    ratio = ewy / spy

    spy_off = spy < spy.rolling(MA, min_periods=MA // 2).mean()
    vix_off = vix > VIX_HIGH
    ratio_off = ratio < ratio.rolling(MA, min_periods=MA // 2).mean()

    g0_off = spy_off | vix_off
    g1_off = g0_off | ratio_off

    variants = {
        "G_none (50/50)": core,
        "G0 (VIX/200MA)": gated(core, g0_off),
        "G1 (+EWY/SPY)": gated(core, g1_off),
    }
    phases = ["train", "validation", "test", "all"]
    print(f"기간 {idx[0].date()}~{idx[-1].date()} {len(idx)}봉")
    print(f"risk-off 일수: G0 {int(g0_off.sum())}  G1 {int(g1_off.sum())}  "
          f"(EWY/SPY 단독 {int(ratio_off.sum())})\n")
    perfs = {}
    for name, nav in variants.items():
        print(f"=== {name} ===")
        for ph in phases:
            p = perf_on(nav, slice_phase(idx, ph))
            perfs[(name, ph)] = p
            print(f"  {ph:11s} {fmt(p)}")
        print()

    print("--- 한계기여: G1 - G0 (Sharpe) ---")
    for ph in phases:
        d = perfs[("G1 (+EWY/SPY)", ph)]["sharpe"] - perfs[("G0 (VIX/200MA)", ph)]["sharpe"]
        print(f"  {ph:11s} ΔSharpe {d:+.4f}")

    rows = [{"variant": n, "phase": ph, **{k: perfs[(n, ph)].get(k) for k in
            ("cagr", "max_drawdown", "sharpe", "total_return")}}
            for n in variants for ph in phases]
    out = ROOT / "reports" / "profile_sizing" / "yoon2_riskoff_gate_compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
