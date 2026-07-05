"""yoon_pattern 멀티심볼 L 결판: SPY 스윕에서 val↔test 순위가 역전된 L을
4개 지수 ETF(SPY·QQQ·IWM·EFA)로 canonical 파이프라인 그대로 재어 확정한다.

- L ∈ {300, 450, 600} × M=50 고정 (L150=all MDD 실격, M100=coverage 실격 기제외)
- risk_overlay(변동성 타게팅 15%) 포함 = 등록 기본값과 동일 경로
- 판정: 심볼별 validation Sharpe의 중앙값 우선, test 부호 일관성 확인

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon_pattern_multisymbol.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.pattern_projection import run_pattern_expectancy_pipeline  # noqa: E402
from profile_sizing.run import rebased_equity, slice_window  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA"]
LS = [300, 450, 600]


def main() -> None:
    cfgd = json.loads((ROOT / "configs" / "strategies" / "yoon_pattern.json").read_text("utf-8"))
    cfg = config_from_dict(cfgd)
    ro = dict(cfgd.get("risk_overlay") or {})

    rows = []
    for sym in SYMBOLS:
        raw = load_yfinance(sym, "1d", "max").dropna(subset=["close"])
        idx = pd.DatetimeIndex(raw.index)
        for L in LS:
            pat = {**cfgd["pattern"], "length": L}
            out = run_pattern_expectancy_pipeline(raw, cfg, pat, risk_overlay=ro)
            rec = {"symbol": sym, "L": L}
            for ph in ("validation", "test"):
                win = slice_window(idx, ph, cfg)
                eq = rebased_equity(out["port_ret"], win)
                p = performance(eq, eq.pct_change().fillna(0.0), "1d")
                rec[f"{ph}_cagr"] = p["cagr"]
                rec[f"{ph}_sharpe"] = p["sharpe"]
                rec[f"{ph}_mdd"] = p["max_drawdown"]
            rows.append(rec)
            print(f"{sym:4s} L={L:3d}  val {rec['validation_cagr']*100:6.2f}%/"
                  f"{rec['validation_sharpe']:6.3f}/{rec['validation_mdd']*100:6.1f}%  "
                  f"test {rec['test_cagr']*100:6.2f}%/{rec['test_sharpe']:6.3f}/"
                  f"{rec['test_mdd']*100:6.1f}%")

    df = pd.DataFrame(rows)
    print("\nL별 요약 (4심볼):")
    for L in LS:
        d = df[df["L"] == L]
        print(f"  L={L:3d}  val Sharpe 중앙 {d['validation_sharpe'].median():6.3f} "
              f"(min {d['validation_sharpe'].min():6.3f}) · "
              f"test Sharpe 중앙 {d['test_sharpe'].median():6.3f} "
              f"(양수 {int((d['test_sharpe']>0).sum())}/4) · "
              f"test MDD 중앙 {d['test_mdd'].median()*100:6.1f}%")

    out_p = ROOT / "reports" / "profile_sizing" / "yoon_pattern_multisymbol.json"
    out_p.write_text(json.dumps({"symbols": SYMBOLS, "ls": LS, "rows": rows},
                                ensure_ascii=False, indent=2, default=float),
                     encoding="utf-8")
    print(f"\nJSON: {out_p}")


if __name__ == "__main__":
    main()
