"""rolling_profile_expectancy 기대값 추정 '표본 설계' 스윕.

진단: 현행(v1)은 train 756봉을 폭 0.2 버킷 10개에 나눠 버킷당 ~75개 —
min_samples=80과 같은 수준이라 극단 버킷(신호가 있을 곳)은 상시 표본 미달로
NaN→비중 0이고, 발행되는 중간 버킷 기대값은 5일 수익 노이즈에 잠긴다.

여기서 코드 수정 없이 config(expectancy 블록)만 바꿔 표본 설계 변형들을 SPY
실데이터로 비교한다: 버킷 축소(10→5→3), 학습창 확대(3년→10년→확장), 그리고
그에 맞춘 min_samples 상향.

사용:  PYTHONPATH=src python scripts/profile_sizing/rolling_expectancy_design_sweep.py
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
from profile_sizing.rolling_expectancy import run_rolling_expectancy_pipeline  # noqa: E402
from profile_sizing.run import rebased_equity, slice_window  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

SYMBOL = "SPY"
BINS10 = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BINS5 = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
BINS3 = [-1.0, -0.33, 0.33, 1.0]
EXPANDING = 10**8          # train_start가 항상 0 → 확장창

VARIANTS = {
    "현행(10버킷·3년)": {},
    "5버킷·3년": {"score_bins": BINS5},
    "10버킷·10년": {"train_window": 2520},
    "5버킷·10년": {"score_bins": BINS5, "train_window": 2520, "min_samples": 150},
    "5버킷·확장창": {"score_bins": BINS5, "train_window": EXPANDING, "min_samples": 200},
    "3버킷·확장창": {"score_bins": BINS3, "train_window": EXPANDING, "min_samples": 200},
}


def phase_perf(port_ret: pd.Series, bnh_close: pd.Series, index, cfg, phase: str) -> dict:
    win = slice_window(index, phase, cfg)
    eq = rebased_equity(port_ret, win)
    p = performance(eq, eq.pct_change().fillna(0.0), cfg.interval)
    b = bnh_close.reindex(win).ffill()
    beq = b / b.iloc[0]
    bp = performance(beq, beq.pct_change().fillna(0.0), cfg.interval)
    return {"phase": phase, "start": str(win[0].date()), "end": str(win[-1].date()),
            "total_return": p["total_return"], "cagr": p["cagr"], "sharpe": p["sharpe"],
            "mdd": p["max_drawdown"], "bnh_cagr": bp["cagr"], "bnh_sharpe": bp["sharpe"],
            "bnh_mdd": bp["max_drawdown"]}


def main() -> None:
    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "rolling_profile_expectancy_v1.json").read_text("utf-8"))
    raw = load_yfinance(SYMBOL, "1d", "max")
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)

    all_rows, out_json = [], []
    for name, ov in VARIANTS.items():
        cfg_dict = json.loads(json.dumps(base_cfg))
        cfg_dict["expectancy"] = {**cfg_dict["expectancy"], **ov}
        cfg = config_from_dict(cfg_dict)
        out = run_rolling_expectancy_pipeline(raw, cfg)
        f = out["forecast"]
        cov = float(np.isfinite(f["expectancy_bps"]).mean())
        avg_w = float(f["actual_weight"].mean())
        n_tr = int(len(out["trades"]))
        phases = [phase_perf(out["port_ret"], close, raw.index, cfg, ph)
                  for ph in ("all", "validation", "test")]
        out_json.append({"variant": name, "overrides": ov, "coverage": cov,
                         "avg_weight": avg_w, "trades": n_tr, "phases": phases})
        for ph in phases:
            all_rows.append((name, ph, cov, avg_w, n_tr))
        print(f"[{name}] coverage {cov*100:.1f}% · 평균비중 {avg_w*100:.1f}% · 거래 {n_tr}")
        for ph in phases:
            print(f"   {ph['phase']:10s} {ph['start']}~{ph['end']}  "
                  f"수익 {ph['total_return']*100:8.1f}% · CAGR {ph['cagr']*100:6.2f}% · "
                  f"Sharpe {ph['sharpe']:6.3f} · MDD {ph['mdd']*100:7.2f}%   "
                  f"(B&H CAGR {ph['bnh_cagr']*100:.2f}% Sharpe {ph['bnh_sharpe']:.3f})")

    path = ROOT / "reports" / "profile_sizing" / "rolling_expectancy_design_sweep.json"
    path.write_text(json.dumps({"symbol": SYMBOL, "variants": out_json},
                               ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {path}")


if __name__ == "__main__":
    main()
