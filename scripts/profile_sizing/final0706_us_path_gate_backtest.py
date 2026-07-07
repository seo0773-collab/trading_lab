"""final0706 US 슬리브: path_gate(경로 틸트) 진단용 백테스트.

`final0706_us_path_study.py` 스크리닝 결과 — FDR 보정 후 **어느 버킷도 채택
기준(FDR 유의 AND WF 3사이클 중 >=2 부호일치)을 통과하지 못했다.** 유일하게
"보류(방향 일치)" 판정을 받은 B[80,100](down-arrival이 up-arrival보다 fwd20/60
높음, t=-2.71/-2.25이나 FDR 미달)조차, 이미 base weight=0.1(사실상 최소 배분
구간)이라 실전 효과가 미미할 가능성이 높다.

이 스크립트는 그럼에도 실제로 얼마나 영향이 있는지 확인하는 **진단용** 백테스트다
(사전 등록된 오버라이드 가중치가 아니므로 정식 채택 게이트가 아니다 — 스윕 없이
딱 한 값만 시험). base sizing.base_target_weight를 몽키패치해 B[80,100]
구간에서 vel<0(down-arrival)일 때만 weight를 0.1→0.3으로 올려본다(스크리닝이
가리킨 방향과 일치하는 유일한 진단 시나리오).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_us_path_gate_backtest.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import profile_sizing.sizing as sizing_mod  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

VEL_WINDOW = 10
BUCKET_LO, BUCKET_HI = 80.0, 100.0
CANDIDATE_WEIGHT = 0.3  # 진단 1개 값(스윕 아님) — base 0.1보다 위, 임의 조정 없음

_orig_base_target_weight = sizing_mod.base_target_weight


def patched_base_target_weight(percentile: np.ndarray, cfg) -> np.ndarray:
    out = _orig_base_target_weight(percentile, cfg)
    pct = np.asarray(percentile, dtype=float)
    pct100 = pct * 100.0
    vel = np.full_like(pct, np.nan)
    vel[VEL_WINDOW:] = pct[VEL_WINDOW:] - pct[:-VEL_WINDOW]
    mask = (pct100 >= BUCKET_LO) & (pct100 <= BUCKET_HI) & (vel < 0)
    out = out.copy()
    out[mask] = CANDIDATE_WEIGHT
    return out


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    s = nav.reindex(window).ffill().dropna() if window is not None else nav.dropna()
    if len(s) < 60:
        return {}
    eq = s / s.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def build_us_only_config(final0706_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    us_leg = copy.deepcopy(next(s for s in cfg["sub_strategies"] if s["currency"] == "USD"))
    us_leg["weight"] = 1.0
    cfg["sub_strategies"] = [us_leg]
    cfg.pop("currency_allocation", None)
    return cfg


def run_variant(us_only_cfg: dict[str, Any]) -> pd.Series:
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", us_only_cfg, synthetic=False)
    art = handler.build_artifacts(raw, us_only_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    return art.equity


def main() -> None:
    final0706_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    us_only_cfg = build_us_only_config(final0706_cfg)

    print("base(원본 US 슬리브) 실행 중...", flush=True)
    nav_base = run_variant(us_only_cfg)

    print(f"path_gate 진단(B[{BUCKET_LO:.0f},{BUCKET_HI:.0f}] down-arrival weight->{CANDIDATE_WEIGHT}) 실행 중...", flush=True)
    sizing_mod.base_target_weight = patched_base_target_weight
    try:
        nav_gate = run_variant(us_only_cfg)
    finally:
        sizing_mod.base_target_weight = _orig_base_target_weight

    idx = nav_base.index
    n = len(idx)
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    v_end = int(n * (train_frac + validation_frac))
    phases = {"all": idx, "test": idx[v_end:],
              "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))]}

    rows = []
    for phase, window in phases.items():
        for label, nav in (("base(원본)", nav_base), (f"path_gate 진단(B80-100 down→{CANDIDATE_WEIGHT})", nav_gate)):
            m = _clean_metric(perf_nav(nav, window))
            rows.append({"phase": phase, "candidate": label, **m})

    idx_c = nav_base.dropna().index.intersection(nav_gate.dropna().index).sort_values()
    ra = nav_base.reindex(idx_c).ffill().pct_change().fillna(0.0).to_numpy()
    rb = nav_gate.reindex(idx_c).ffill().pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(7)
    bidx = block_indices(len(idx_c), 20, 8000, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    boot = {
        "sharpe_gate_gt_base": float((sb > sa).mean()),
        "cagr_gate_gt_base": float((cb > ca).mean()),
        "mdd_gate_shallower": float((mb > ma).mean()),
    }

    print(pd.DataFrame(rows).to_string(index=False))
    print(boot)

    lines = [
        "# final0706 US 슬리브 path_gate 진단 백테스트 (비공식 — 사전등록 아님)",
        "",
        "- `final0706_us_path_study.py` 스크리닝: FDR 보정 후 채택 버킷 없음. B[80,100]만"
        " '보류(방향 일치)'(t=-2.71/-2.25, FDR 미달) — 이미 base weight=0.1인 최소배분 구간이라"
        " 실효성이 낮을 것으로 예상됐다.",
        f"- 이 백테스트는 그 유일한 보류 신호 방향으로 딱 1개 값(B[80,100] 구간에서 down-arrival"
        f" 시 weight 0.1→{CANDIDATE_WEIGHT})만 진단 삼아 넣어본 것 — 스윕도, 사전등록도 아니다.",
        "",
        "| phase | candidate | CAGR | MDD | Sharpe |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['candidate']} | {r['cagr']*100:.2f}% | {r['max_drawdown']*100:.2f}% | {r['sharpe']:.3f} |")
    lines += [
        "",
        "## 페어드 부트스트랩 (8000 iters, block=20, 전체 기간)",
        "",
        f"- P(Sharpe 개선) {boot['sharpe_gate_gt_base']*100:.1f}%, "
        f"P(CAGR 개선) {boot['cagr_gate_gt_base']*100:.1f}%, "
        f"P(MDD 더 얕음) {boot['mdd_gate_shallower']*100:.1f}%",
        "",
        "## 결론",
        "",
        "스크리닝이 사전 등록 채택 기준(FDR 유의 + WF 3사이클 중 2 이상 부호일치)을 통과하지"
        " 못했으므로, 이 백테스트 결과가 어느 방향이든 **final0706에 path_gate를 정식 채택하지"
        " 않는다.** 이 결과는 참고용 진단일 뿐이다.",
    ]
    out_md = ROOT / "reports" / "profile_sizing" / "final0706_us_path_gate_backtest.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD: {out_md}")


if __name__ == "__main__":
    main()
