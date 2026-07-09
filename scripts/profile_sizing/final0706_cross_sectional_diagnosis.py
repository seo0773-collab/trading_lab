"""final0706 US 유니버스 종목 간 percentile 교차비교 불가능성 진단.

각 ETF의 `final_target_weight`(score)는 그 ETF **자기 자신의 누적분포** 대비
위치에서 나온다(cumulative_percentile → bucket/exp → regime_cap). top_k 선정은
이 점수들을 서로 다른 ETF끼리 직접 비교(`nlargest`)한다. 만약 어떤 ETF가 단지
자기 percentile 변동폭이 넓다는 이유만으로(진짜 더 매력적이어서가 아니라) top_k에
더 자주 뽑힌다면, 이건 "같은 잣대인 척하는 서로 다른 잣대"의 부작용이다.

진단: compute_universe로 11개 ETF의 score 패널을 뽑아 (a) ETF별 score 분포
통계(평균·표준편차·고점수 도달 빈도) (b) 월간 리밸런스 시점마다 top_k=3 선정
빈도 (c) "선정 빈도"와 "score 표준편차/범위"의 상관관계를 본다 — 상관이 높으면
변동폭 큰 ETF가 구조적으로 유리하다는 뜻(교차비교 편향 실재).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_cross_sectional_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.portfolio import compute_universe, rebalance_dates  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
UNIVERSE = ["VGT", "XLF", "IYH", "FENY", "FIDU", "FSTA", "IYC", "XLU", "XLB", "FREL", "FCOM"]


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    us_overrides = next(
        s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD"
    )["overrides"]
    cfg_dict = json.loads((CONFIGS_DIR / "yoon1j_lp.json").read_text("utf-8"))
    cfg_dict.update(us_overrides)
    cfg = config_from_dict(cfg_dict)

    print("데이터 로딩 중...", flush=True)
    panels = {}
    for sym in UNIVERSE:
        panels[sym] = load_yfinance(sym, cfg.interval, cfg.period)

    scores, prices = compute_universe(panels, cfg)
    scores = scores[scores.index >= pd.Timestamp("2007-01-29")]
    prices = prices.reindex(scores.index)

    # (a) ETF별 score 분포 통계 — 상장 이후 구간만(가격 NaN 아닌 곳)
    stats_rows = []
    for sym in UNIVERSE:
        s = scores[sym]
        p = prices[sym]
        valid = s[p.notna()]
        pos = valid[valid > 0]
        stats_rows.append({
            "symbol": sym,
            "listed_bars": int(p.notna().sum()),
            "score_mean": float(valid.mean()),
            "score_std": float(valid.std()),
            "score_p90": float(valid.quantile(0.9)),
            "frac_score_gt_0.7": float((valid > 0.7).mean()),
        })
    stats_df = pd.DataFrame(stats_rows).set_index("symbol")

    # (b) 월간 리밸런스 시점 top_k=3 선정 빈도
    rebal = rebalance_dates(scores.index, "monthly")
    top_k = int(us_overrides.get("top_k", 3))
    sel_count = {sym: 0 for sym in UNIVERSE}
    n_rebal = 0
    for t in rebal:
        row = scores.loc[t]
        avail = row[(row > 0) & prices.loc[t].notna()]
        if avail.empty:
            continue
        n_rebal += 1
        topk = avail.nlargest(min(top_k, len(avail)))
        for sym in topk.index:
            sel_count[sym] += 1
    stats_df["selection_freq"] = pd.Series(
        {sym: sel_count[sym] / n_rebal for sym in UNIVERSE})

    # (c) 상관관계: score_std vs selection_freq (변동폭 큰 종목이 더 자주 뽑히는가)
    corr_std = stats_df["score_std"].corr(stats_df["selection_freq"])
    corr_p90 = stats_df["score_p90"].corr(stats_df["selection_freq"])

    print("\n=== ETF별 score 분포 및 선정 빈도 ===")
    print(stats_df.round(3).to_string())
    print(f"\n상관(score_std vs selection_freq): {corr_std:.3f}")
    print(f"상관(score_p90 vs selection_freq): {corr_p90:.3f}")
    print(f"(월간 리밸런스 {n_rebal}회, top_k={top_k})")

    out = {
        "stats": stats_df.reset_index().to_dict(orient="records"),
        "corr_std_vs_selection_freq": float(corr_std),
        "corr_p90_vs_selection_freq": float(corr_p90),
        "n_rebal": n_rebal,
        "top_k": top_k,
    }
    out_path = ROOT / "reports" / "profile_sizing" / "final0706_cross_sectional_diagnosis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
