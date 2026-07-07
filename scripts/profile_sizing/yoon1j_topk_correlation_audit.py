"""yoon1j_kr/yoon1j_lp top3 선택의 실제 상관계수 분포 진단.

현재 `_target_weights`(scripts/profile_sizing/portfolio.py)는 순수 점수 랭킹
(`nlargest`)만으로 top_k를 고르고, 선택된 종목들이 서로 얼마나 같이 움직이는지는
전혀 보지 않는다. 이 스크립트는 실제 프로덕션 파라미터(top_k=3, hold_buffer=2)로
매달 어떤 3종목이 뽑혔는지 재현하고, 그 시점 기준 과거 126일 일간수익률 상관계수로
"선택된 3종목의 평균 쌍별 상관"을 계산해 실제로 문제가 되는 수준인지 확인한다.
비교군으로 "그 시점에 점수>0인 후보 중 무작위 3종목 조합의 평균 상관" 분포도 함께
낸다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon1j_topk_correlation_audit.py
"""
from __future__ import annotations

import itertools
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

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.portfolio import compute_universe, rebalance_dates  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402

CORR_LOOKBACK = 126
TOP_K = 3
HOLD_BUFFER = 2


def select_topk_over_time(scores: pd.DataFrame, top_k: int, hold_buffer: int, rebal: set) -> dict[pd.Timestamp, list[str]]:
    """_target_weights의 hold_buffer 히스테리시스를 그대로 재현한 top_k 선택 이력."""
    sig = scores.shift(1).fillna(0.0)
    held: set[str] = set()
    selections: dict[pd.Timestamp, list[str]] = {}
    for t in scores.index:
        if t not in rebal:
            continue
        row = sig.loc[t]
        avail = row[row > 0]
        if avail.empty:
            selections[t] = []
            continue
        strict = avail.nlargest(min(top_k, len(avail)))
        if held and hold_buffer > 0:
            extended_n = min(top_k + hold_buffer, len(avail))
            extended = avail.nlargest(extended_n)
            keep_held = [s for s in extended.index if s in held]
            selected = set(strict.index) | set(keep_held)
        else:
            selected = set(strict.index)
        selections[t] = sorted(selected)
        held = selected
    return selections


def avg_pairwise_corr(returns: pd.DataFrame, syms: list[str], as_of: pd.Timestamp, lookback: int) -> float | None:
    if len(syms) < 2:
        return None
    window = returns.loc[:as_of].tail(lookback)
    sub = window[syms].dropna(how="any")
    if len(sub) < lookback // 2:
        return None
    corr = sub.corr()
    pairs = list(itertools.combinations(syms, 2))
    vals = [corr.loc[a, b] for a, b in pairs if pd.notna(corr.loc[a, b])]
    return float(np.mean(vals)) if vals else None


def random_baseline_corr(returns: pd.DataFrame, candidates: list[str], as_of: pd.Timestamp, lookback: int, k: int, rng: np.random.Generator, n_samples: int = 30) -> float | None:
    if len(candidates) < k:
        return None
    window = returns.loc[:as_of].tail(lookback)
    sub = window[candidates].dropna(how="any", axis=0)
    if len(sub) < lookback // 2 or sub.shape[1] < k:
        return None
    corr = sub.corr()
    vals = []
    cols = list(sub.columns)
    for _ in range(n_samples):
        pick = rng.choice(cols, size=k, replace=False)
        pairs = list(itertools.combinations(pick, 2))
        pv = [corr.loc[a, b] for a, b in pairs if pd.notna(corr.loc[a, b])]
        if pv:
            vals.append(float(np.mean(pv)))
    return float(np.mean(vals)) if vals else None


def audit_universe(strategy_id: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    from trading_lab.strategies.yoon1k import Yoon1kHandler

    config = Yoon1kHandler._sub_config(strategy_id, overrides)
    cfg = config_from_dict(config)
    handler = ProfilePortfolioHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)
    panels = handler._from_wide(raw)
    scores, prices = compute_universe(panels, cfg)
    returns = prices.pct_change()

    rebal = rebalance_dates(scores.index, str(config.get("rebalance_freq", "monthly")))
    selections = select_topk_over_time(scores, TOP_K, HOLD_BUFFER, rebal)

    rng = np.random.default_rng(7)
    rows = []
    for t, syms in selections.items():
        if len(syms) < 2:
            continue
        avail = scores.shift(1).loc[t]
        candidates = list(avail[avail > 0].index)
        sel_corr = avg_pairwise_corr(returns, syms, t, CORR_LOOKBACK)
        rand_corr = random_baseline_corr(returns, candidates, t, CORR_LOOKBACK, len(syms), rng)
        if sel_corr is None:
            continue
        rows.append({
            "date": str(t.date()), "symbols": syms,
            "selected_avg_corr": sel_corr, "random_avg_corr": rand_corr,
            "n_candidates": len(candidates),
        })

    df = pd.DataFrame(rows)
    high_corr_months = int((df["selected_avg_corr"] > 0.7).sum()) if not df.empty else 0
    return {
        "strategy_id": strategy_id,
        "universe": config.get("universe"),
        "n_rebalances": len(df),
        "selected_avg_corr_mean": float(df["selected_avg_corr"].mean()) if not df.empty else None,
        "selected_avg_corr_median": float(df["selected_avg_corr"].median()) if not df.empty else None,
        "random_avg_corr_mean": float(df["random_avg_corr"].mean()) if not df.empty else None,
        "high_corr_months_gt_0.7": high_corr_months,
        "high_corr_pct": high_corr_months / len(df) if len(df) else None,
        "rows": rows,
    }


def main() -> None:
    composite = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx_credit_gate.json").read_text("utf-8"))
    overrides_by_id = {s["config"]: s.get("overrides") for s in composite["sub_strategies"]}

    results = {}
    for sid in ["yoon1j_kr", "yoon1j_lp"]:
        print(f"Auditing {sid} (override: {overrides_by_id[sid]})...")
        results[sid] = audit_universe(sid, overrides_by_id[sid])
        r = results[sid]
        print(
            f"  {sid}: n_rebalances={r['n_rebalances']}, "
            f"selected_avg_corr(mean/median)={r['selected_avg_corr_mean']:.3f}/{r['selected_avg_corr_median']:.3f}, "
            f"random_avg_corr(mean)={r['random_avg_corr_mean']:.3f}, "
            f"high_corr(>0.7) months={r['high_corr_months_gt_0.7']} ({r['high_corr_pct']*100:.1f}%)"
        )

    out_json = ROOT / "reports" / "profile_sizing" / "yoon1j_topk_correlation_audit.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = ["# yoon1j_kr/yoon1j_lp top3 선택 상관계수 진단", ""]
    for sid, r in results.items():
        lines.extend([
            f"## {sid}",
            "",
            f"- 유니버스: `{r['universe']}`",
            f"- 리밸런스 횟수: `{r['n_rebalances']}`",
            f"- 선택된 top3 평균 쌍별 상관(전체 평균/중앙값): `{r['selected_avg_corr_mean']:.3f}` / `{r['selected_avg_corr_median']:.3f}`",
            f"- 같은 시점 무작위 3종목 조합 평균 상관: `{r['random_avg_corr_mean']:.3f}`",
            f"- 평균 상관 0.7 초과 개월 수: `{r['high_corr_months_gt_0.7']}/{r['n_rebalances']}` (`{r['high_corr_pct']*100:.1f}%`)",
            "",
        ])
    out_md = ROOT / "reports" / "profile_sizing" / "yoon1j_topk_correlation_audit.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
