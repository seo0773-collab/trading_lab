"""yoon2m_defensive(3x3 레짐 필드 점수) 엣지 검증: 부트스트랩 + 워크포워드.

yoon2m에 적용한 동일 검정(yoon2m_validate.py)을 yoon2m_defensive에 그대로 적용한다.
  - 부트스트랩: Sharpe·MDD·초과수익 우위가 운인가 (페어드 블록)
  - 워크포워드: 구간을 갈라도 우위가 반복되는가 (독립 연도/롤링3년)

NAV는 등록된 핸들러 파이프라인(phase=all)에서 직접 생성한다.
벤치 = 50/50 SPY원화+KODEX200 (핸들러 benchmark 그대로).

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2m_defensive_validate.py [--iters N --block L]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import block_indices, _stats, pct  # noqa: E402
from profile_sizing.yoon2a_walkforward import windows, run  # noqa: E402
from trading_lab.strategies.registry import get_handler, get_strategy  # noqa: E402

STRATEGY_ID = "yoon2m_defensive"


def build_nav() -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    defn = get_strategy(STRATEGY_ID)
    config = json.loads(defn.config_path.read_text())
    handler = get_handler(STRATEGY_ID)
    raw = handler.load_data("PORTFOLIO", config)
    art = handler.build_artifacts(
        raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252
    )
    nav = art.equity.dropna()
    bench = art.benchmark.dropna()
    idx = nav.index.intersection(bench.index).sort_values()
    nav = nav.reindex(idx) / nav.reindex(idx).iloc[0]
    bench = bench.reindex(idx) / bench.reindex(idx).iloc[0]
    return nav, bench, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    nav, bench, idx = build_nav()
    p_a = performance(nav, nav.pct_change().fillna(0.0), "1d")
    p_b = performance(bench, bench.pct_change().fillna(0.0), "1d")

    print(f"{STRATEGY_ID}  기간 {idx[0].date()}~{idx[-1].date()} · {len(idx)}봉\n")
    print("=== 점 추정 ===")
    print(f"  {STRATEGY_ID:16s} CAGR {p_a['cagr']*100:6.2f}%  MDD {p_a['max_drawdown']*100:7.2f}%  Sharpe {p_a['sharpe']:.3f}")
    print(f"  {'50/50 벤치':16s} CAGR {p_b['cagr']*100:6.2f}%  MDD {p_b['max_drawdown']*100:7.2f}%  Sharpe {p_b['sharpe']:.3f}")

    ra = nav.pct_change().fillna(0.0).to_numpy()
    rb = bench.pct_change().fillna(0.0).to_numpy()
    n = len(ra)
    rng = np.random.default_rng(args.seed)
    bidx = block_indices(n, args.block, args.iters, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    d_sharpe, d_cagr, d_mdd = sa - sb, ca - cb, ma - mb

    def line(name, arr, scale=1.0, unit=""):
        lo, md, hi = pct(arr * scale)
        print(f"  {name:26s} [{lo:8.3f}, {hi:8.3f}]{unit}  중앙 {md:8.3f}{unit}")

    print(f"\n=== 부트스트랩 95% CI ({args.iters}회, 블록 {args.block}일) ===")
    line("CAGR", ca, 100, "%")
    line("MDD", ma, 100, "%")
    line("Sharpe", sa)
    print("  --- 페어드(벤치 대비) ---")
    line("초과 Sharpe", d_sharpe)
    line("초과 CAGR", d_cagr, 100, "%")
    line("MDD차 (+=방어우위)", d_mdd, 100, "%")
    print("\n=== 가설 확률 ===")
    print(f"  P(Sharpe > 0)        = {(sa > 0).mean()*100:5.1f}%")
    print(f"  P(Sharpe > 벤치)     = {(d_sharpe > 0).mean()*100:5.1f}%")
    print(f"  P(MDD 벤치보다 얕음) = {(d_mdd > 0).mean()*100:5.1f}%")
    print(f"  P(초과CAGR > 0)      = {(d_cagr > 0).mean()*100:5.1f}%")

    yearly, rolling = windows(idx)
    _, yw = run(nav, bench, idx, yearly, "워크포워드: 연도별 (비중복)")
    _, rw = run(nav, bench, idx, rolling, "워크포워드: 롤링 3년 (1년 스텝)")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_defensive_validate.json"
    out.write_text(json.dumps({
        "strategy": STRATEGY_ID, "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": n,
        "point": {"strategy": p_a, "bench": p_b},
        "ci95": {"cagr": pct(ca), "mdd": pct(ma), "sharpe": pct(sa),
                 "excess_sharpe": pct(d_sharpe), "excess_cagr": pct(d_cagr), "mdd_diff": pct(d_mdd)},
        "prob": {"sharpe_pos": float((sa > 0).mean()), "sharpe_beats_bench": float((d_sharpe > 0).mean()),
                 "mdd_shallower": float((d_mdd > 0).mean()), "excess_cagr_pos": float((d_cagr > 0).mean())},
        "walkforward": {"yearly": yw, "rolling": rw},
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
