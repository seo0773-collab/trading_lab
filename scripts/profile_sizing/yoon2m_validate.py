"""yoon2m(저예산 집중형, top_k=3·SSO없음·50:50) 엣지 검증: 부트스트랩 + 워크포워드.

yoon2a에 적용한 동일 검정을 yoon2m에 그대로 적용한다.
  - 부트스트랩: 본 데이터 내에서 Sharpe·MDD·초과수익 우위가 운인가 (페어드 블록)
  - 워크포워드: 구간을 갈라도(독립 연도/롤링3년) 우위가 반복되는가 (일반화)

yoon2m NAV = top_k=3 yoon1j_lp(USD→KRW) / yoon1j_kr(KRW) 50:50 월간 리밸런스.
벤치 = 50/50 SPY원화+KODEX200 패시브 (build_series 재사용).

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2m_validate.py [--iters N --block L]
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
from profile_sizing.yoon2a_bootstrap import build_series, to_base, block_indices, _stats, pct  # noqa: E402
from profile_sizing.yoon2k_compact_backtest import sleeve_equity, combine_w  # noqa: E402
from profile_sizing.yoon2a_walkforward import perf, windows, run  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

TOP_K = 3


def build_yoon2m() -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    _, bench, base_idx = build_series()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    lp = sleeve_equity("yoon1j_lp", TOP_K)
    kr = sleeve_equity("yoon1j_kr", TOP_K)
    idx = base_idx.intersection(lp.index).intersection(kr.index).sort_values()
    us = to_base((lp.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
    krn = (kr.reindex(idx).ffill()); krn = krn / krn.iloc[0]
    nav = combine_w([us, krn], [0.5, 0.5], idx)
    bench = (bench.reindex(idx).ffill()); bench = bench / bench.iloc[0]
    return nav, bench, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    nav, bench, idx = build_yoon2m()
    p_a = performance(nav / nav.iloc[0], nav.pct_change().fillna(0.0), "1d")
    p_b = performance(bench / bench.iloc[0], bench.pct_change().fillna(0.0), "1d")

    print(f"yoon2m (top_k={TOP_K}, SSO없음, 50:50)  기간 {idx[0].date()}~{idx[-1].date()} · {len(idx)}봉\n")
    print("=== 점 추정 ===")
    print(f"  yoon2m     CAGR {p_a['cagr']*100:6.2f}%  MDD {p_a['max_drawdown']*100:7.2f}%  Sharpe {p_a['sharpe']:.3f}")
    print(f"  50/50 벤치 CAGR {p_b['cagr']*100:6.2f}%  MDD {p_b['max_drawdown']*100:7.2f}%  Sharpe {p_b['sharpe']:.3f}")

    # ---- 부트스트랩 ----
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
    line("yoon2m CAGR", ca, 100, "%")
    line("yoon2m MDD", ma, 100, "%")
    line("yoon2m Sharpe", sa)
    print("  --- 페어드(벤치 대비) ---")
    line("초과 Sharpe", d_sharpe)
    line("초과 CAGR", d_cagr, 100, "%")
    line("MDD차 (+=방어우위)", d_mdd, 100, "%")
    print("\n=== 가설 확률 ===")
    print(f"  P(Sharpe > 0)        = {(sa > 0).mean()*100:5.1f}%")
    print(f"  P(Sharpe > 벤치)     = {(d_sharpe > 0).mean()*100:5.1f}%")
    print(f"  P(MDD 벤치보다 얕음) = {(d_mdd > 0).mean()*100:5.1f}%")
    print(f"  P(초과CAGR > 0)      = {(d_cagr > 0).mean()*100:5.1f}%")

    # ---- 워크포워드 ----
    yearly, rolling = windows(idx)
    _, yw = run(nav, bench, idx, yearly, "워크포워드: 연도별 (비중복)")
    _, rw = run(nav, bench, idx, rolling, "워크포워드: 롤링 3년 (1년 스텝)")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_validate.json"
    out.write_text(json.dumps({
        "strategy": f"yoon2m_topk{TOP_K}", "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": n,
        "point": {"yoon2m": p_a, "bench": p_b},
        "ci95": {"cagr": pct(ca), "mdd": pct(ma), "sharpe": pct(sa),
                 "excess_sharpe": pct(d_sharpe), "excess_cagr": pct(d_cagr), "mdd_diff": pct(d_mdd)},
        "prob": {"sharpe_pos": float((sa > 0).mean()), "sharpe_beats_bench": float((d_sharpe > 0).mean()),
                 "mdd_shallower": float((d_mdd > 0).mean()), "excess_cagr_pos": float((d_cagr > 0).mean())},
        "walkforward": {"yearly": yw, "rolling": rw},
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
