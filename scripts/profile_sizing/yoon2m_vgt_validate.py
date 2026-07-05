"""yoon2m IXN→VGT 유니버스 교체 검증 — 부트스트랩 방어 게이트 재통과 확인.

배경(yoon2m_seed_universe_sweep.md): 추적오차 8.08%p의 주범이 고가 IXN(정수주
라운딩). VGT(18.1만원, corr 0.969)는 전 시드에서 이상적 +0.15~0.27%p 일관 개선.
채택 조건(사전 고정): ① VGT 변형이 yoon2m 방어 게이트 P(MDD<벤치) ≥ 95% 유지,
② 점 추정 CAGR·Sharpe 비열위. 통과 시 yoon2m.json universe override 반영.

사용:  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/yoon2m_vgt_validate.py
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
from profile_sizing.yoon2k_compact_backtest import combine_w  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402

TOP_K = 3


def sleeve_equity_uv(cfg_id: str, top_k: int, universe: list[str] | None) -> pd.Series:
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / f"{cfg_id}.json").read_text("utf-8")))
    cfg["top_k"] = top_k
    if universe is not None:
        cfg["universe"] = universe
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    return art.equity


def build_nav(lp_universe: list[str] | None) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    _, bench, base_idx = build_series()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    lp = sleeve_equity_uv("yoon1j_lp", TOP_K, lp_universe)
    kr = sleeve_equity_uv("yoon1j_kr", TOP_K, None)
    idx = base_idx.intersection(lp.index).intersection(kr.index).sort_values()
    us = to_base((lp.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
    krn = kr.reindex(idx).ffill()
    krn = krn / krn.iloc[0]
    nav = combine_w([us, krn], [0.5, 0.5], idx)
    bench = bench.reindex(idx).ffill()
    return nav, bench / bench.iloc[0], idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base_cfg = json.loads((ROOT / "configs" / "strategies" / "yoon1j_lp.json").read_text("utf-8"))
    uv_ixn = list(base_cfg["universe"])
    uv_vgt = ["VGT" if s == "IXN" else s for s in uv_ixn]

    nav_ixn, bench, idx1 = build_nav(None)
    nav_vgt, bench_v, idx2 = build_nav(uv_vgt)
    idx = idx1.intersection(idx2).sort_values()
    a = nav_vgt.reindex(idx).ffill()
    b = nav_ixn.reindex(idx).ffill()
    bm = bench.reindex(idx).ffill()

    rows = {}
    for name, s in (("yoon2m_VGT", a), ("yoon2m_IXN(현행)", b), ("벤치 50/50", bm)):
        e = s / s.iloc[0]
        rows[name] = performance(e, e.pct_change().fillna(0.0), "1d")
    print(f"기간 {idx[0].date()}~{idx[-1].date()} · {len(idx)}봉")
    for name, p in rows.items():
        print(f"  {name:16s} CAGR {p['cagr']*100:6.2f}%  MDD {p['max_drawdown']*100:7.2f}%  Sharpe {p['sharpe']:.3f}")

    ra = a.pct_change().fillna(0.0).to_numpy()
    rb = b.pct_change().fillna(0.0).to_numpy()
    rm = bm.pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(args.seed)
    bidx = block_indices(len(ra), args.block, args.iters, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    sm, cm, mm = _stats(rm[bidx])

    prob = {
        "P(MDD<벤치)_VGT": float((ma - mm > 0).mean()),
        "P(MDD<벤치)_IXN": float((mb - mm > 0).mean()),
        "P(Sharpe>벤치)_VGT": float((sa - sm > 0).mean()),
        "P(Sharpe_VGT>IXN)": float((sa - sb > 0).mean()),
        "P(CAGR_VGT>IXN)": float((ca - cb > 0).mean()),
    }
    print(f"\n=== 부트스트랩 ({args.iters}회, 블록 {args.block}일) ===")
    for k, v in prob.items():
        print(f"  {k:20s} = {v*100:5.1f}%")
    gate = prob["P(MDD<벤치)_VGT"] >= 0.95
    nonworse = (rows["yoon2m_VGT"]["cagr"] >= rows["yoon2m_IXN(현행)"]["cagr"]
                and rows["yoon2m_VGT"]["sharpe"] >= rows["yoon2m_IXN(현행)"]["sharpe"])
    verdict = "채택" if (gate and nonworse) else "기각"
    print(f"\n판정: {verdict} (방어 게이트 {'통과' if gate else '미달'}, "
          f"점추정 {'비열위' if nonworse else '열위'})")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_vgt_validate.json"
    out.write_text(json.dumps({
        "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": len(idx),
        "universe_lp": uv_vgt, "point": rows, "prob": prob, "verdict": verdict,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
