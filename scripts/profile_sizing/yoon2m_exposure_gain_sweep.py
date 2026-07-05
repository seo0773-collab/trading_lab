"""yoon2m exposure_gain 스윕 — 만성 저노출(수익 양보)의 다이얼 실험.

감사 실측: yoon2m 슬리브의 주식노출 평균 70~75%(exposure=mean top-K 점수가 버킷
중간값으로 뭉개짐) = 벤치 대비 수익 양보의 원천. simulate_portfolio에 이미 있는
exposure_gain(노출에 게인 곱, 1.0 클립=무레버)을 1.0→1.5 스윕해 "방어 정체성
(P(MDD<벤치)≥95%, 롤링WF MDD 방어)을 유지하면서 수익 갭을 얼마나 줄이는지" 본다.

구성: yoon2m과 동일(yoon1j_lp/kr top_k=3, 50:50 월간, SSO 없음). 부트스트랩은
yoon2m_validate와 동일 설정(5000회·블록20·seed7)으로 비교 가능.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2m_exposure_gain_sweep.py
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

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import (  # noqa: E402
    build_series, to_base, block_indices, _stats,
)
from profile_sizing.yoon2k_compact_backtest import combine_w  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

TOPK = 3
GAINS = [1.0, 1.2, 1.3, 1.4, 1.5]
ITERS, BLOCK, SEED = 5000, 20, 7


def sleeve_eq(cfg_id: str, gain: float) -> pd.Series:
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / f"{cfg_id}.json").read_text("utf-8")))
    cfg["top_k"] = TOPK
    cfg["exposure_gain"] = gain
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    return h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252).equity


def metr(eq: pd.Series) -> dict:
    eq = (eq / eq.iloc[0]).dropna()
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {"cagr": p["cagr"], "mdd": p["max_drawdown"], "sharpe": p["sharpe"],
            "tot": eq.iloc[-1] - 1.0}


def rolling_wf_mdd(nav: pd.Series, bench: pd.Series, idx: pd.DatetimeIndex) -> tuple[int, int]:
    """롤링 3년(1년 스텝) 창에서 MDD가 벤치보다 얕은 창 수 / 전체 창 수."""
    years = sorted(set(idx.year))
    wins = tot = 0
    for y0 in years:
        lo, hi = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y0 + 3}-01-01")
        win = idx[(idx >= lo) & (idx < hi)]
        if len(win) < 500 or win[-1] < hi - pd.Timedelta(days=40):
            continue
        def mdd(s):
            e = s.reindex(win).ffill().dropna()
            e = e / e.iloc[0]
            return float((e / e.cummax() - 1.0).min())
        tot += 1
        if mdd(nav) > mdd(bench):
            wins += 1
    return wins, tot


def main() -> None:
    _, bench, base_idx = build_series()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    rng = np.random.default_rng(SEED)

    rows = []
    for g in GAINS:
        lp, kr = sleeve_eq("yoon1j_lp", g), sleeve_eq("yoon1j_kr", g)
        idx = base_idx.intersection(lp.index).intersection(kr.index).sort_values()
        us = to_base((lp.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
        krn = kr.reindex(idx).ffill()
        krn = krn / krn.iloc[0]
        nav = combine_w([us, krn], [0.5, 0.5], idx)
        b = bench.reindex(idx).ffill()
        b = b / b.iloc[0]

        m, mb = metr(nav), metr(b)
        ra = nav.pct_change().fillna(0.0).to_numpy()
        rb = b.pct_change().fillna(0.0).to_numpy()
        bidx = block_indices(len(ra), BLOCK, ITERS, rng)
        sa, ca, ma = _stats(ra[bidx])
        sb, cb, mb_ = _stats(rb[bidx])
        p_mdd = float(((ma - mb_) > 0).mean())
        p_sharpe = float(((sa - sb) > 0).mean())
        exc_cagr_med = float(np.median(ca - cb))
        wf_w, wf_n = rolling_wf_mdd(nav, b, idx)
        rows.append({"gain": g, **{f"m_{k}": v for k, v in m.items()},
                     "p_mdd_shallower": p_mdd, "p_sharpe_beats": p_sharpe,
                     "excess_cagr_median": exc_cagr_med,
                     "wf_rolling_mdd": f"{wf_w}/{wf_n}"})
        print(f"gain {g:.1f}: CAGR {m['cagr']*100:6.2f}% MDD {m['mdd']*100:7.2f}% "
              f"Sharpe {m['sharpe']:.3f} 총 {m['tot']*100:6.0f}% · "
              f"P(MDD<벤치) {p_mdd*100:5.1f}% · P(Sharpe>벤치) {p_sharpe*100:5.1f}% · "
              f"초과CAGR중앙 {exc_cagr_med*100:+.2f}%p · 롤링WF {wf_w}/{wf_n}")
        if g == GAINS[0]:
            print(f"   (벤치: CAGR {mb['cagr']*100:.2f}% MDD {mb['mdd']*100:.2f}% "
                  f"Sharpe {mb['sharpe']:.3f})")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_exposure_gain_sweep.json"
    out.write_text(json.dumps({"topk": TOPK, "iters": ITERS, "block": BLOCK, "seed": SEED,
                               "rows": rows}, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
