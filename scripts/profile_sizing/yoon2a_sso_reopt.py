"""yoon2a SSO 비중 재최적화: Strict(MDD≤B&H) 단독 vs Strict+Sharpe(≥B&H).

직전 yoon2a 점검에서 채택값(코어65/SSO35)이 MDD 가드는 통과하나 Sharpe가 B&H보다
낮다는 문제를 확인했다. 여기서는 (1) Sharpe 제약을 추가하고 (2) 1%p 미세격자로
프런티어 경계를 정밀화한다. 전략/registry/config 미변경 임시 계산.

코어 = yoon2(yoon1j_lp/yoon1j_kr 50/50, KRW-base). 오버레이 = SSO(USD→KRW).
B&H = yoon2 핸들러의 50/50 SPY원화+KODEX200 벤치.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2a_sso_reopt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from profile_sizing.yoon2b_core_tilt_compare import slice_phase  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

GRID = np.round(np.arange(0.20, 0.451, 0.01), 2)


def to_base(nav: pd.Series, fx: pd.Series) -> pd.Series:
    f = fx.reindex(nav.index).ffill()
    return nav * (f / f.dropna().iloc[0])


def combine2(core: pd.Series, sso: pd.Series, w: float, freq: str = "monthly") -> pd.Series:
    idx = core.index
    reb = rebalance_dates(idx, freq)
    rc = core.pct_change().fillna(0.0).to_numpy()
    rs = sso.pct_change().fillna(0.0).to_numpy()
    vc, vs = 1.0 - w, w
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        vc *= 1.0 + rc[i]
        vs *= 1.0 + rs[i]
        out[i] = vc + vs
        if t in reb:
            tot = vc + vs
            vc, vs = tot * (1.0 - w), tot * w
    return pd.Series(out, index=idx)


def perf_on(nav: pd.Series, window: pd.DatetimeIndex) -> dict:
    eq = nav.reindex(window).dropna()
    if eq.empty:
        return {}
    eq = eq / eq.iloc[0]
    return performance(eq, eq.pct_change().fillna(0.0), "1d")


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "strategies" / "yoon2.json").read_text("utf-8"))
    h = Yoon1kHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    core_full = art.equity
    bench_full = art.benchmark

    sso_close = load_yfinance("SSO", "1d", "max")["close"].dropna()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]

    idx = core_full.index.intersection(sso_close.index)
    idx = idx.intersection(bench_full.dropna().index).sort_values()
    core = (core_full.reindex(idx).ffill())
    core = core / core.iloc[0]
    sso = sso_close.reindex(idx).ffill()
    sso = to_base(sso / sso.iloc[0], fx)
    bench = bench_full.reindex(idx).ffill()
    bench = bench / bench.iloc[0]

    bperf = {ph: perf_on(bench, slice_phase(idx, ph)) for ph in ("all", "train", "validation")}
    bh = bperf["all"]
    print(f"기간 {idx[0].date()}~{idx[-1].date()} {len(idx)}봉")
    print(f"B&H(50/50)  all: CAGR {bh['cagr']*100:.2f}% MDD {bh['max_drawdown']*100:.2f}% "
          f"Sharpe {bh['sharpe']:.3f}")
    print(f"            train Sharpe {bperf['train']['sharpe']:.3f} / "
          f"val Sharpe {bperf['validation']['sharpe']:.3f}\n")

    print(f"{'SSO':>5} {'코어':>5} | {'CAGR':>7} {'MDD':>8} {'Sharpe':>7} | "
          f"{'MDD≤BH':>6} {'Shp≥BH':>6} | {'tr Shp':>6} {'val Shp':>7}")
    rows = []
    for w in GRID:
        nav = combine2(core, sso, float(w))
        p = perf_on(nav, idx)
        ptr = perf_on(nav, slice_phase(idx, "train"))
        pval = perf_on(nav, slice_phase(idx, "validation"))
        strict = p["max_drawdown"] >= bh["max_drawdown"]
        sharpe_ok = p["sharpe"] >= bh["sharpe"]
        rows.append({"sso_w": float(w), "cagr": p["cagr"], "mdd": p["max_drawdown"],
                     "sharpe": p["sharpe"], "strict": strict, "sharpe_ge_bh": sharpe_ok,
                     "train_sharpe": ptr.get("sharpe"), "val_sharpe": pval.get("sharpe"),
                     "bh_cagr": bh["cagr"], "bh_mdd": bh["max_drawdown"], "bh_sharpe": bh["sharpe"]})
        mark = "*" if abs(w - 0.35) < 1e-9 else " "
        print(f"{w*100:4.0f}% {(1-w)*100:4.0f}%{mark}| {p['cagr']*100:6.2f}% "
              f"{p['max_drawdown']*100:7.2f}% {p['sharpe']:6.3f} | "
              f"{'  ✓' if strict else '  ✗':>6} {'  ✓' if sharpe_ok else '  ✗':>6} | "
              f"{ptr.get('sharpe', float('nan')):6.3f} {pval.get('sharpe', float('nan')):7.3f}")

    df = pd.DataFrame(rows)
    strict_only = df[df.strict]
    both = df[df.strict & df.sharpe_ge_bh]
    print("\n--- 최적점 ---")
    if not strict_only.empty:
        r = strict_only.loc[strict_only.cagr.idxmax()]
        print(f"Strict 단독 최대 CAGR : SSO {r.sso_w*100:.0f}%  CAGR {r.cagr*100:.2f}% "
              f"MDD {r.mdd*100:.2f}% Sharpe {r.sharpe:.3f}")
    if not both.empty:
        r = both.loc[both.cagr.idxmax()]
        print(f"Strict+Sharpe 최대CAGR: SSO {r.sso_w*100:.0f}%  CAGR {r.cagr*100:.2f}% "
              f"MDD {r.mdd*100:.2f}% Sharpe {r.sharpe:.3f}")
    else:
        print("Strict+Sharpe: 충족 조합 없음")
    inc = df[df.sso_w == 0.35].iloc[0]
    print(f"현행(35%)            : CAGR {inc.cagr*100:.2f}% MDD {inc.mdd*100:.2f}% "
          f"Sharpe {inc.sharpe:.3f}")

    out = ROOT / "reports" / "profile_sizing" / "yoon2a_sso_reopt.csv"
    df.to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
