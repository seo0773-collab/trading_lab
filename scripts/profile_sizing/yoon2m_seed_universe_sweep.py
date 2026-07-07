"""yoon2m 실집행 개선 실측: 시드 스윕(200~400만) × IXN 대체 저가 ETF.

yoon2m 정수주 실집행의 CAGR drag(-2.95%p, 200만 기준) 주범은 (1) 소액 정수주
라운딩 — 특히 유니버스 최고가 IXN(≈21.6만원/주), (2) 해외 마찰이다. 여기서 두
지렛대를 실측한다.

  - 시드 스윕: 200/250/300/350/400만원 — 라운딩 오차가 시드에 어떻게 수렴하는가
  - IXN 대체: IXN→VGT(≈18.1만원, corr 0.969) / IXN→ONEQ(≈15.8만원, corr 0.935)
    후보는 가격·상관·히스토리(2007 커버) 스크리닝 통과분. FTEC/IYW/XLK는 IXN보다
    비싸고 IETC는 2018 상장이라 제외.

이상적 NAV는 yoon2m_validate.py와 동일한 슬리브 50:50 결합(등록 수치 재현 검증됨),
정수주 시뮬은 yoon2m_live_verify.py의 floor+그리디 로직을 따른다.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2m_seed_universe_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from profile_sizing.yoon2a_bootstrap import build_series, to_base  # noqa: E402
from profile_sizing.yoon2k_compact_backtest import combine_w  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

TOPK = 3
FREQ = "monthly"
COST_KR = 0.000646527  # 0.0647%/side = KIS 국내 수수료 0.0147% + 슬리피지 0.05% (계좌 실조건 2026-07-05)
COST_US = 0.0025103    # 0.2510%/side = 수수료 0.1%우대+제비용 + 환전 0.1%(90%우대) + 슬리피지 0.05%
SEEDS = [2_000_000, 2_500_000, 3_000_000, 3_500_000, 4_000_000]
VARIANTS = {          # lp 슬리브 유니버스에서 IXN 자리를 바꾼다
    "현행(IXN)": None,
    "IXN→VGT": ("IXN", "VGT"),
    "IXN→ONEQ": ("IXN", "ONEQ"),
}


def sleeve(cfg_id: str, swap: tuple[str, str] | None) -> tuple[pd.Series, pd.DataFrame]:
    """슬리브 equity(연속비중 이상적)와 portfolio_wave(종목별 목표비중)를 함께 얻는다."""
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / f"{cfg_id}.json").read_text("utf-8")))
    cfg["top_k"] = TOPK
    if swap:
        old, new = swap
        cfg["universe"] = [new if s == old else s for s in cfg["universe"]]
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    w = art.extras["portfolio_wave"]
    if "time" in w.columns:
        w = w.set_index("time")
    w.index = pd.to_datetime(w.index)
    w = w.drop(columns=[c for c in w.columns if c == "cash"], errors="ignore")
    return art.equity, w


def real_symbol(col: str) -> str:
    return col[:-5] if col.endswith("_SAFE") else col


def metr(eq: pd.Series) -> dict:
    eq = (eq / eq.iloc[0]).dropna()
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {"cagr": p["cagr"], "mdd": p["max_drawdown"], "sharpe": p["sharpe"],
            "tot": eq.iloc[-1] - 1.0}


def integer_sim(seed: float, idx, wreal: pd.DataFrame, pxk: pd.DataFrame,
                cost_of: dict[str, float]) -> tuple[pd.Series, dict]:
    """월간 정수주 리밸런스(floor + 예산제약 그리디). NAV와 집행 품질 지표를 돌려준다."""
    reb = set(rebalance_dates(idx, FREQ))
    syms = list(wreal.columns)
    shares = {s: 0.0 for s in syms}
    cash = seed
    nav = np.empty(len(idx))
    werrs, invs = [], []
    for i, t in enumerate(idx):
        price = pxk.loc[t]
        val = cash + sum(shares[s] * (price[s] if not math.isnan(price[s]) else 0.0) for s in syms)
        if t in reb:
            tgt = wreal.loc[t]
            avail = {s: price[s] for s in syms if not math.isnan(price[s]) and price[s] > 0}
            desired = {s: math.floor(val * tgt[s] / avail[s]) if s in avail else 0 for s in syms}
            new_sh = dict(desired)

            def spent_cash(sh):
                buy = sum(max(sh[s] - shares[s], 0) * avail[s] * (1 + cost_of[s]) for s in avail)
                sell = sum(max(shares[s] - sh[s], 0) * avail[s] * (1 - cost_of[s]) for s in avail)
                return buy - sell

            while spent_cash(new_sh) > cash + 1e-6:
                over = max(avail, key=lambda s: (new_sh[s] * avail[s] / val - tgt[s]) if new_sh[s] > 0 else -9)
                if new_sh[over] <= 0:
                    break
                new_sh[over] -= 1
            while True:
                cand = [s for s in avail if spent_cash({**new_sh, s: new_sh[s] + 1}) <= cash + 1e-6]
                if not cand:
                    break
                u = max(cand, key=lambda s: tgt[s] - new_sh[s] * avail[s] / val)
                if tgt[u] - new_sh[u] * avail[u] / val <= 0:
                    break
                new_sh[u] += 1
            cash -= spent_cash(new_sh)
            shares = {s: float(new_sh.get(s, 0)) for s in syms}
            val = cash + sum(shares[s] * (price[s] if not math.isnan(price[s]) else 0.0) for s in syms)
            werrs.append(sum(abs(shares[s] * (price[s] if s in avail else 0.0) / val - tgt[s]) for s in syms))
            invs.append(1.0 - cash / val)
        nav[i] = val
    q = {"werr_mean": float(np.mean(werrs)), "werr_max": float(np.max(werrs)),
         "invested_mean": float(np.mean(invs))}
    return pd.Series(nav, index=idx), q


def main() -> None:
    _, bench, base_idx = build_series()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    kr_eq, kr_w = sleeve("yoon1j_kr", None)

    results = []
    for vname, swap in VARIANTS.items():
        lp_eq, lp_w = sleeve("yoon1j_lp", swap)

        # ---- 이상적(연속비중) 50:50 결합 ----
        idx = base_idx.intersection(lp_eq.index).intersection(kr_eq.index).sort_values()
        us = to_base((lp_eq.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
        krn = kr_eq.reindex(idx).ffill()
        krn = krn / krn.iloc[0]
        ideal = combine_w([us, krn], [0.5, 0.5], idx)

        # ---- 종목별 목표비중 + 원화 가격 ----
        widx = lp_w.index.intersection(kr_w.index).intersection(idx).sort_values()
        cur = {c: "USD" for c in lp_w.columns} | {c: "KRW" for c in kr_w.columns}
        w = pd.DataFrame(0.0, index=widx, columns=list(lp_w.columns) + list(kr_w.columns))
        w.loc[:, lp_w.columns] = 0.5 * lp_w.reindex(widx).fillna(0.0).values
        w.loc[:, kr_w.columns] = 0.5 * kr_w.reindex(widx).fillna(0.0).values
        fxw = fx.reindex(widx).ffill()
        px = {}
        for col in w.columns:
            s = real_symbol(col)
            if s in px:
                continue
            try:
                p = load_yfinance(s, "1d", "max")["close"].reindex(widx).ffill()
            except Exception:  # noqa: BLE001
                p = pd.Series(np.nan, index=widx)
            px[s] = p * fxw if cur[col] == "USD" else p
        pxk = pd.DataFrame({real_symbol(c): px[real_symbol(c)] for c in w.columns})
        cost_of = {real_symbol(c): (COST_US if cur[c] == "USD" else COST_KR) for c in w.columns}
        wreal = pd.DataFrame(0.0, index=widx, columns=sorted({real_symbol(c) for c in w.columns}))
        for c in w.columns:
            wreal[real_symbol(c)] += w[c]

        mi = metr(ideal)
        row = {"variant": vname, "ideal": mi, "seeds": {}}
        for seed in SEEDS:
            nav, q = integer_sim(seed, widx, wreal, pxk, cost_of)
            m = metr(nav)
            corr = nav.pct_change().corr(ideal.reindex(widx).ffill().pct_change())
            row["seeds"][int(seed)] = {**m, **q, "corr_ideal": float(corr),
                                       "cagr_drag": m["cagr"] - mi["cagr"],
                                       "final_krw": float(nav.iloc[-1])}
            results.append((vname, seed, mi, m, q, corr))
        globals().setdefault("_rows", []).append(row)

    bench_m = metr(bench.reindex(base_idx).ffill())
    print(f"yoon2m 시드×유니버스 실측 · 마찰 KR {COST_KR*100:.2f}%/US {COST_US*100:.2f}% 편도 · {FREQ}")
    print(f"벤치(50/50 패시브): CAGR {bench_m['cagr']*100:.2f}% MDD {bench_m['mdd']*100:.2f}% Sharpe {bench_m['sharpe']:.3f}\n")
    print(f"{'유니버스':12s}{'시드':>8s}{'CAGR':>8s}{'MDD':>9s}{'Sharpe':>8s}{'drag':>8s}{'비중오차':>9s}{'투자율':>8s}{'상관':>7s}")
    for vname, seed, mi, m, q, corr in results:
        print(f"{vname:12s}{seed/10000:7.0f}만{m['cagr']*100:7.2f}%{m['mdd']*100:8.2f}%{m['sharpe']:8.3f}"
              f"{(m['cagr']-mi['cagr'])*100:+7.2f}p{q['werr_mean']*100:8.2f}p{q['invested_mean']*100:7.1f}%{corr:7.3f}")
    print("\n이상적(연속비중) 기준선:")
    seen = set()
    for vname, _, mi, _, _, _ in results:
        if vname in seen:
            continue
        seen.add(vname)
        print(f"  {vname:12s} CAGR {mi['cagr']*100:6.2f}% MDD {mi['mdd']*100:7.2f}% Sharpe {mi['sharpe']:.3f}")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_seed_universe_sweep.json"
    out.write_text(json.dumps({
        "freq": FREQ, "topk": TOPK, "cost_kr": COST_KR, "cost_us": COST_US,
        "seeds": SEEDS, "bench": bench_m, "variants": globals()["_rows"],
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
