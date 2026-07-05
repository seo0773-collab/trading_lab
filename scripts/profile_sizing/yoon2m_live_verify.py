"""yoon2m 실거래 대비 검증: 이상적 백테스트 vs 200만원 정수주 실집행 시뮬.

이상적 백테스트는 연속비중 + 내부비용(5+5bps)만 반영한다. 실거래는 추가로
(1) 정수주 라운딩(소액일수록 추적오차 큼), (2) 실제 수수료, (3) 슬리피지,
(4) USD 다리 환전 스프레드를 문다. 여기서 200만원 시드로 월간 정수주 리밸런스를
실제 마찰 비용과 함께 시뮬레이션해 CAGR/MDD가 얼마나 보존되는지 본다.

소스: yoon2m 슬리브(top_k=3) portfolio_wave 전 구간 → 월간 종목별 목표비중.
가격: yfinance 일봉(USD는 환율로 원화환산). 벤치: 50/50 패시브.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2m_live_verify.py [--seed-krw 2000000]
"""
from __future__ import annotations

import argparse
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
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from profile_sizing.yoon2a_bootstrap import build_series  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

TOPK = 3
# 실거래 마찰(편도). 국내 ETF는 저렴, 해외는 수수료+환전스프레드로 높게 잡음(보수적).
COST_KR = 0.0015     # 0.15%/side (수수료+슬리피지)
COST_US = 0.0060     # 0.60%/side (수수료+슬리피지+환전 스프레드 ~0.5%)


def sleeve_wave(cfg_id: str, freq: str = "monthly",
                overrides: dict | None = None) -> pd.DataFrame:
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / f"{cfg_id}.json").read_text("utf-8")))
    if overrides:            # yoon2m sub_strategies의 inline override(top_k·universe 등)
        cfg.update(overrides)
    cfg["top_k"] = TOPK
    cfg["rebalance_freq"] = freq
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    w = art.extras["portfolio_wave"]
    if "time" in w.columns:
        w = w.set_index("time")
    w.index = pd.to_datetime(w.index)
    return w.drop(columns=[c for c in w.columns if c == "cash"], errors="ignore")


def real_symbol(col: str) -> str:
    return col[:-5] if col.endswith("_SAFE") else col


def metrics(eq: pd.Series, label: str) -> dict:
    eq = (eq / eq.iloc[0]).dropna()
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {"label": label, "cagr": p["cagr"], "mdd": p["max_drawdown"],
            "sharpe": p["sharpe"], "tot": eq.iloc[-1] - 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-krw", type=float, default=2_000_000)
    ap.add_argument("--cost-kr", type=float, default=COST_KR)
    ap.add_argument("--cost-us", type=float, default=COST_US)
    ap.add_argument("--freq", type=str, default="monthly", help="monthly|quarterly")
    args = ap.parse_args()
    P = args.seed_krw
    freq = args.freq
    globals()["COST_KR"], globals()["COST_US"] = args.cost_kr, args.cost_us

    # ---- 이상적 백테스트 (핸들러, 연속비중) ----
    cfg = json.loads((ROOT / "configs" / "strategies" / "yoon2m.json").read_text("utf-8"))
    cfg["rebalance_freq"] = freq
    for s in cfg["sub_strategies"]:
        s.setdefault("overrides", {})["rebalance_freq"] = freq
    h = Yoon1kHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    ideal = {ph: h.build_artifacts(raw, cfg, symbol="SYNTH", phase=ph, bars_per_year=252).equity
             for ph in ("all", "train", "validation")}

    # ---- 종목별 월간 목표비중(0.5 lp + 0.5 kr) ----
    sub_ov = {s["config"]: (s.get("overrides") or {}) for s in cfg["sub_strategies"]}
    lp = sleeve_wave("yoon1j_lp", freq, sub_ov.get("yoon1j_lp"))
    kr = sleeve_wave("yoon1j_kr", freq, sub_ov.get("yoon1j_kr"))
    idx = lp.index.intersection(kr.index).sort_values()
    cur = {}
    for c in lp.columns:
        cur[c] = "USD"
    for c in kr.columns:
        cur[c] = "KRW"
    w = pd.DataFrame(0.0, index=idx, columns=list(lp.columns) + list(kr.columns))
    w.loc[:, lp.columns] = 0.5 * lp.reindex(idx).fillna(0.0).values
    w.loc[:, kr.columns] = 0.5 * kr.reindex(idx).fillna(0.0).values

    # ---- 가격(원화) ----
    fx = load_yfinance("KRW=X", "1d", "max")["close"].reindex(idx).ffill()
    px = {}
    for col in w.columns:
        s = real_symbol(col)
        if s in px:
            continue
        try:
            p = load_yfinance(s, "1d", "max")["close"].reindex(idx).ffill()
        except Exception:  # noqa: BLE001
            p = pd.Series(np.nan, index=idx)
        px[s] = p * fx if cur[col] == "USD" else p
    pxk = pd.DataFrame({real_symbol(c): px[real_symbol(c)] for c in w.columns})
    cost_of = {real_symbol(c): (COST_US if cur[c] == "USD" else COST_KR) for c in w.columns}
    # _SAFE/일반 중복 컬럼을 실심볼 기준 비중 합산
    wreal = pd.DataFrame(0.0, index=idx, columns=sorted({real_symbol(c) for c in w.columns}))
    for c in w.columns:
        wreal[real_symbol(c)] += w[c]

    reb = set(rebalance_dates(idx, freq))
    syms = list(wreal.columns)
    shares = {s: 0.0 for s in syms}
    cash = P
    nav = np.empty(len(idx))
    for i, t in enumerate(idx):
        price = pxk.loc[t]
        val = cash + sum(shares[s] * (price[s] if not math.isnan(price[s]) else 0.0) for s in syms)
        if t in reb:
            tgt = wreal.loc[t]
            avail = {s: price[s] for s in syms if not math.isnan(price[s]) and price[s] > 0}
            # 1) 목표금액 → floor 정수주
            desired = {s: math.floor(val * tgt[s] / avail[s]) if s in avail else 0 for s in syms}
            # 2) 매도/매수 비용 차감하며 현금 제약 그리디 보정
            #    먼저 전량 목표로 맞춘 뒤 예산 초과분은 floor가 막아줌. 잔여현금으로 추적오차 최소화.
            new_sh = dict(desired)
            def spent_cash(sh):
                buy = sum(max(sh[s]-shares[s],0)*avail[s]*(1+cost_of[s]) for s in avail)
                sell = sum(max(shares[s]-sh[s],0)*avail[s]*(1-cost_of[s]) for s in avail)
                return buy - sell
            # 예산 내려갈 때까지 가장 과대 종목 1주씩 감축
            while spent_cash(new_sh) > cash + 1e-6:
                over = max(avail, key=lambda s: (new_sh[s]*avail[s]/val - tgt[s]) if new_sh[s]>0 else -9)
                if new_sh[over] <= 0:
                    break
                new_sh[over] -= 1
            # 잔여현금으로 가장 과소 종목 1주씩 추가
            while True:
                cand=[s for s in avail if spent_cash({**new_sh,s:new_sh[s]+1})<=cash+1e-6]
                if not cand: break
                u=max(cand, key=lambda s: tgt[s]-new_sh[s]*avail[s]/val)
                if tgt[u]-new_sh[u]*avail[u]/val <= 0: break
                new_sh[u]+=1
            cash -= spent_cash(new_sh)
            shares = {s: float(new_sh.get(s,0)) for s in syms}
            val = cash + sum(shares[s]*(price[s] if not math.isnan(price[s]) else 0.0) for s in syms)
        nav[i] = val
    real_nav = pd.Series(nav, index=idx)

    # ---- 벤치 ----
    _, bench, bidx = build_series()
    bench = bench.reindex(idx).ffill()

    print(f"yoon2m 실거래 대비 검증 · 시드 {P:,.0f}원 · {idx[0].date()}~{idx[-1].date()} {len(idx)}봉")
    print(f"실거래 마찰 가정: 국내 {COST_KR*100:.2f}%/편도, 해외 {COST_US*100:.2f}%/편도(환전 포함)\n")
    rows = [metrics(ideal["all"], "이상적 백테스트(연속비중, 5+5bps)"),
            metrics(real_nav, f"실집행 시뮬(정수주 {P/10000:.0f}만, 실마찰)"),
            metrics(bench, "패시브 50/50 벤치")]
    print(f"{'구분':36s}{'CAGR':>8}{'MDD':>9}{'Sharpe':>8}{'총수익':>10}")
    for m in rows:
        print(f"{m['label']:36s}{m['cagr']*100:7.2f}%{m['mdd']*100:8.2f}%{m['sharpe']:8.3f}{m['tot']*100:9.1f}%")

    di, dr = rows[0], rows[1]
    print(f"\n실거래 drag: CAGR {(di['cagr']-dr['cagr'])*100:+.2f}%p · "
          f"MDD {(dr['mdd']-di['mdd'])*100:+.2f}%p · Sharpe {dr['sharpe']-di['sharpe']:+.3f}")
    print("\n구간별 이상적 백테스트 CAGR/MDD/Sharpe:")
    for ph in ("all", "train", "validation"):
        m = metrics(ideal[ph], ph)
        print(f"  {ph:11s} CAGR {m['cagr']*100:6.2f}% · MDD {m['mdd']*100:7.2f}% · Sharpe {m['sharpe']:.3f}")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_live_verify.json"
    out.write_text(json.dumps({"seed_krw": P, "period": f"{idx[0].date()}~{idx[-1].date()}",
        "rebalance_freq": freq,
        "cost_kr": COST_KR, "cost_us": COST_US,
        "rows": [{k: (v if k=="label" else float(v)) for k,v in m.items()} for m in rows],
        "ideal_phases": {ph: {k:(v if k=="label" else float(v)) for k,v in metrics(ideal[ph],ph).items()}
                          for ph in ("all","train","validation")}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
