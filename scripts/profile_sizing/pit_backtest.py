#!/usr/bin/env python
"""PIT(point-in-time) S&P 500 유니버스로 yoon1b 재검증 — 생존편향 영향 정량화.

각 날짜에 '그 시점 S&P 500 구성원'만 후보가 되도록 점수를 마스킹한다(편입일 마스킹).
이로써 '2026년 승자 30개를 손으로 고른' 편향을 제거한다. 비교:
  · PIT yoon1b  vs  SPY(시장)  vs  PIT 등가중(공정·같은 유니버스)
  · 그리고 기존 승자30 yoon1b 와 나란히 → 생존편향이 얼마나 부풀렸는지.

한계: yfinance에 현재 구성원만 있고 상폐/편출 종목은 빠짐 → 잔여 생존편향 존재.
따라서 'PIT에서도 줄어든 우위'는 보수적 하한이다(실제 편향은 더 클 수 있음).

선행: pit_universe.py(멤버십 캐시) + 현재구성 가격 다운로드 완료.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/pit_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.portfolio import compute_universe, simulate_portfolio  # noqa: E402
from trading_lab.market_data import market_data_path  # noqa: E402
from trading_lab.portfolio_universes import STOCK_UNIVERSE  # noqa: E402

CACHE = ROOT / "var" / "pit"


def perf(nav: pd.Series) -> dict:
    nav = pd.Series(nav).dropna()
    if len(nav) < 60:
        return {"cagr": None, "sharpe": None, "mdd": None}
    eq = nav / nav.iloc[0]
    ret = eq.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    return {"cagr": float(eq.iloc[-1]) ** (1 / years) - 1,
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else None,
            "mdd": float((eq / eq.cummax() - 1).min())}


def _f(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def load_cached(symbols, interval, period):
    from run_kalman_pipeline import load_yfinance
    panels = {}
    for s in symbols:
        if not market_data_path(s, interval).exists():
            continue
        try:
            d = load_yfinance(s, interval, period)
            if d is not None and not d.empty and "close" in d:
                panels[s] = d
        except Exception:  # noqa: BLE001
            continue
    return panels


def build_mask(index, columns, membership: dict) -> pd.DataFrame:
    me_dates = sorted(pd.Timestamp(k) for k in membership)
    me_sets = [set(membership[d.strftime("%Y-%m-%d")]) for d in me_dates]
    pos = np.clip(np.searchsorted(me_dates, index, side="right") - 1, 0, len(me_dates) - 1)
    mask = pd.DataFrame(False, index=index, columns=columns)
    for s in columns:
        grid = np.array([s in me_sets[k] for k in range(len(me_dates))])
        mask[s] = grid[pos]
    return mask


def main() -> int:
    raw = json.loads((ROOT / "configs" / "strategies" / "yoon1b.json").read_text())
    cfg = config_from_dict(raw)
    mf = raw.get("market_filter") or {}
    top_k, rebal = int(raw["top_k"]), str(raw["rebalance_freq"])
    mkw = dict(market_ma_len=int(mf.get("ma_len", 200)),
               market_off_scale=float(mf.get("off_scale", 0.5)), exposure_gain=1.25)

    membership = json.loads((CACHE / "sp500_membership.json").read_text())
    current = json.loads((CACHE / "sp500_current.json").read_text())

    from run_kalman_pipeline import load_yfinance
    spy = load_yfinance("SPY", cfg.interval, cfg.period)["close"]

    print(f"PIT 유니버스 가격 로드(캐시된 현재구성) ...", flush=True)
    panels = load_cached(current, cfg.interval, cfg.period)
    print(f"  로드 {len(panels)}/{len(current)}종", flush=True)
    scores, prices = compute_universe(panels, cfg)
    idx = prices.index

    print("PIT 멤버십 마스크 ...", flush=True)
    mask = build_mask(idx, list(prices.columns), membership)
    scores_pit = scores.where(mask, 0.0)

    # PIT 등가중 벤치(공정): 매일 '그 시점 회원 & 상장된' 종목 등가중.
    elig = mask & prices.notna()
    rets = prices.pct_change()
    pit_ew = (1 + rets.where(elig).mean(axis=1, skipna=True).fillna(0.0)).cumprod()

    print("PIT yoon1b 시뮬 ...", flush=True)
    sim_pit = simulate_portfolio(scores_pit, prices, cfg, top_k=top_k,
                                 rebal_freq=rebal, market_close=spy, **mkw)

    print("기존 승자30 yoon1b 시뮬(대조) ...", flush=True)
    p30 = load_cached(STOCK_UNIVERSE, cfg.interval, cfg.period)
    sc30, px30 = compute_universe(p30, cfg)
    sim30 = simulate_portfolio(sc30, px30, cfg, top_k=top_k, rebal_freq=rebal,
                               market_close=spy, **mkw)

    spy_nav = pd.Series(spy)
    rows = []
    for label, nav in [("PIT yoon1b (S&P500 시점구성)", sim_pit["nav"]),
                       ("PIT 등가중(공정 벤치)", pit_ew),
                       ("승자30 yoon1b (기존)", sim30["nav"]),
                       ("SPY(시장)", spy_nav)]:
        for phase, w in [("all", nav.index),
                         ("2013~", nav.index[nav.index >= "2013-01-01"])]:
            sub = nav.reindex(w).dropna()
            common = sub.index.intersection(spy_nav.index)
            p = perf(sub)
            rows.append({"label": label, "phase": phase, **p})

    md = _markdown(rows, len(panels), len(current))
    out = ROOT / "reports" / "profile_sizing" / "pit_backtest.md"
    out.write_text(md, encoding="utf-8")
    (ROOT / "reports" / "profile_sizing" / "pit_backtest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {out}")
    return 0


def _markdown(rows, n_loaded, n_current) -> str:
    def g(label, phase):
        for r in rows:
            if r["label"] == label and r["phase"] == phase:
                return r
        return {}
    lines = [
        "# PIT S&P 500 유니버스 — yoon1b 생존편향 재검증",
        "",
        f"각 시점 S&P 500 구성원만 후보(편입일 마스킹). 로드 {n_loaded}/{n_current}종(현재구성). "
        "top_k 20·monthly·gain 1.25·SPY필터. ⚠️ 상폐/편출 종목은 yfinance에 없어 제외 → "
        "잔여 생존편향 있음(개선폭은 보수적 하한).",
        "",
        "| 구성 | 구간 | CAGR | MDD | Sharpe |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['label']} | {r['phase']} | {_f(r['cagr'], True)} | "
                     f"{_f(r['mdd'], True)} | {_f(r['sharpe'])} |")
    a = g("PIT yoon1b (S&P500 시점구성)", "2013~")
    e = g("PIT 등가중(공정 벤치)", "2013~")
    s = g("SPY(시장)", "2013~")
    w30 = g("승자30 yoon1b (기존)", "2013~")
    lines += [
        "",
        "## 해석 (2013~ 기준)",
        f"- PIT yoon1b: CAGR {_f(a.get('cagr'),True)} / Sharpe {_f(a.get('sharpe'))} / MDD {_f(a.get('mdd'),True)}",
        f"- 승자30 yoon1b: CAGR {_f(w30.get('cagr'),True)} / Sharpe {_f(w30.get('sharpe'))} / MDD {_f(w30.get('mdd'),True)}",
        f"- SPY: CAGR {_f(s.get('cagr'),True)} / Sharpe {_f(s.get('sharpe'))}",
        f"- PIT 등가중(같은 유니버스 공정 벤치): CAGR {_f(e.get('cagr'),True)} / Sharpe {_f(e.get('sharpe'))}",
        "",
        "→ 승자30 대비 PIT에서 수익/Sharpe가 얼마나 내려갔는지가 **손픽 생존편향의 크기**. "
        "PIT yoon1b가 **PIT 등가중 대비** 여전히 낮은 낙폭/유사~우위 Sharpe면, 알고리즘의 "
        "방어 가치는 유니버스와 무관하게 유지된다는 뜻(이게 핵심 신뢰 포인트).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
