#!/usr/bin/env python
"""yoon1/yoon1b — 시총가중 시장(SPY) 및 확장 유니버스 대비 비교.

지금까지 주 벤치마크는 EW 지수(등가중·이론적·상시완전투자)였다. 그러나 실무 투자자가
실제로 살 수 있는 패시브 대안은 **시총가중 시장(SPY)**이다. 그래서 전략(yoon1·yoon1b)을
두 벤치마크와 동시에 비교한다:

  · EW 지수      : 유니버스 등가중 상시투자(기존 공정 벤치마크)
  · 시장(SPY)    : 시총가중 시장 프록시 buy & hold(투자 가능한 패시브 대안)

또한 기본 30종목 vs 부진 종목 섞은 확장 유니버스 양쪽에서 평가해 견고성을 본다.
점수는 gain과 무관하므로 유니버스마다 compute_universe 1회만 한다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/benchmark_compare.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import compute_universe, simulate_portfolio  # noqa: E402
from profile_sizing.run import slice_window  # noqa: E402
from profile_sizing.survivorship import BASE30, LAGGARDS  # noqa: E402

VARIANTS = [("yoon1", 1.0), ("yoon1b", 1.25)]


def _perf(nav: pd.Series, window, interval) -> dict:
    s = nav.reindex(window)
    eq = s / s.dropna().iloc[0]
    ret = s.pct_change().fillna(0.0)
    p = performance(eq, ret, interval)
    return {"cagr": p["cagr"], "sharpe": p["sharpe"], "mdd": p["max_drawdown"]}


def _f(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def evaluate(panels, spy, base_cfg, mf, top_k, rebal, label) -> list[dict]:
    scores, prices = compute_universe(panels, base_cfg)
    if prices.empty:
        return []
    idx = prices.index
    rows = []
    spy_nav = pd.Series(spy).reindex(idx).ffill() if spy is not None else None
    for phase in ("all", "test"):
        window = slice_window(idx, phase, base_cfg)
        ew_nav = None
        spy_p = (_perf(spy_nav, window, base_cfg.interval)
                 if spy_nav is not None else {"cagr": None, "sharpe": None, "mdd": None})
        for vname, gain in VARIANTS:
            sim = simulate_portfolio(
                scores, prices, base_cfg, top_k=top_k, rebal_freq=rebal,
                market_close=spy, market_ma_len=int(mf.get("ma_len", 200)),
                market_off_scale=float(mf.get("off_scale", 0.5)),
                exposure_gain=gain)
            if ew_nav is None:
                ew_nav = sim["benchmark_ew"]
            sp = _perf(sim["nav"], window, base_cfg.interval)
            ew = _perf(ew_nav, window, base_cfg.interval)
            rows.append({
                "universe": label, "phase": phase, "variant": vname,
                "cagr": sp["cagr"], "mdd": sp["mdd"], "sharpe": sp["sharpe"],
                "ew_cagr": ew["cagr"], "ew_mdd": ew["mdd"], "ew_sharpe": ew["sharpe"],
                "spy_cagr": spy_p["cagr"], "spy_mdd": spy_p["mdd"],
                "spy_sharpe": spy_p["sharpe"],
                "sharpe_vs_spy": (sp["sharpe"] - spy_p["sharpe"])
                if (sp["sharpe"] is not None and spy_p["sharpe"] is not None) else None,
            })
    return rows


def main() -> int:
    base_raw = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1.json").read_text())
    base_cfg = config_from_dict(base_raw)
    mf = base_raw.get("market_filter") or {}
    top_k = int(base_raw["top_k"])
    rebal = str(base_raw["rebalance_freq"])

    from run_kalman_pipeline import load_yfinance
    ext = BASE30 + [s for s in LAGGARDS if s not in BASE30]
    print(f"데이터 로드: 확장 {len(ext)}종목 + SPY ...", flush=True)
    all_panels = {}
    for s in ext:
        try:
            all_panels[s] = load_yfinance(s, base_cfg.interval, base_cfg.period)
        except Exception:  # noqa: BLE001
            continue
    try:
        spy = load_yfinance(mf.get("symbol", "SPY"), base_cfg.interval,
                            base_cfg.period)["close"]
    except Exception:  # noqa: BLE001
        spy = None

    base_panels = {s: all_panels[s] for s in BASE30 if s in all_panels}
    print(f"우량주 30({len(base_panels)}) 평가 ...", flush=True)
    rows = evaluate(base_panels, spy, base_cfg, mf, top_k, rebal, "우량주 30")
    print(f"확장({len(all_panels)}) 평가 ...", flush=True)
    rows += evaluate(all_panels, spy, base_cfg, mf, top_k, rebal, "확장(부진 포함)")

    md = _markdown(rows)
    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "benchmark_compare.md").write_text(md, encoding="utf-8")
    (outdir / "benchmark_compare.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {outdir / 'benchmark_compare.md'}")
    return 0


def _markdown(rows) -> str:
    lines = [
        "# yoon1/yoon1b — 시총가중 시장(SPY)·확장 유니버스 대비 비교",
        "",
        "전략을 EW 지수(등가중 상시투자)와 시총가중 시장 프록시(SPY buy&hold) 양쪽과 비교. "
        "top_k=20·monthly·floor=1.0·시장필터 ON. 벤치마크엔 비용 미부과.",
        "",
        "> ⚠️ **all 구간의 SPY 수치는 왜곡**이다: 유니버스 인덱스는 1962 시작인데 SPY는 "
        "1993부터라 64년 기간에 33년 성장만 펼쳐져 CAGR/Sharpe가 부당하게 낮게 나온다. "
        "**SPY와의 공정한 비교는 SPY가 전 구간 존재하는 holdout(test, 2013~) 구간**이다.",
        "",
        "| 유니버스 | 구간 | 전략 | 전략 CAGR | 전략 MDD | 전략 Sharpe | "
        "EW CAGR | EW Sharpe | SPY CAGR | SPY MDD | SPY Sharpe | 전략−SPY Sharpe |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['universe']} | {r['phase']} | {r['variant']} | "
            f"{_f(r['cagr'], True)} | {_f(r['mdd'], True)} | {_f(r['sharpe'])} | "
            f"{_f(r['ew_cagr'], True)} | {_f(r['ew_sharpe'])} | "
            f"{_f(r['spy_cagr'], True)} | {_f(r['spy_mdd'], True)} | "
            f"{_f(r['spy_sharpe'])} | **{_f(r['sharpe_vs_spy'])}** |")
    # 핵심 요약: 우량주30 test 기준.
    def pick(u, ph, v):
        for r in rows:
            if r["universe"] == u and r["phase"] == ph and r["variant"] == v:
                return r
        return None
    y1 = pick("우량주 30", "test", "yoon1")
    y1b = pick("우량주 30", "test", "yoon1b")
    lines += ["", "## 요약 (우량주 30, holdout test)"]
    if y1 and y1b:
        lines += [
            f"- SPY(시총가중 시장): CAGR {_f(y1['spy_cagr'], True)} · "
            f"MDD {_f(y1['spy_mdd'], True)} · Sharpe {_f(y1['spy_sharpe'])}",
            f"- yoon1 : CAGR {_f(y1['cagr'], True)} · MDD {_f(y1['mdd'], True)} · "
            f"Sharpe {_f(y1['sharpe'])} (vs SPY Sharpe {_f(y1['sharpe_vs_spy'])})",
            f"- yoon1b: CAGR {_f(y1b['cagr'], True)} · MDD {_f(y1b['mdd'], True)} · "
            f"Sharpe {_f(y1b['sharpe'])} (vs SPY Sharpe {_f(y1b['sharpe_vs_spy'])})",
            "",
            "*SPY는 단일 시총가중 지수라 분산된 우량주 바스켓보다 변동성이 다를 수 있음. "
            "전략−SPY Sharpe 부호로 시장 대비 위험조정 우열을 판단.*",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
