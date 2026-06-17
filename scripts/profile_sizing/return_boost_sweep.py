#!/usr/bin/env python
"""yoon1 2순위 — 수익 갭 좁히기 후보 스윕 (validation 선택 → test 확인).

yoon1은 '노출 = mean(top-K 점수)' 구조라 평범한 상승장에서도 평균이 희석돼 풀투자에
못 미친다 → 절대수익을 EW 지수에 양보한다. 방어 성격(약세장 현금화)을 유지한 채 상승장
참여를 끌어올릴 두 레버를 격자 탐색한다:

  ① exposure_gain  : 노출에 게인 후 1.0 클립. 평상장 풀투자 근접, 약세장은 낮게 유지.
  ② recovery 가속  : RECOVERY cap·단계 상향. 하락 후 반등 구간 재진입을 앞당김
                     (수익이 새는 지점이 주로 회복 초기라는 가설).

regime cap(recovery)은 종목 점수(compute_universe)에 영향을 주므로 recovery 프로필마다
compute_universe를 1회만 하고, gain은 simulate에서만 바꿔 비용을 줄인다.

선택 기준 = validation에서 **EW 대비 CAGR 갭을 줄이되 Sharpe가 yoon1 baseline 이상**.
그 선택이 holdout(test)에서도 유지되는지 확인한다(과적합 점검).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/return_boost_sweep.py
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

GAINS = [1.0, 1.25, 1.5, 2.0]
# (라벨, RECOVERY cap, 단계 caps)
RECOVERY_PROFILES = [
    ("base", 0.6, [0.6, 0.8, 1.0]),
    ("fast", 0.8, [0.8, 1.0, 1.0]),
]


def _phase(sim, idx, phase, cfg) -> dict:
    window = slice_window(idx, phase, cfg)
    nav = sim["nav"].reindex(window)
    eq = nav / nav.dropna().iloc[0]
    ret = nav.pct_change().fillna(0.0)
    bench = sim["benchmark_ew"].reindex(window)
    beq = bench / bench.dropna().iloc[0]
    bret = bench.pct_change().fillna(0.0)
    p = performance(eq, ret, cfg.interval)
    b = performance(beq, bret, cfg.interval)
    return {
        "cagr": p["cagr"], "sharpe": p["sharpe"], "mdd": p["max_drawdown"],
        "ew_cagr": b["cagr"], "ew_sharpe": b["sharpe"], "ew_mdd": b["max_drawdown"],
        "cagr_gap": (p["cagr"] - b["cagr"])
        if (p["cagr"] is not None and b["cagr"] is not None) else None,
        "sharpe_vs_ew": (p["sharpe"] - b["sharpe"])
        if (p["sharpe"] is not None and b["sharpe"] is not None) else None,
    }


def _f(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def main() -> int:
    base_raw = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1.json").read_text())
    base_cfg = config_from_dict(base_raw)
    universe = list(base_raw["universe"])
    mf = base_raw.get("market_filter") or {}
    top_k = int(base_raw["top_k"])
    rebal = str(base_raw["rebalance_freq"])

    from run_kalman_pipeline import load_yfinance
    print(f"데이터 로드: {len(universe)}종목 + SPY ...", flush=True)
    panels = {}
    for s in universe:
        try:
            panels[s] = load_yfinance(s, base_cfg.interval, base_cfg.period)
        except Exception:  # noqa: BLE001
            continue
    try:
        spy = load_yfinance(mf.get("symbol", "SPY"), base_cfg.interval,
                            base_cfg.period)["close"]
    except Exception:  # noqa: BLE001
        spy = None

    rows = []
    for rlabel, rcap, rcaps in RECOVERY_PROFILES:
        cfg = replace(base_cfg, regime_cap=replace(
            base_cfg.regime_cap, RECOVERY=rcap,
            recovery_stage_caps=tuple(rcaps)))
        print(f"[recovery={rlabel}] compute_universe ...", flush=True)
        scores, prices = compute_universe(panels, cfg)
        idx = prices.index
        for gain in GAINS:
            sim = simulate_portfolio(
                scores, prices, cfg, top_k=top_k, rebal_freq=rebal,
                market_close=spy, market_ma_len=int(mf.get("ma_len", 200)),
                market_off_scale=float(mf.get("off_scale", 0.5)),
                exposure_gain=gain)
            val = _phase(sim, idx, "validation", cfg)
            test = _phase(sim, idx, "test", cfg)
            rows.append({"recovery": rlabel, "gain": gain,
                         "val": val, "test": test})
            print(f"  recovery={rlabel} gain={gain:g} | "
                  f"val CAGR {_f(val['cagr'], True)} (gap {_f(val['cagr_gap'], True)}) "
                  f"Sharpe {_f(val['sharpe'])} | "
                  f"test CAGR {_f(test['cagr'], True)} Sharpe {_f(test['sharpe'])}",
                  flush=True)

    baseline = next(r for r in rows if r["recovery"] == "base" and r["gain"] == 1.0)
    base_val_sharpe = baseline["val"]["sharpe"]

    # 선택: val Sharpe가 baseline 이상이면서 val CAGR gap(음수)이 가장 0에 가까운 조합.
    eligible = [r for r in rows
                if r["val"]["sharpe"] is not None
                and base_val_sharpe is not None
                and r["val"]["sharpe"] >= base_val_sharpe - 1e-9]
    pick = max(eligible, key=lambda r: r["val"]["cagr_gap"]) if eligible else baseline

    md = _markdown(rows, baseline, pick)
    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "return_boost_sweep.md").write_text(md, encoding="utf-8")
    (outdir / "return_boost_sweep.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {outdir / 'return_boost_sweep.md'}")
    return 0


def _markdown(rows, baseline, pick) -> str:
    def line(r):
        v, t = r["val"], r["test"]
        star = " ★선택" if r is pick else (
            " (yoon1)" if r is baseline else "")
        return (
            f"| {r['recovery']} | {r['gain']:g}{star} | "
            f"{_f(v['cagr'], True)} | {_f(v['cagr_gap'], True)} | {_f(v['mdd'], True)} | "
            f"{_f(v['sharpe'])} | {_f(v['sharpe_vs_ew'])} | "
            f"{_f(t['cagr'], True)} | {_f(t['cagr_gap'], True)} | {_f(t['mdd'], True)} | "
            f"{_f(t['sharpe'])} | {_f(t['sharpe_vs_ew'])} |")

    bv, pv, pt = baseline["val"], pick["val"], pick["test"]
    bt = baseline["test"]
    improved = (pick is not baseline)
    lines = [
        "# yoon1 수익 갭 좁히기 스윕 (validation 선택 → test 확인)",
        "",
        "고정: top_k=20 · monthly · trend floor=1.0 · 시장필터 ON. 레버 = 노출 게인 × "
        "RECOVERY 프로필. 벤치마크=상시완전투자 EW 지수. **선택 기준 = val Sharpe ≥ "
        "yoon1 이면서 val CAGR 갭(음수)을 가장 0에 근접**.",
        "",
        f"## 선택: recovery={pick['recovery']}, gain={pick['gain']:g}"
        + ("  (= yoon1 그대로가 최선, 개선 후보 없음)" if not improved else ""),
        "",
    ]
    if improved:
        lines += [
            f"- **validation**: CAGR {_f(bv['cagr'], True)} → {_f(pv['cagr'], True)} "
            f"(EW 대비 갭 {_f(bv['cagr_gap'], True)} → {_f(pv['cagr_gap'], True)}), "
            f"Sharpe {_f(bv['sharpe'])} → {_f(pv['sharpe'])}, "
            f"MDD {_f(bv['mdd'], True)} → {_f(pv['mdd'], True)}",
            f"- **test(holdout)**: CAGR {_f(bt['cagr'], True)} → {_f(pt['cagr'], True)} "
            f"(갭 {_f(bt['cagr_gap'], True)} → {_f(pt['cagr_gap'], True)}), "
            f"Sharpe {_f(bt['sharpe'])} → {_f(pt['sharpe'])}, "
            f"MDD {_f(bt['mdd'], True)} → {_f(pt['mdd'], True)}",
            "",
            "→ validation에서 고른 개선이 test에서도 유지되면 채택 검토, 아니면 보류.",
        ]
    else:
        lines += [
            "어떤 게인/recovery 조합도 yoon1 baseline의 val Sharpe를 지키면서 CAGR 갭을 "
            "줄이지 못했다 → **현행 yoon1이 이 격자에서 최선**. 수익을 더 끌어올리면 "
            "위험조정이 깎이는 트레이드오프가 확인됨(방어 성격이 본질).",
        ]
    lines += [
        "",
        "## 전체 격자",
        "",
        "| recovery | gain | val CAGR | val 갭 | val MDD | val Sharpe | val−EW | "
        "test CAGR | test 갭 | test MDD | test Sharpe | test−EW |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [line(r) for r in rows]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
