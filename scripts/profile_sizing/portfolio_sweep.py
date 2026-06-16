#!/usr/bin/env python
"""profile-portfolio-v1 파라미터 스윕 (top_k × 리밸런스 × 추세 floor).

검증(validation) 구간에서 조합을 평가해 **위험조정(Sharpe) 최선**을 고르고, 한 번도
탐색에 쓰지 않은 holdout(test) 구간에서 그 선택이 유지되는지 확인한다(과적합 방지).
전략에 데이터 적합은 없지만 파라미터 '선택'이 일어나므로 validation→test 분리를 지킨다.

점수(scores)는 floor에만 의존하므로 floor마다 compute_universe 1회만 하고, top_k·
리밸런스 조합은 같은 점수/가격으로 simulate만 반복해 비용을 줄인다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/portfolio_sweep.py
"""
from __future__ import annotations

import argparse
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

DEFAULT_UNIVERSE = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "BAC", "WFC", "GS", "JNJ", "PFE", "MRK", "UNH",
    "KO", "PEP", "PG", "WMT", "MCD", "COST",
    "CAT", "DE", "HON", "GE", "XOM", "CVX", "NEE", "DIS",
]


def _phase_perf(sim: dict, prices_index, phase: str, cfg) -> dict:
    window = slice_window(prices_index, phase, cfg)
    nav = sim["nav"].reindex(window)
    eq = (nav / nav.iloc[0])
    ret = nav.pct_change().fillna(0.0)
    # 공정 벤치마크 = 상시 완전투자 EW 지수(진짜 buy&hold는 cash-drag 편향).
    bench = sim["benchmark_ew"].reindex(window)
    beq = (bench / bench.iloc[0])
    bret = bench.pct_change().fillna(0.0)
    p = performance(eq, ret, cfg.interval)
    b = performance(beq, bret, cfg.interval)
    return {
        "cagr": p["cagr"], "sharpe": p["sharpe"], "mdd": p["max_drawdown"],
        "bnh_cagr": b["cagr"], "bnh_sharpe": b["sharpe"], "bnh_mdd": b["max_drawdown"],
        "sharpe_minus_bnh": (p["sharpe"] - b["sharpe"])
        if (p["sharpe"] is not None and b["sharpe"] is not None) else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--top-k", default="5,10,15,20")
    parser.add_argument("--floors", default="0.0,0.7,0.9,1.0")
    parser.add_argument("--rebals", default="monthly,weekly")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tks = [int(x) for x in args.top_k.split(",")]
    floors = [float(x) for x in args.floors.split(",")]
    rebals = [s.strip() for s in args.rebals.split(",")]

    from run_kalman_pipeline import load_yfinance
    base_raw = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1.json").read_text()
    )
    base_cfg = config_from_dict(base_raw)

    print(f"데이터 로드: {len(symbols)}종목 ...", flush=True)
    panels = {}
    for s in symbols:
        try:
            panels[s] = load_yfinance(s, base_cfg.interval, base_cfg.period)
        except Exception:  # noqa: BLE001
            continue

    rows = []
    for floor in floors:
        cfg = replace(base_cfg, trend_overlay=replace(
            base_cfg.trend_overlay, enabled=True, floor=floor))
        print(f"[floor={floor}] compute_universe ...", flush=True)
        scores, prices = compute_universe(panels, cfg)
        for top_k in tks:
            for rebal in rebals:
                sim = simulate_portfolio(scores, prices, cfg,
                                         top_k=top_k, rebal_freq=rebal)
                val = _phase_perf(sim, prices.index, "validation", cfg)
                test = _phase_perf(sim, prices.index, "test", cfg)
                rows.append({
                    "top_k": top_k, "rebal": rebal, "floor": floor,
                    "val_sharpe": val["sharpe"], "val_cagr": val["cagr"],
                    "val_mdd": val["mdd"], "val_sharpe_vs_bnh": val["sharpe_minus_bnh"],
                    "test_sharpe": test["sharpe"], "test_cagr": test["cagr"],
                    "test_mdd": test["mdd"], "test_sharpe_vs_bnh": test["sharpe_minus_bnh"],
                    "bnh_val_sharpe": val["bnh_sharpe"], "bnh_test_sharpe": test["bnh_sharpe"],
                })
                print(f"  top_k={top_k} rebal={rebal} floor={floor} | "
                      f"val Sharpe {_f(val['sharpe'])} (B&H {_f(val['bnh_sharpe'])}) | "
                      f"test Sharpe {_f(test['sharpe'])}", flush=True)

    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_json(args.outdir / "portfolio_sweep.json", orient="records", indent=2)

    md = _to_markdown(df)
    (args.outdir / "portfolio_sweep.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {args.outdir / 'portfolio_sweep.md'}")
    return 0


def _f(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def _to_markdown(df: pd.DataFrame) -> str:
    best = df.iloc[0]
    lines = [
        "# profile-portfolio-v1 파라미터 스윕 (validation 선정 → test 확인)",
        "",
        "검증(validation) Sharpe 기준 정렬. **선택 기준은 validation, test는 확인용**(과적합 점검).",
        "",
        f"## 검증 최선 조합: top_k={int(best['top_k'])}, "
        f"rebal={best['rebal']}, floor={best['floor']}",
        "",
        f"- validation: Sharpe {_f(best['val_sharpe'])} (B&H {_f(best['bnh_val_sharpe'])}, "
        f"차 {_f(best['val_sharpe_vs_bnh'])}) · CAGR {_f(best['val_cagr'], True)} · "
        f"MDD {_f(best['val_mdd'], True)}",
        f"- **test(holdout)**: Sharpe {_f(best['test_sharpe'])} "
        f"(B&H {_f(best['bnh_test_sharpe'])}, 차 {_f(best['test_sharpe_vs_bnh'])}) · "
        f"CAGR {_f(best['test_cagr'], True)} · MDD {_f(best['test_mdd'], True)}",
        "",
        "## 전체 조합 (validation Sharpe 내림차순)",
        "",
        "| top_k | 리밸런스 | floor | val Sharpe | val−B&H | val CAGR | val MDD | "
        "test Sharpe | test−B&H | test CAGR | test MDD |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['top_k'])} | {r['rebal']} | {r['floor']} | "
            f"{_f(r['val_sharpe'])} | {_f(r['val_sharpe_vs_bnh'])} | "
            f"{_f(r['val_cagr'], True)} | {_f(r['val_mdd'], True)} | "
            f"{_f(r['test_sharpe'])} | {_f(r['test_sharpe_vs_bnh'])} | "
            f"{_f(r['test_cagr'], True)} | {_f(r['test_mdd'], True)} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
