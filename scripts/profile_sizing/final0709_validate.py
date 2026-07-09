"""final0709 검증: 공식 파이프라인 point-estimate + 부트스트랩 + 롤링 3년 안정성
+ 회전율/비용 민감도.

final0709 = final0706과 완전히 동일한 구조에 위험게이트만 credit_spread(HYG/IEF)
→ credit_or_small_cap(HYG/IEF OR IWM/SPY 소형주 상대강도)으로 바꾼 파생.
사용자 지시: "다음단계 진행" — 등록 후 아직 안 한 3가지(공식 파이프라인 실측,
롤링 워크포워드, 비용 민감도)를 이 스크립트에서 전부 수행한다.

Part A: `Yoon1kHandler.build_artifacts` 공식 산출물로 final0706 vs final0709
        point estimate(여러 구간) + SPY/QQQ원화 벤치·서로 간 페어드 부트스트랩.
Part B: 롤링 3년 윈도우로 final0709가 final0706을 얼마나 일관되게 앞서는지.
Part C: 게이트 발동일수/에피소드 수 비교 + 왕복비용(0~50bp) 민감도
        (`yoon2_final_fx_credit_gate_regrid_cost.py`의 기존 검증된 방법론 재사용
        — 이 파트만 manual context(빠른 재계산용)를 쓰고, 공식 헤드라인 수치는
        Part A가 유일한 소스).

final0706.json/final0709.json 등 config는 건드리지 않는다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0709_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_kalman_pipeline import load_yfinance  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.metrics import coverage_label  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices, pct  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

from profile_sizing.yoon2_final_fx_validate import CURRENT, load_context, relmom_weights  # noqa: E402
from profile_sizing.yoon2_final_fx_leading_signals import credit_trigger  # noqa: E402
from profile_sizing.yoon2_final_fx_credit_gate_regrid_cost import (  # noqa: E402
    combine_with_ief_and_cost,
    denoise,
    episode_count,
    gate_weights,
)

BENCH_SYMBOLS = ["SPY", "QQQ"]
DATA_RESUME = pd.Timestamp("2009-04-09")
PRIMARY_START = pd.Timestamp("2011-01-01")
RALLY_START = pd.Timestamp("2025-01-01")
COST_BPS = [0, 5, 10, 25, 50]
MIN_CONSECUTIVE = 2


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    s = nav.reindex(window).ffill().dropna() if window is not None else nav.dropna()
    if len(s) < 60:
        return {}
    eq = s / s.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def paired_bootstrap(
    nav_a: pd.Series, nav_b: pd.Series, *, label_a: str, label_b: str,
    window: pd.DatetimeIndex, iters: int = 8000, block: int = 20, seed: int = 7,
) -> dict[str, Any]:
    idx = nav_a.dropna().index.intersection(nav_b.dropna().index)
    idx = idx.intersection(window).sort_values()
    a = nav_a.reindex(idx).ffill()
    b = nav_b.reindex(idx).ffill()
    ra = a.pct_change().fillna(0.0).to_numpy()
    rb = b.pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    bidx = block_indices(len(idx), block, iters, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    return {
        "label_a": label_a, "label_b": label_b,
        "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": len(idx),
        "prob": {
            "sharpe_a_gt_b": float((sa > sb).mean()),
            "cagr_a_gt_b": float((ca > cb).mean()),
            "mdd_a_shallower": float((ma > mb).mean()),
        },
    }


def part_a(handler: Yoon1kHandler, raw: pd.DataFrame, cfg706: dict, cfg709: dict) -> dict[str, Any]:
    print("[Part A] 공식 파이프라인 point estimate + 부트스트랩", flush=True)
    art706 = handler.build_artifacts(raw, cfg706, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    art709 = handler.build_artifacts(raw, cfg709, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav706, nav709 = art706.equity, art709.equity
    idx_full = nav706.index

    fx = handler._fx(raw, str(cfg706.get("fx_symbol", "KRW=X")))
    fx = pd.Series(fx).reindex(idx_full).ffill() if fx is not None else None
    bench_krw: dict[str, pd.Series] = {}
    for sym in BENCH_SYMBOLS:
        px = load_yfinance(sym, "1d", "max")["close"].reindex(idx_full).ffill()
        usd_nav = px / px.dropna().iloc[0]
        if fx is not None:
            bench_krw[sym] = (usd_nav * (fx / fx.dropna().iloc[0])).rename(sym)

    idx = idx_full[idx_full >= DATA_RESUME]
    train_frac = float(cfg706.get("train_frac", 0.6))
    validation_frac = float(cfg706.get("validation_frac", 0.2))
    v_end = int(len(idx_full) * (train_frac + validation_frac))
    phases = {
        "primary_2011 (권위)": idx[idx >= PRIMARY_START],
        "2011~2024 (랠리 제외)": idx[(idx >= PRIMARY_START) & (idx < RALLY_START)],
        "2025~ (랠리 구간)": idx[idx >= RALLY_START],
        "test (holdout)": idx_full[v_end:],
    }

    rows = []
    for phase, window in phases.items():
        rows.append({"phase": phase, "candidate": "final0706", **_clean_metric(perf_nav(nav706, window))})
        rows.append({"phase": phase, "candidate": "final0709", **_clean_metric(perf_nav(nav709, window))})
        for sym in BENCH_SYMBOLS:
            if sym in bench_krw:
                rows.append({"phase": phase, "candidate": f"{sym}_KRW_bnh",
                             **_clean_metric(perf_nav(bench_krw[sym], window))})

    for r in rows:
        print(f"  [{r['phase']}] {r['candidate']}: CAGR {_fmt_pct(r.get('cagr'))} "
              f"MDD {_fmt_pct(r.get('max_drawdown'))} Sharpe {_fmt_num(r.get('sharpe'))}", flush=True)

    primary_window = phases["primary_2011 (권위)"]
    boot = {
        "final0709_vs_final0706": paired_bootstrap(
            nav709, nav706, label_a="final0709", label_b="final0706", window=primary_window),
    }
    for sym, nav in bench_krw.items():
        boot[f"final0709_vs_{sym}_KRW"] = paired_bootstrap(
            nav709, nav, label_a="final0709", label_b=f"{sym}_KRW_bnh", window=primary_window)

    print("\n  [부트스트랩, 권위구간 2011~, 8000 iters block=20]", flush=True)
    for k, b in boot.items():
        print(f"    {k}: P(Sharpe>) {b['prob']['sharpe_a_gt_b']*100:.1f}% "
              f"P(CAGR>) {b['prob']['cagr_a_gt_b']*100:.1f}% "
              f"P(MDD얕음) {b['prob']['mdd_a_shallower']*100:.1f}%", flush=True)

    return {
        "raw_period": coverage_label(idx_full),
        "phases": {k: (f"{v[0].date()}~{v[-1].date()}" if len(v) else "-") for k, v in phases.items()},
        "rows": rows,
        "bootstrap": boot,
        "nav706": nav706,
        "nav709": nav709,
    }


def part_b(nav706: pd.Series, nav709: pd.Series) -> dict[str, Any]:
    print("\n[Part B] 롤링 3년 안정성 (final0709 vs final0706)", flush=True)
    idx = nav706.index[nav706.index >= DATA_RESUME]
    years = sorted({int(d.year) for d in idx})
    rows = []
    for start in range(years[0], years[-1] - 2 + 1):
        window = idx[(idx.year >= start) & (idx.year <= start + 2)]
        if len(window) < 500:
            continue
        p706 = _clean_metric(perf_nav(nav706, window))
        p709 = _clean_metric(perf_nav(nav709, window))
        rows.append({"window": f"{start}-{start + 2}", "final0706": p706, "final0709": p709})
    wins = {
        "sharpe": sum((r["final0709"]["sharpe"] or -999) > (r["final0706"]["sharpe"] or -999) for r in rows),
        "cagr": sum((r["final0709"]["cagr"] or -999) > (r["final0706"]["cagr"] or -999) for r in rows),
        "mdd": sum((r["final0709"]["max_drawdown"] or -999) > (r["final0706"]["max_drawdown"] or -999) for r in rows),
        "n": len(rows),
    }
    print(f"  {len(rows)}개 윈도우 중 final0709가 이긴 횟수: "
          f"Sharpe {wins['sharpe']}/{wins['n']}, CAGR {wins['cagr']}/{wins['n']}, "
          f"MDD더얕음 {wins['mdd']}/{wins['n']}", flush=True)
    for r in rows:
        print(f"    {r['window']}: final0706 Sharpe {_fmt_num(r['final0706']['sharpe'])} "
              f"MDD {_fmt_pct(r['final0706']['max_drawdown'])} | final0709 Sharpe "
              f"{_fmt_num(r['final0709']['sharpe'])} MDD {_fmt_pct(r['final0709']['max_drawdown'])}", flush=True)
    return {"rows": rows, "wins": wins}


def part_c() -> dict[str, Any]:
    print("\n[Part C] 게이트 발동빈도 + 회전율/비용 민감도 (manual context, "
          "regrid_cost.py 방법론 재사용)", flush=True)
    ctx = load_context()
    idx = ctx.idx
    usd_series = relmom_weights(ctx, **CURRENT)[ctx.sub_ids[0]].reindex(idx).ffill().fillna(0.5)
    ief_px = load_yfinance("IEF", "1d", "max")["close"].reindex(idx).ffill()
    ief_nav = (ief_px / ief_px.dropna().iloc[0]).rename("IEF")

    credit_only = denoise(credit_trigger(idx), MIN_CONSECUTIVE)
    iwm = load_yfinance("IWM", "1d", "max")["close"].reindex(idx).ffill()
    spy = load_yfinance("SPY", "1d", "max")["close"].reindex(idx).ffill()
    ratio = (iwm / spy).reindex(idx).ffill()
    mom = ratio.pct_change(10)
    below_sma = ratio < ratio.rolling(20).mean()
    small_cap_raw = ((mom < -0.02) & below_sma).fillna(False)
    small_cap_only = denoise(small_cap_raw, MIN_CONSECUTIVE)
    combined = denoise(credit_trigger(idx) | small_cap_raw, MIN_CONSECUTIVE)

    trig_stats = {
        "credit_only(final0706)": {
            "days": int(credit_only.sum()), "episodes": episode_count(credit_only),
        },
        "credit_or_small_cap(final0709)": {
            "days": int(combined.sum()), "episodes": episode_count(combined),
        },
    }
    print(f"  발동일수/에피소드: 기존(신용단독) {trig_stats['credit_only(final0706)']}, "
          f"신규(OR결합) {trig_stats['credit_or_small_cap(final0709)']}", flush=True)

    cost_rows = []
    for label, trig in [("final0706(credit_only)", credit_only), ("final0709(credit_or_small_cap)", combined)]:
        turnovers = None
        for bps in COST_BPS:
            nav, turnover = combine_with_ief_and_cost(
                ctx, gate_weights(ctx, usd_series, trig, 0.5), ief_nav, bps)
            turnovers = turnover
            m = _clean_metric(perf_nav(nav))
            cost_rows.append({"label": label, "cost_bps": bps, "turnover": turnover, **m})
    print("\n  [비용 민감도, 0~50bp 왕복]", flush=True)
    for r in cost_rows:
        print(f"    {r['label']} cost={r['cost_bps']}bp: CAGR {_fmt_pct(r.get('cagr'))} "
              f"MDD {_fmt_pct(r.get('max_drawdown'))} Sharpe {_fmt_num(r.get('sharpe'))}", flush=True)

    return {"trigger_stats": trig_stats, "cost_rows": cost_rows}


def main() -> None:
    handler = Yoon1kHandler()
    cfg709 = json.loads((ROOT / "configs" / "strategies" / "final0709.json").read_text("utf-8"))
    cfg706 = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    print("데이터 로드 중 (final0709 config 기준, __GATE__에 HYG/IEF/IWM/SPY 전부 포함 — "
          "final0706 아티팩트도 같은 raw로 재사용)...", flush=True)
    raw = handler.load_data("PORTFOLIO", cfg709, synthetic=False)

    a = part_a(handler, raw, cfg706, cfg709)
    b = part_b(a["nav706"], a["nav709"])
    c = part_c()

    result = {
        "part_a": {k: v for k, v in a.items() if k not in ("nav706", "nav709")},
        "part_b": b,
        "part_c": c,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0709_validate.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
