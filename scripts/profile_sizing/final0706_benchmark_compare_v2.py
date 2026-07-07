"""final0706 vs SPY/QQQ/QLD buy&hold 비교 v2 — 수치 정직화판 (개선계획 A3).

v1(final0706_benchmark_compare.py) 대비 세 가지를 고친다
(근거: reports/profile_sizing/final0706_data_gap_diagnosis.md):

1. **달력 기준 연환산** — engine.performance가 metrics.years_elapsed를 쓰도록
   수정됨(A1). Yahoo .KS 데이터 갭(2007-02-08~2009-04-08) 탓에 봉 수 기준이
   CAGR을 부풀렸다(공표 16.84% → 달력 13.96%).
2. **평가 표준 시작점** — stray 8봉(2007-01~02)은 항상 제외. 권위 구간은
   2011-01-01~(반등만 포함되는 2009-04~2010 비대칭 배제 + KR 유니버스 7/8 존재).
3. **레짐 분해** — 2011~2024(2025~26 랠리 제외)와 2025~(랠리 구간)를 분리 보고.
   구간 라벨에 결측 갭을 병기(coverage_label).

방법론(공식 파이프라인 equity 그대로, 원화환산 벤치, 페어드 부트스트랩)은 v1과 동일.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_benchmark_compare_v2.py
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

BENCH_SYMBOLS = ["SPY", "QQQ", "QLD"]
DATA_RESUME = pd.Timestamp("2009-04-09")   # .KS 갭 이후 데이터 연속 시작
PRIMARY_START = pd.Timestamp("2011-01-01")  # 권위 구간 시작(진단 노트 규칙 1)
RALLY_START = pd.Timestamp("2025-01-01")    # 2025~26 랠리 분리점(주도=KOSPI, regime_attribution 참고)


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
        "ci95": {
            "excess_sharpe": pct(sa - sb),
            "excess_cagr": pct(ca - cb),
            "excess_mdd": pct(ma - mb),
        },
    }


def main() -> None:
    config = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)
    art = handler.build_artifacts(raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    strat_nav = art.equity
    idx_full = strat_nav.index

    fx = handler._fx(raw, str(config.get("fx_symbol", "KRW=X")))
    fx = pd.Series(fx).reindex(idx_full).ffill() if fx is not None else None

    bench_krw: dict[str, pd.Series] = {}
    bench_usd: dict[str, pd.Series] = {}
    print("벤치마크 종목 로드 중...", flush=True)
    for sym in BENCH_SYMBOLS:
        px = load_yfinance(sym, "1d", "max")["close"].reindex(idx_full).ffill()
        usd_nav = (px / px.dropna().iloc[0]).rename(sym)
        bench_usd[sym] = usd_nav
        if fx is not None:
            bench_krw[sym] = (usd_nav * (fx / fx.dropna().iloc[0])).rename(sym)

    # holdout(test)은 파이프라인과 동일하게 전체 인덱스 frac으로 정의(비교 가능성 유지)
    train_frac = float(config.get("train_frac", 0.6))
    validation_frac = float(config.get("validation_frac", 0.2))
    v_end = int(len(idx_full) * (train_frac + validation_frac))

    idx = idx_full[idx_full >= DATA_RESUME]  # stray 8봉 상시 제외
    phases = {
        "primary_2011 (권위)": idx[idx >= PRIMARY_START],
        "2011~2024 (2025~26 랠리 제외)": idx[(idx >= PRIMARY_START) & (idx < RALLY_START)],
        "2025~ (랠리 구간)": idx[idx >= RALLY_START],
        "test (holdout)": idx_full[v_end:],
        "since_2009-04 (참고)": idx,
    }

    rows = []
    for phase, window in phases.items():
        rows.append({"phase": phase, "candidate": "final0706",
                     **_clean_metric(perf_nav(strat_nav, window))})
        for sym in BENCH_SYMBOLS:
            if fx is not None:
                rows.append({"phase": phase, "candidate": f"{sym}_KRW_bnh",
                             **_clean_metric(perf_nav(bench_krw[sym], window))})
            rows.append({"phase": phase, "candidate": f"{sym}_USD_bnh",
                         **_clean_metric(perf_nav(bench_usd[sym], window))})

    boot_results = {}
    primary_window = phases["primary_2011 (권위)"]
    for sym in BENCH_SYMBOLS:
        if fx is not None:
            boot_results[f"final0706_vs_{sym}_KRW"] = paired_bootstrap(
                strat_nav, bench_krw[sym], label_a="final0706",
                label_b=f"{sym}_KRW_bnh", window=primary_window)

    result = {
        "raw_period": coverage_label(idx_full),
        "standard_start": str(PRIMARY_START.date()),
        "phases": {k: (f"{v[0].date()}~{v[-1].date()}" if len(v) else "-")
                   for k, v in phases.items()},
        "rows": rows,
        "bootstrap": boot_results,
        "notes": "달력 기준 연환산(A1)·표준 시작점(A2) 적용판. v1 대비 방법론 동일, 계측만 수정.",
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_benchmark_compare_v2.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")

    lines = [
        "# final0706 vs SPY/QQQ/QLD buy&hold 비교 v2 (수치 정직화판)",
        "",
        f"- 원시 equity 구간: `{result['raw_period']}` — stray 8봉과 결측 갭은 평가에서 제외",
        f"- **권위 구간 = {PRIMARY_START.date()}~** (근거: `final0706_data_gap_diagnosis.md`)",
        "- CAGR은 달력 기준 연환산(v1의 봉 수 기준은 갭 탓에 과대표기 — 공표 16.84%는 정정 대상).",
        "- final0706 equity는 공식 파이프라인(`Yoon1kHandler.build_artifacts`) 값 그대로(재구현 없음).",
        "",
    ]
    for phase in phases:
        lines.append(f"## {phase} ({result['phases'][phase]})")
        lines.append("")
        lines.append("| candidate | CAGR | MDD | Sharpe | Total Return |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for r in rows:
            if r["phase"] != phase:
                continue
            lines.append(
                f"| {r['candidate']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
                f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('total_return'))} |")
        lines.append("")

    lines.append("## 페어드 부트스트랩 (권위 구간, 8000 iters, block=20)")
    lines.append("")
    lines.append("| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, b in boot_results.items():
        lines.append(
            f"| {key} | {b['prob']['sharpe_a_gt_b']*100:.1f}% | "
            f"{b['prob']['cagr_a_gt_b']*100:.1f}% | {b['prob']['mdd_a_shallower']*100:.1f}% |")

    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])
    out_md = ROOT / "reports" / "profile_sizing" / "final0706_benchmark_compare_v2.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
