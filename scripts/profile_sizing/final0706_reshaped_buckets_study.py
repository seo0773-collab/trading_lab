"""final0706 buckets 재설계 — train+val 피팅 → holdout(test) 검증.

이전 진단(final0706_low_cycle_alpha_check.py)에서 원신호가 단조감소가 아니라
"최저가권 1등, 중간(40-60%) 최악, 최고가권은 오히려 회복"하는 U자형임을 확인했다.
이번엔 그 발견을 **train+validation(<2022-12-19, 과거) 구간에서만** 이용해 새
buckets 곡선을 피팅하고, **holdout(test, 2022-12-19~2026-07-08)에서** 원본
buckets([0.8,0.6,0.5,0.3,0.1], 단조감소)와 정면으로 비교한다 — 이 랩의 다른
검증들과 같은 규율("val 선정 → test 확인")로, 관찰한 데이터에 곡선을 맞추는
과적합을 피한다.

곡선 산출: fwd63(3개월) 순방향수익률을 fit window(전 종목 풀링)에서 분위별로
집계 → min-max로 [0.1, 0.8] 구간에 정규화(원본과 같은 최소/최대 범위, 모양만
데이터 기반으로 교체). regime_cap 등 다른 로직은 전혀 안 건드림.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_reshaped_buckets_study.py
"""
from __future__ import annotations

import copy
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

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.cycle_state import forward_returns, newey_west_mean_test  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.run import run_pipeline  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
UNIVERSE = ["VGT", "XLF", "IYH", "FENY", "FIDU", "FSTA", "IYC", "XLU", "XLB", "FREL", "FCOM"]
BUCKET_EDGES = [0, 20, 40, 60, 80, 100]
BUCKET_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
ORIGINAL_BUCKETS = [[0, 20, 0.8], [20, 40, 0.6], [40, 60, 0.5], [60, 80, 0.3], [80, 100, 0.1]]
FIT_HORIZON = 63
TRAIN_VAL_END = "2022-12-19"  # 미포함(exclusive) — 이 날짜부터 test(holdout)


def _fmt_pct(v):
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v):
    return "-" if v is None else f"{v:.3f}"


def _clean_metric(p):
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def perf_nav(nav, window=None):
    s = nav.reindex(window).ffill().dropna() if window is not None else nav.dropna()
    if len(s) < 60:
        return {}
    eq = s / s.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def paired_bootstrap(nav_a, nav_b, *, label_a, label_b, iters=8000, block=20, seed=7):
    idx = nav_a.dropna().index.intersection(nav_b.dropna().index).sort_values()
    a = nav_a.reindex(idx).ffill()
    b = nav_b.reindex(idx).ffill()
    ra = a.pct_change().fillna(0.0).to_numpy()
    rb = b.pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    bidx = block_indices(len(idx), block, iters, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    return {
        "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": len(idx),
        "prob": {
            "sharpe_a_gt_b": float((sa > sb).mean()),
            "cagr_a_gt_b": float((ca > cb).mean()),
            "mdd_a_shallower": float((ma > mb).mean()),
        },
    }


def fit_new_buckets() -> tuple[list, pd.Series]:
    """train+val(<TRAIN_VAL_END) fwd63 분위 평균으로 새 buckets 곡선을 피팅."""
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    us_overrides = next(
        s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD"
    )["overrides"]
    cfg_dict = json.loads((CONFIGS_DIR / "yoon1j_lp.json").read_text("utf-8"))
    cfg_dict.update(us_overrides)
    cfg = config_from_dict(cfg_dict)

    print("피팅용 데이터 로딩 중 (train+val, <2022-12-19)...", flush=True)
    fwd_col = f"fwd_{FIT_HORIZON}"
    bucket_means = {b: [] for b in range(5)}
    for sym in UNIVERSE:
        raw = load_yfinance(sym, cfg.interval, cfg.period)
        out = run_pipeline(raw, cfg)
        fc = out["forecast"]
        idx = fc.index
        pct = pd.Series(fc["cumulative_percentile"].to_numpy(), index=idx) * 100.0
        close = pd.Series(fc["close"].to_numpy(), index=idx)
        fwd = forward_returns(close, (FIT_HORIZON,))
        df = fwd.copy()
        df["pct"] = pct
        df["bucket"] = pd.cut(df["pct"], BUCKET_EDGES, labels=range(5), include_lowest=True)
        df = df[(df.index >= pd.Timestamp("2007-01-29")) & (df.index < pd.Timestamp(TRAIN_VAL_END))]
        for b in range(5):
            vals = df.loc[df["bucket"] == b, fwd_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            bucket_means[b].append(vals)

    fit_means = pd.Series({
        b: float(np.concatenate(bucket_means[b]).mean()) for b in range(5)
    })
    print("피팅 구간 분위별 평균 fwd63:", fit_means.round(4).to_dict(), flush=True)

    # min-max → [0.1, 0.8] (원본과 동일 범위), 모양만 데이터 기반
    lo, hi = fit_means.min(), fit_means.max()
    scaled = 0.1 + (fit_means - lo) / (hi - lo) * (0.8 - 0.1)
    new_buckets = [
        [BUCKET_EDGES[i], BUCKET_EDGES[i + 1], round(float(scaled[i]), 3)]
        for i in range(5)
    ]
    print("새 buckets(train+val 피팅):", new_buckets, flush=True)
    return new_buckets, fit_means


def build_variant(final0706_cfg, buckets):
    cfg = copy.deepcopy(final0706_cfg)
    for sub in cfg["sub_strategies"]:
        if sub["currency"] == "USD":
            sub["overrides"]["buckets"] = buckets
    return cfg


def main() -> None:
    new_buckets, fit_means = fit_new_buckets()

    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    variant_cfg = build_variant(final0706_cfg, new_buckets)

    print("\n전체 파이프라인 백테스트 중 ...", flush=True)
    k_handler = Yoon1kHandler()
    navs = {}
    for label, cfg in [("원본(단조감소)", final0706_cfg), ("재설계(train+val 피팅)", variant_cfg)]:
        raw = k_handler.load_data("PORTFOLIO", cfg, synthetic=False)
        art = k_handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        navs[label] = art.equity
        print(f"  {label} 완료", flush=True)

    base_label = "원본(단조감소)"
    idx = navs[base_label].index
    n = len(idx)
    t_end = int(n * 0.6)
    v_end = int(n * 0.8)
    phases = {
        "train+val(피팅 구간, in-sample)": idx[:v_end],
        "test(holdout, 진짜 검증)": idx[v_end:],
        "all(참고용)": idx,
    }

    perf_rows = []
    for phase, window in phases.items():
        for label, nav in navs.items():
            m = _clean_metric(perf_nav(nav, window))
            perf_rows.append({"phase": phase, "candidate": label, **m})

    # holdout(test)에서만 부트스트랩(진짜 검증)
    test_window = phases["test(holdout, 진짜 검증)"]
    nav_new_test = navs["재설계(train+val 피팅)"].reindex(test_window).ffill()
    nav_orig_test = navs[base_label].reindex(test_window).ffill()
    boot_test = paired_bootstrap(nav_new_test, nav_orig_test, label_a="재설계", label_b="원본")

    result = {
        "fit_horizon": FIT_HORIZON,
        "fit_window_bucket_means": fit_means.to_dict(),
        "new_buckets": new_buckets,
        "original_buckets": ORIGINAL_BUCKETS,
        "performance": perf_rows,
        "bootstrap_test_holdout": boot_test,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_reshaped_buckets_study.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 buckets 재설계 — train+val 피팅 → holdout(test) 검증",
        "",
        f"- 원본 buckets: `{ORIGINAL_BUCKETS}` (단조감소)",
        f"- 재설계 buckets(train+val, <{TRAIN_VAL_END}, fwd{FIT_HORIZON} 분위평균 min-max "
        f"[0.1,0.8] 정규화): `{new_buckets}`",
        f"- 피팅 구간(train+val) 분위별 평균 fwd{FIT_HORIZON}: "
        f"{ {BUCKET_LABELS[b]: f'{fit_means[b]*100:.2f}%' for b in range(5)} }",
        "- **test(holdout, 2022-12-19~2026-07-08)는 피팅에 전혀 쓰이지 않은 구간** — "
        "여기서 원본과 붙여야 진짜 검증.",
        "",
        "## 성과 비교",
        "",
    ]
    for phase in phases:
        lines.append(f"### {phase}")
        lines.append("")
        lines.append("| candidate | CAGR | MDD | Sharpe | Total Return |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for r in perf_rows:
            if r["phase"] != phase:
                continue
            lines.append(
                f"| {r['candidate']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
                f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('total_return'))} |"
            )
        lines.append("")

    p = boot_test["prob"]
    lines += [
        "## Holdout(test) 페어드 블록 부트스트랩 (재설계 vs 원본, iters=8000, block=20)",
        "",
        f"기간: {boot_test['period']} ({boot_test['bars']}봉)",
        "",
        "| P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |",
        "| ---: | ---: | ---: |",
        f"| {p['sharpe_a_gt_b']*100:.1f}% | {p['cagr_a_gt_b']*100:.1f}% | {p['mdd_a_shallower']*100:.1f}% |",
    ]

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_reshaped_buckets_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
