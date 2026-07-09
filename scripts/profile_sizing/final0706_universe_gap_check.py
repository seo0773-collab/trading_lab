"""final0706 US 유니버스 상장일 비대칭 오염도 점검.

실측: US 11개 섹터 ETF 중 FENY/FIDU/FSTA/FCOM(2013-10-24 상장)·FREL(2015-02-05
상장)는 final0706 백테스트 시작(2007-01-29)보다 한참 늦게 생겼다. 즉 2007~2013
구간은 top_k=3을 "11개 중 3개"가 아니라 "6개(VGT/XLF/IYH/IYC/XLU/XLB) 중 3개"에서
뽑고 있었다 — 선택 경쟁강도가 기간별로 다른데 백테스트 지표는 이를 구분하지 않는다.

이 스크립트는 두 각도로 오염도를 잰다:
  A. 기존 11종목 유니버스(원본, 상장 갭 있음)를 2013-10-24 기준 전/후로 쪼개
     US 슬리브 단독 성과·평균 보유종목수를 비교 — 두 구간의 "선택 난이도"가
     실제로 다른 특성을 보이는지 확인.
  B. "6종목만(전 구간 상장돼 있던 것만)" 유니버스로 통일했을 때 원본(11종목,
     갭 있음) 대비 전체 성과가 얼마나 달라지는지 — 늦게 합류한 5개 ETF가
     헤드라인 수치에 기여한 정도를 역산.

공식 핸들러(ProfilePortfolioHandler/Yoon1kHandler)로 계산, 재구현 없음.
final0706.json은 그대로 두고 즉석 변형으로 비교한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_universe_gap_check.py
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

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
FULL_UNIVERSE = ["VGT", "XLF", "IYH", "FENY", "FIDU", "FSTA", "IYC", "XLU", "XLB", "FREL", "FCOM"]
LONG_HISTORY_UNIVERSE = ["VGT", "XLF", "IYH", "IYC", "XLU", "XLB"]  # 2007-01-29 이전부터 상장
GAP_CUTOFF = "2013-10-24"  # FENY/FIDU/FSTA/FCOM 상장일(FREL은 2015-02-05로 더 늦음)


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    s = nav.reindex(window).ffill().dropna() if window is not None else nav.dropna()
    if len(s) < 60:
        return {}
    eq = s / s.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def paired_bootstrap(
    nav_a: pd.Series, nav_b: pd.Series, *, label_a: str, label_b: str,
    iters: int = 8000, block: int = 20, seed: int = 7,
) -> dict[str, Any]:
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


def us_sleeve_nav_and_holdings(overrides: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    handler = ProfilePortfolioHandler()
    cfg = json.loads((CONFIGS_DIR / "yoon1j_lp.json").read_text("utf-8"))
    cfg.update(overrides)
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    eq = art.equity
    eq = eq[eq.index >= pd.Timestamp("2007-01-29")]
    eq = eq / eq.iloc[0]
    n_hold = art.forecast["n_holdings"].reindex(eq.index)
    return eq, n_hold


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    us_overrides_orig = next(
        s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD"
    )["overrides"]

    print("=== A. 원본(11종목, 상장갭) US 슬리브: 2013-10-24 전/후 비교 ===", flush=True)
    eq_orig, n_hold_orig = us_sleeve_nav_and_holdings(us_overrides_orig)
    cutoff = pd.Timestamp(GAP_CUTOFF)
    pre = eq_orig[eq_orig.index < cutoff]
    post = eq_orig[eq_orig.index >= cutoff]
    pre_perf = _clean_metric(perf_nav(pre))
    post_perf = _clean_metric(perf_nav(post))
    pre_hold = float(n_hold_orig[n_hold_orig.index < cutoff].mean())
    post_hold = float(n_hold_orig[n_hold_orig.index >= cutoff].mean())
    print(f"  pre-2013.10 (6종목 경쟁, {len(pre)}봉): CAGR={_fmt_pct(pre_perf['cagr'])} "
          f"Sharpe={_fmt_num(pre_perf['sharpe'])} MDD={_fmt_pct(pre_perf['max_drawdown'])} "
          f"평균보유={pre_hold:.2f}", flush=True)
    print(f"  post-2013.10 (11종목 경쟁, {len(post)}봉): CAGR={_fmt_pct(post_perf['cagr'])} "
          f"Sharpe={_fmt_num(post_perf['sharpe'])} MDD={_fmt_pct(post_perf['max_drawdown'])} "
          f"평균보유={post_hold:.2f}", flush=True)

    print("\n=== B. 6종목(전구간 상장) vs 11종목(원본, 갭 있음) — US 슬리브 전체기간 ===", flush=True)
    overrides_6 = copy.deepcopy(us_overrides_orig)
    overrides_6["universe"] = LONG_HISTORY_UNIVERSE
    eq_6, n_hold_6 = us_sleeve_nav_and_holdings(overrides_6)
    perf_11 = _clean_metric(perf_nav(eq_orig))
    perf_6 = _clean_metric(perf_nav(eq_6))
    print(f"  11종목(원본): CAGR={_fmt_pct(perf_11['cagr'])} Sharpe={_fmt_num(perf_11['sharpe'])} "
          f"MDD={_fmt_pct(perf_11['max_drawdown'])} 평균보유={n_hold_orig.mean():.2f}", flush=True)
    print(f"  6종목만:      CAGR={_fmt_pct(perf_6['cagr'])} Sharpe={_fmt_num(perf_6['sharpe'])} "
          f"MDD={_fmt_pct(perf_6['max_drawdown'])} 평균보유={n_hold_6.mean():.2f}", flush=True)
    boot_6v11 = paired_bootstrap(eq_6, eq_orig, label_a="6종목", label_b="11종목(원본)")

    print("\n=== C. 전체 파이프라인(Yoon1kHandler): 6종목 US 슬리브로 final0706 재계산 ===", flush=True)
    final0706_6_cfg = copy.deepcopy(final0706_cfg)
    for sub in final0706_6_cfg["sub_strategies"]:
        if sub["currency"] == "USD":
            sub["overrides"]["universe"] = LONG_HISTORY_UNIVERSE
    k_handler = Yoon1kHandler()
    raw_orig = k_handler.load_data("PORTFOLIO", final0706_cfg, synthetic=False)
    art_orig = k_handler.build_artifacts(
        raw_orig, final0706_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    raw_6 = k_handler.load_data("PORTFOLIO", final0706_6_cfg, synthetic=False)
    art_6 = k_handler.build_artifacts(
        raw_6, final0706_6_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_orig, nav_6 = art_orig.equity, art_6.equity
    perf_pipe_orig = _clean_metric(perf_nav(nav_orig))
    perf_pipe_6 = _clean_metric(perf_nav(nav_6))
    print(f"  final0706(원본 11종목): CAGR={_fmt_pct(perf_pipe_orig['cagr'])} "
          f"Sharpe={_fmt_num(perf_pipe_orig['sharpe'])} MDD={_fmt_pct(perf_pipe_orig['max_drawdown'])}",
          flush=True)
    print(f"  final0706(US 6종목만):  CAGR={_fmt_pct(perf_pipe_6['cagr'])} "
          f"Sharpe={_fmt_num(perf_pipe_6['sharpe'])} MDD={_fmt_pct(perf_pipe_6['max_drawdown'])}",
          flush=True)
    boot_pipe = paired_bootstrap(nav_6, nav_orig, label_a="6종목", label_b="11종목(원본)")

    result = {
        "gap_cutoff": GAP_CUTOFF,
        "A_pre_post_split": {
            "pre_2013_10": {"bars": int(len(pre)), "avg_holdings": pre_hold, **pre_perf},
            "post_2013_10": {"bars": int(len(post)), "avg_holdings": post_hold, **post_perf},
        },
        "B_us_sleeve_6_vs_11": {
            "universe_11_original": {"avg_holdings": float(n_hold_orig.mean()), **perf_11},
            "universe_6_consistent": {"avg_holdings": float(n_hold_6.mean()), **perf_6},
            "bootstrap": boot_6v11,
        },
        "C_full_pipeline_6_vs_11": {
            "final0706_original": perf_pipe_orig,
            "final0706_us6_variant": perf_pipe_6,
            "bootstrap": boot_pipe,
        },
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_universe_gap_check.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 US 유니버스 상장일 비대칭 오염도 점검",
        "",
        f"- 실측 확인: FENY/FIDU/FSTA/FCOM 상장일 2013-10-24, FREL 2015-02-05"
        f"(final0706 시작일 2007-01-29보다 6~8년 늦음). VGT/XLF/IYH/IYC/XLU/XLB "
        "6종목만 전체 기간(2007~) 상장돼 있었음.",
        "- 공식 핸들러로 계산(재구현 없음), final0706.json 미수정(즉석 변형 비교).",
        "",
        "## A. US 슬리브(원본 11종목, 상장갭 있음): 2013-10-24 전/후 비교",
        "",
        "| 구간 | 봉수 | CAGR | MDD | Sharpe | 평균 보유종목수 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| pre-2013.10(6종목 경쟁) | {result['A_pre_post_split']['pre_2013_10']['bars']} | "
        f"{_fmt_pct(pre_perf['cagr'])} | {_fmt_pct(pre_perf['max_drawdown'])} | "
        f"{_fmt_num(pre_perf['sharpe'])} | {pre_hold:.2f} |",
        f"| post-2013.10(11종목 경쟁) | {result['A_pre_post_split']['post_2013_10']['bars']} | "
        f"{_fmt_pct(post_perf['cagr'])} | {_fmt_pct(post_perf['max_drawdown'])} | "
        f"{_fmt_num(post_perf['sharpe'])} | {post_hold:.2f} |",
        "",
        "## B. US 슬리브 전체기간: 6종목(전구간 상장) vs 11종목(원본, 갭)",
        "",
        "| universe | CAGR | MDD | Sharpe | 평균 보유종목수 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| 11종목(원본) | {_fmt_pct(perf_11['cagr'])} | {_fmt_pct(perf_11['max_drawdown'])} | "
        f"{_fmt_num(perf_11['sharpe'])} | {n_hold_orig.mean():.2f} |",
        f"| 6종목만 | {_fmt_pct(perf_6['cagr'])} | {_fmt_pct(perf_6['max_drawdown'])} | "
        f"{_fmt_num(perf_6['sharpe'])} | {n_hold_6.mean():.2f} |",
        "",
        f"부트스트랩(6종목 vs 11종목원본, {boot_6v11['period']}): "
        f"P(Sharpe우위) {boot_6v11['prob']['sharpe_a_gt_b']*100:.1f}%, "
        f"P(CAGR우위) {boot_6v11['prob']['cagr_a_gt_b']*100:.1f}%, "
        f"P(MDD더얕음) {boot_6v11['prob']['mdd_a_shallower']*100:.1f}%",
        "",
        "## C. 전체 파이프라인(final0706 구조): US 6종목 변형 vs 원본",
        "",
        "| candidate | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: |",
        f"| final0706(원본 11종목) | {_fmt_pct(perf_pipe_orig['cagr'])} | "
        f"{_fmt_pct(perf_pipe_orig['max_drawdown'])} | {_fmt_num(perf_pipe_orig['sharpe'])} |",
        f"| final0706(US 6종목만) | {_fmt_pct(perf_pipe_6['cagr'])} | "
        f"{_fmt_pct(perf_pipe_6['max_drawdown'])} | {_fmt_num(perf_pipe_6['sharpe'])} |",
        "",
        f"부트스트랩(6종목 vs 11종목원본, {boot_pipe['period']}): "
        f"P(Sharpe우위) {boot_pipe['prob']['sharpe_a_gt_b']*100:.1f}%, "
        f"P(CAGR우위) {boot_pipe['prob']['cagr_a_gt_b']*100:.1f}%, "
        f"P(MDD더얕음) {boot_pipe['prob']['mdd_a_shallower']*100:.1f}%",
    ]
    out_md = ROOT / "reports" / "profile_sizing" / "final0706_universe_gap_check.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
