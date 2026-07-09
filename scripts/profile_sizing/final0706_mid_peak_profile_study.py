"""final0706 '프로파일 중간지점 고점수' 가설 스터디.

기존 buckets 모델([0,20)=0.8 ~ [80,100)=0.1)은 percentile(자기 사이클 내 누적
위치, 0=역대급 저가권~1=역대급 고가권)이 낮을수록(=쌀수록) 점수가 높은 순수
컨트래리언(monotonic) 곡선이다. 이 스크립트는 "프로파일 중간(40~60%)이 제일
높고 양끝(극단적 저가권·고가권)으로 흩어질수록 낮아지는" 텐트형(mid-peak) 곡선
가설을 검증한다 — 원본 곡선이 쓰던 값(0.8 최대, 0.5 중간값, 0.1 최소)을 그대로
재사용해 대칭 텐트로 재배치했을 뿐 새 파라미터는 도입하지 않았다.

buckets는 이미 config JSON에서 완전히 자유롭게 정의 가능한 구조라(scripts/
profile_sizing/sizing.py의 base_target_weight, config.py의 buckets 필드) 코드
수정 없이 sub_strategies overrides에 buckets만 주입해 실험한다.
final0706.json 자체(동결 스냅샷)와 yoon1j_lp/kr.json(다른 전략이 참조하는 활성
config)은 그대로 두고, 즉석 변형 config로만 비교한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_mid_peak_profile_study.py
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

ORIGINAL_BUCKETS = [[0, 20, 0.8], [20, 40, 0.6], [40, 60, 0.5], [60, 80, 0.3], [80, 100, 0.1]]
# 텐트형: 원본과 같은 값(0.8/0.5/0.1)을 재사용, 중간(40~60)에 최댓값 배치.
MID_PEAK_BUCKETS = [[0, 20, 0.1], [20, 40, 0.5], [40, 60, 0.8], [60, 80, 0.5], [80, 100, 0.1]]

VARIANTS = {
    "원본(저가권 우위)": ORIGINAL_BUCKETS,
    "mid_peak(중간 우위)": MID_PEAK_BUCKETS,
}


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
        "label_a": label_a, "label_b": label_b,
        "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": len(idx),
        "prob": {
            "sharpe_a_gt_b": float((sa > sb).mean()),
            "cagr_a_gt_b": float((ca > cb).mean()),
            "mdd_a_shallower": float((ma > mb).mean()),
        },
    }


def build_variant(final0706_cfg: dict[str, Any], buckets: list) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    for sub in cfg["sub_strategies"]:
        sub.setdefault("overrides", {})
        sub["overrides"]["buckets"] = buckets
    return cfg


def sleeve_perf(sub_id: str, overrides: dict[str, Any], buckets: list) -> dict[str, Any]:
    handler = ProfilePortfolioHandler()
    cfg = json.loads((CONFIGS_DIR / f"{sub_id}.json").read_text("utf-8"))
    cfg.update(overrides)
    cfg["buckets"] = buckets
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    return _clean_metric(art.metrics)


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    kr_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "KRW")

    print("슬리브 단독(US/KR) 성과 비교 중 ...", flush=True)
    sleeve_rows = []
    for label, buckets in VARIANTS.items():
        us_m = sleeve_perf(us_sub["config"], us_sub["overrides"], buckets)
        kr_m = sleeve_perf(kr_sub["config"], kr_sub["overrides"], buckets)
        sleeve_rows.append({"sleeve": f"US({us_sub['config']})", "variant": label, **us_m})
        sleeve_rows.append({"sleeve": f"KR({kr_sub['config']})", "variant": label, **kr_m})
        print(f"  {label}: US CAGR={_fmt_pct(us_m['cagr'])} Sharpe={_fmt_num(us_m['sharpe'])} "
              f"MDD={_fmt_pct(us_m['max_drawdown'])} | KR CAGR={_fmt_pct(kr_m['cagr'])} "
              f"Sharpe={_fmt_num(kr_m['sharpe'])} MDD={_fmt_pct(kr_m['max_drawdown'])}", flush=True)

    print("전체 파이프라인(Yoon1kHandler) 성과 비교 중 ...", flush=True)
    k_handler = Yoon1kHandler()
    navs: dict[str, pd.Series] = {}
    for label, buckets in VARIANTS.items():
        variant_cfg = build_variant(final0706_cfg, buckets)
        raw = k_handler.load_data("PORTFOLIO", variant_cfg, synthetic=False)
        art = k_handler.build_artifacts(
            raw, variant_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        navs[label] = art.equity
        print(f"  {label} 완료", flush=True)

    base_label = "원본(저가권 우위)"
    idx = navs[base_label].index
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    n = len(idx)
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    perf_rows = []
    for phase, window in phases.items():
        for label, nav in navs.items():
            m = _clean_metric(perf_nav(nav, window))
            perf_rows.append({"phase": phase, "candidate": label, **m})

    boot = {
        "mid_peak_vs_원본": paired_bootstrap(
            navs["mid_peak(중간 우위)"], navs[base_label],
            label_a="mid_peak", label_b="원본"),
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "sleeve_performance": sleeve_rows,
        "performance": perf_rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_mid_peak_profile_study.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 '프로파일 중간지점 고점수(mid-peak)' 가설 스터디",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화(KRW) 기준)",
        "- 공식 핸들러로 계산(재구현 없음). `final0706.json`/`yoon1j_lp.json`/`yoon1j_kr.json`은 "
        "그대로 두고 sub_strategies overrides에 `buckets`만 즉석 주입.",
        "- 원본 buckets(percentile 0=역대급 저가권 ~ 100=역대급 고가권): "
        f"`{ORIGINAL_BUCKETS}` (단조감소, 쌀수록 고점수).",
        f"- mid_peak buckets: `{MID_PEAK_BUCKETS}` (원본과 같은 값(0.8/0.5/0.1)을 "
        "40~60%(사이클 중간)에 최댓값이 오도록 대칭 재배치, 양끝은 최저점).",
        "",
        "## 1. 슬리브 단독 성과 (US yoon1j_lp / KR yoon1j_kr, top_k=3)",
        "",
        "| sleeve | variant | CAGR | MDD | Sharpe | Total Return |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in sleeve_rows:
        lines.append(
            f"| {r['sleeve']} | {r['variant']} | {_fmt_pct(r.get('cagr'))} | "
            f"{_fmt_pct(r.get('max_drawdown'))} | {_fmt_num(r.get('sharpe'))} | "
            f"{_fmt_pct(r.get('total_return'))} |"
        )
    lines += [
        "",
        "## 2. 전체 파이프라인(final0706 구조) 성과 비교",
        "",
    ]
    for phase in phases:
        lines.append(f"### {phase} ({result['phases'][phase]})")
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

    lines.append("## 3. 페어드 블록 부트스트랩 (mid_peak vs 원본, iters=8000, block=20)")
    lines.append("")
    lines.append("| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for label, b in boot.items():
        p = b["prob"]
        lines.append(
            f"| {label} | {b['period']} | {p['sharpe_a_gt_b']*100:.1f}% | "
            f"{p['cagr_a_gt_b']*100:.1f}% | {p['mdd_a_shallower']*100:.1f}% |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_mid_peak_profile_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
