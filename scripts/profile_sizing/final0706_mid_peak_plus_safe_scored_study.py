"""final0706 조합 스터디: mid_peak buckets + 안전자산 profile_scored 동시 적용.

두 개의 독립 실험을 결합한다:
  1. mid_peak buckets (percentile 40~60%에 최댓값, 이전 스터디에서 주식 유니버스
     단독 적용 시 기각됨 — final0706_mid_peak_profile_study.py)
  2. safe_sleeve.profile_scored=true (안전자산도 같은 profile-sizing 엔진으로
     스코어링해 ON 자산 간 배분을 점수 비례로, 이전 스터디에서 소폭 개선
     — final0706_safe_sleeve_profile_study.py)

주의: 안전자산 스코어링(profile_scored)은 메인 유니버스와 **같은 cfg**(따라서
같은 buckets)를 공유하는 구조라, "buckets를 mid_peak으로 바꾼다"는 결정은
주식 유니버스뿐 아니라 안전자산(TLT/GLD 등) 스코어링에도 함께 적용된다 —
별도 buckets 슬롯은 없다(코드 수정 없이 이 결합을 그대로 실측).

4개 변형을 모두 비교한다: 원본 / mid_peak만 / profile_scored만 / 결합.
final0706.json·yoon1j_lp.json·yoon1j_kr.json은 그대로 두고 overrides로만 실험.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_mid_peak_plus_safe_scored_study.py
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
MID_PEAK_BUCKETS = [[0, 20, 0.1], [20, 40, 0.5], [40, 60, 0.8], [60, 80, 0.5], [80, 100, 0.1]]

VARIANTS = {
    "원본": {"mid_peak": False, "safe_scored": False},
    "mid_peak만": {"mid_peak": True, "safe_scored": False},
    "safe_scored만": {"mid_peak": False, "safe_scored": True},
    "결합(mid_peak+safe_scored)": {"mid_peak": True, "safe_scored": True},
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


def scored_safe_sleeve(sub_id: str) -> dict[str, Any]:
    base_cfg = json.loads((CONFIGS_DIR / f"{sub_id}.json").read_text("utf-8"))
    ss = copy.deepcopy(base_cfg.get("safe_sleeve") or {})
    ss["profile_scored"] = True
    return ss


def build_variant(final0706_cfg: dict[str, Any], mid_peak: bool, safe_scored: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    for sub in cfg["sub_strategies"]:
        sub.setdefault("overrides", {})
        if mid_peak:
            sub["overrides"]["buckets"] = MID_PEAK_BUCKETS
        if safe_scored:
            sub["overrides"]["safe_sleeve"] = scored_safe_sleeve(sub["config"])
    return cfg


def sleeve_perf(sub_id: str, overrides: dict[str, Any], mid_peak: bool, safe_scored: bool) -> dict[str, Any]:
    handler = ProfilePortfolioHandler()
    cfg = json.loads((CONFIGS_DIR / f"{sub_id}.json").read_text("utf-8"))
    cfg.update(overrides)
    if mid_peak:
        cfg["buckets"] = MID_PEAK_BUCKETS
    if safe_scored:
        cfg["safe_sleeve"] = scored_safe_sleeve(sub_id)
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    m = _clean_metric(art.metrics)
    m["avg_safe_exposure"] = art.metadata.get("avg_safe_exposure")
    return m


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    kr_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "KRW")

    print("슬리브 단독(US/KR) 성과 비교 중 ...", flush=True)
    sleeve_rows = []
    for label, flags in VARIANTS.items():
        us_m = sleeve_perf(us_sub["config"], us_sub["overrides"], **flags)
        kr_m = sleeve_perf(kr_sub["config"], kr_sub["overrides"], **flags)
        sleeve_rows.append({"sleeve": f"US({us_sub['config']})", "variant": label, **us_m})
        sleeve_rows.append({"sleeve": f"KR({kr_sub['config']})", "variant": label, **kr_m})
        print(f"  {label}: US CAGR={_fmt_pct(us_m['cagr'])} Sharpe={_fmt_num(us_m['sharpe'])} "
              f"MDD={_fmt_pct(us_m['max_drawdown'])} | KR CAGR={_fmt_pct(kr_m['cagr'])} "
              f"Sharpe={_fmt_num(kr_m['sharpe'])} MDD={_fmt_pct(kr_m['max_drawdown'])}", flush=True)

    print("전체 파이프라인(Yoon1kHandler) 성과 비교 중 ...", flush=True)
    k_handler = Yoon1kHandler()
    navs: dict[str, pd.Series] = {}
    for label, flags in VARIANTS.items():
        variant_cfg = build_variant(final0706_cfg, **flags)
        raw = k_handler.load_data("PORTFOLIO", variant_cfg, synthetic=False)
        art = k_handler.build_artifacts(
            raw, variant_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        navs[label] = art.equity
        print(f"  {label} 완료", flush=True)

    base_label = "원본"
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

    combo_label = "결합(mid_peak+safe_scored)"
    boot = {
        f"{v}_vs_원본": paired_bootstrap(navs[v], navs[base_label], label_a=v, label_b="원본")
        for v in VARIANTS if v != base_label
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "sleeve_performance": sleeve_rows,
        "performance": perf_rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_mid_peak_plus_safe_scored_study.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 조합 스터디: mid_peak buckets + 안전자산 profile_scored",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화(KRW) 기준)",
        "- 공식 핸들러로 계산(재구현 없음). `final0706.json`/`yoon1j_lp.json`/`yoon1j_kr.json`은 "
        "그대로 두고 overrides로만 실험.",
        "- **주의**: `safe_sleeve.profile_scored`는 메인 유니버스와 같은 `cfg`(=같은 buckets)를 "
        "공유하므로, mid_peak을 켜면 주식 유니버스뿐 아니라 안전자산(TLT/GLD 등) 스코어링에도 "
        "함께 적용된다(별도 buckets 슬롯 없음).",
        f"- 원본 buckets: `{ORIGINAL_BUCKETS}` / mid_peak buckets: `{MID_PEAK_BUCKETS}`",
        "",
        "## 1. 슬리브 단독 성과 (US yoon1j_lp / KR yoon1j_kr)",
        "",
        "| sleeve | variant | CAGR | MDD | Sharpe | Total Return | 평균 안전자산 비중 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sleeve_rows:
        lines.append(
            f"| {r['sleeve']} | {r['variant']} | {_fmt_pct(r.get('cagr'))} | "
            f"{_fmt_pct(r.get('max_drawdown'))} | {_fmt_num(r.get('sharpe'))} | "
            f"{_fmt_pct(r.get('total_return'))} | {_fmt_pct(r.get('avg_safe_exposure'))} |"
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

    lines.append("## 3. 페어드 블록 부트스트랩 (각 변형 vs 원본, iters=8000, block=20)")
    lines.append("")
    lines.append("| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for label, b in boot.items():
        p = b["prob"]
        lines.append(
            f"| {label} | {b['period']} | {p['sharpe_a_gt_b']*100:.1f}% | "
            f"{p['cagr_a_gt_b']*100:.1f}% | {p['mdd_a_shallower']*100:.1f}% |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_mid_peak_plus_safe_scored_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
