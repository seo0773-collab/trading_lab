"""final0706 안전자산(TLT/GLD) 프로파일 점수화 스터디.

기존 안전자산 슬리브는 추세 게이트(종가>MA)로 ON/OFF만 판정하고, ON인 자산들
사이는 균등 배분한다(portfolio.py `_target_weights`와 별개 로직). 이 스크립트는
안전자산도 메인 유니버스와 **같은 profile-sizing 엔진**(cycle_multiple percentile
→ buckets → regime_cap = final_target_weight)으로 스코어링해, ON 자산 간 배분을
균등이 아니라 점수 비례로 바꾸는 `safe_sleeve.profile_scored=true` 옵션(opt-in,
기본 off·다른 전략 불변)을 실측 검증한다. 추세 게이트(ON/OFF) 자체는 그대로
유지해 기존 방어 효과(2022류 안전자산 동반 하락 회피)는 보존한 채, "ON인
TLT·GLD 중 어느 쪽이 자기 사이클상 더 유리한 위치인지"만 반영한다.

final0706.json / yoon1j_lp.json / yoon1j_kr.json은 그대로 두고, sub_strategies
overrides에 safe_sleeve(원본 값 + profile_scored=true)만 즉석 주입한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_safe_sleeve_profile_study.py
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


def build_variant(final0706_cfg: dict[str, Any], scored: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    for sub in cfg["sub_strategies"]:
        sub.setdefault("overrides", {})
        if scored:
            sub["overrides"]["safe_sleeve"] = scored_safe_sleeve(sub["config"])
    return cfg


def sleeve_perf(sub_id: str, overrides: dict[str, Any], scored: bool) -> dict[str, Any]:
    handler = ProfilePortfolioHandler()
    cfg = json.loads((CONFIGS_DIR / f"{sub_id}.json").read_text("utf-8"))
    cfg.update(overrides)
    if scored:
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
    for label, scored in [("원본(균등 배분)", False), ("profile_scored(점수 비례)", True)]:
        us_m = sleeve_perf(us_sub["config"], us_sub["overrides"], scored)
        kr_m = sleeve_perf(kr_sub["config"], kr_sub["overrides"], scored)
        sleeve_rows.append({"sleeve": f"US({us_sub['config']})", "variant": label, **us_m})
        sleeve_rows.append({"sleeve": f"KR({kr_sub['config']})", "variant": label, **kr_m})
        print(f"  {label}: US CAGR={_fmt_pct(us_m['cagr'])} Sharpe={_fmt_num(us_m['sharpe'])} "
              f"MDD={_fmt_pct(us_m['max_drawdown'])} | KR CAGR={_fmt_pct(kr_m['cagr'])} "
              f"Sharpe={_fmt_num(kr_m['sharpe'])} MDD={_fmt_pct(kr_m['max_drawdown'])}", flush=True)

    print("전체 파이프라인(Yoon1kHandler) 성과 비교 중 ...", flush=True)
    k_handler = Yoon1kHandler()
    navs: dict[str, pd.Series] = {}
    for label, scored in [("원본(균등 배분)", False), ("profile_scored(점수 비례)", True)]:
        variant_cfg = build_variant(final0706_cfg, scored)
        raw = k_handler.load_data("PORTFOLIO", variant_cfg, synthetic=False)
        art = k_handler.build_artifacts(
            raw, variant_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        navs[label] = art.equity
        print(f"  {label} 완료", flush=True)

    base_label = "원본(균등 배분)"
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
        "profile_scored_vs_원본": paired_bootstrap(
            navs["profile_scored(점수 비례)"], navs[base_label],
            label_a="scored", label_b="원본"),
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "sleeve_performance": sleeve_rows,
        "performance": perf_rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_safe_sleeve_profile_study.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 안전자산(TLT/GLD) 프로파일 점수화 스터디",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화(KRW) 기준)",
        "- 공식 핸들러로 계산(재구현 없음). `final0706.json`/`yoon1j_lp.json`/`yoon1j_kr.json`은 "
        "그대로 두고 sub_strategies overrides에 `safe_sleeve.profile_scored=true`만 즉석 주입.",
        "- 추세 게이트(종가>MA100, ON/OFF)는 원본과 동일하게 유지 — 바뀌는 것은 ON인 "
        "TLT·GLD 사이의 배분 비율뿐(균등 → 각자 profile-sizing 점수 비례).",
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

    lines.append("## 3. 페어드 블록 부트스트랩 (profile_scored vs 원본, iters=8000, block=20)")
    lines.append("")
    lines.append("| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for label, b in boot.items():
        p = b["prob"]
        lines.append(
            f"| {label} | {b['period']} | {p['sharpe_a_gt_b']*100:.1f}% | "
            f"{p['cagr_a_gt_b']*100:.1f}% | {p['mdd_a_shallower']*100:.1f}% |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_safe_sleeve_profile_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
