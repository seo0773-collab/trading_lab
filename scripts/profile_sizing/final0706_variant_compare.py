"""final0706 3종 비교: 원본(50/50 US+KR) vs 미국장 단일 vs final0706_qld.

세 변형 모두 공식 파이프라인 핸들러(Yoon1kHandler / Yoon1kLeverageMixHandler)로
계산한다(재구현 없음). "미국장 단일"은 final0706의 미국 슬리브(yoon1j_lp, VGT
유니버스, top_k=3, hold_buffer=2)만 weight=1.0으로 두고 통화배분/신용게이트를
제거한 즉석 config다(원화 기준 유지, base_currency=KRW로 환산해 원본과 동일
조건에서 비교).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_variant_compare.py
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
from profile_sizing.yoon2a_bootstrap import _stats, block_indices, pct  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from trading_lab.strategies.yoon1k_leverage_mix import Yoon1kLeverageMixHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"


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


def build_us_only_config(final0706_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    us_leg = next(s for s in cfg["sub_strategies"] if s["currency"] == "USD")
    us_leg = copy.deepcopy(us_leg)
    us_leg["weight"] = 1.0
    cfg["sub_strategies"] = [us_leg]
    cfg.pop("currency_allocation", None)
    cfg["benchmark_symbols"] = {"USD": "SPY"}
    cfg["benchmark_label"] = "SPY(원화환산)"
    return cfg


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    final0706_qld_cfg = json.loads((CONFIGS_DIR / "final0706_qld.json").read_text("utf-8"))
    us_only_cfg = build_us_only_config(final0706_cfg)

    k_handler = Yoon1kHandler()
    lev_handler = Yoon1kLeverageMixHandler()

    print("final0706 로드 중...", flush=True)
    raw_orig = k_handler.load_data("PORTFOLIO", final0706_cfg, synthetic=False)
    art_orig = k_handler.build_artifacts(raw_orig, final0706_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_orig = art_orig.equity

    print("final0706 미국장 단일 로드 중...", flush=True)
    raw_us = k_handler.load_data("PORTFOLIO", us_only_cfg, synthetic=False)
    art_us = k_handler.build_artifacts(raw_us, us_only_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_us = art_us.equity

    print("final0706_qld 로드 중...", flush=True)
    raw_qld = lev_handler.load_data("PORTFOLIO", final0706_qld_cfg, synthetic=False)
    art_qld = lev_handler.build_artifacts(raw_qld, final0706_qld_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_qld = art_qld.equity

    navs = {
        "final0706 (원본 50/50)": nav_orig,
        "final0706 미국장 단일": nav_us,
        "final0706_qld (core63.5%+QLD36.5%)": nav_qld,
    }

    idx = nav_orig.index
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    n = len(idx)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    rows = []
    for phase, window in phases.items():
        for label, nav in navs.items():
            m = _clean_metric(perf_nav(nav, window))
            rows.append({"phase": phase, "candidate": label, **m})

    boot = {
        "미국장단일_vs_원본": paired_bootstrap(
            nav_us, nav_orig, label_a="us_only", label_b="orig"),
        "qld_vs_원본": paired_bootstrap(
            nav_qld, nav_orig, label_a="qld", label_b="orig"),
        "qld_vs_미국장단일": paired_bootstrap(
            nav_qld, nav_us, label_a="qld", label_b="us_only"),
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "rows": rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_variant_compare.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 3종 비교: 원본 vs 미국장 단일 vs final0706_qld",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화(KRW) 기준)",
        "- 세 변형 모두 공식 핸들러(`Yoon1kHandler`/`Yoon1kLeverageMixHandler`)로 계산(재구현 없음).",
        "- `final0706 미국장 단일` = final0706의 미국 슬리브(yoon1j_lp, VGT유니버스, top_k=3, hold_buffer=2)만 100% 편입,"
        " 한국 슬리브·원/달러 통화틸트·신용스프레드 게이트 모두 제거(원화환산 유지).",
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
                f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('total_return'))} |"
            )
        lines.append("")

    lines.append("## 페어드 부트스트랩 (8000 iters, block=20)")
    lines.append("")
    lines.append("| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, b in boot.items():
        lines.append(
            f"| {key} | {b['prob']['sharpe_a_gt_b']*100:.1f}% | "
            f"{b['prob']['cagr_a_gt_b']*100:.1f}% | {b['prob']['mdd_a_shallower']*100:.1f}% |"
        )

    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_variant_compare.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
