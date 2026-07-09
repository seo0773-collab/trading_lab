"""final0706 섹터 대장주 치환 스터디: ETF 유니버스 vs 시기별 개별종목 대장주.

미국 슬리브(yoon1j_lp)의 11개 섹터 ETF(VGT/XLF/IYH/FENY/FIDU/FSTA/IYC/XLU/XLB/
FREL/FCOM)를, 각 섹터를 대표하는 "시기별 시가총액 1위 개별종목"으로 치환한
`final0706_sector_leader` 파생을 원본과 비교한다. 리더십 시대는 웹검색으로 일부
크로스오버 날짜를 확인하고 나머지는 근사(사용자 확인 후 진행) — 정확도의 상한은
그 근사치에 달려있음을 유의. 구현은 opt-in `era_gate`(compute_universe): 섹터당
여러 종목을 유니버스에 나란히 두고 각자 재임구간 밖에서는 점수 0.

KR 슬리브는 원본과 동일(질문이 미국 ETF 대상이라 미국 슬리브에만 적용).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_sector_leader_study.py
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


def us_leg_overrides(final_cfg: dict[str, Any]) -> dict[str, Any]:
    return next(s for s in final_cfg["sub_strategies"] if s["currency"] == "USD")["overrides"]


def sleeve_perf(sub_id: str, overrides: dict[str, Any], *, window_start: str) -> dict[str, Any]:
    """단독 슬리브 성과. 개별종목은 ETF보다 상장이 훨씬 길어(MSFT 1986~/XOM 1962~)
    master index가 그만큼 앞으로 늘어난다 — window_start로 final0706 실제 시작일
    (2007-01-29, 원/달러 KRW=X 등 다른 컴포넌트가 정하는 공통 시작)에 맞춰 잘라야
    ETF 원본과 공정 비교가 된다(안 자르면 1986년 블랙먼데이까지 포함돼 total_return이
    허수로 폭증)."""
    handler = ProfilePortfolioHandler()
    cfg = json.loads((CONFIGS_DIR / f"{sub_id}.json").read_text("utf-8"))
    cfg.update(overrides)
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    eq = art.equity
    eq = eq[eq.index >= pd.Timestamp(window_start)]
    eq = eq / eq.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    m = _clean_metric(performance(eq, ret, "1d"))
    n_hold = art.forecast["n_holdings"].reindex(eq.index) if "n_holdings" in art.forecast else None
    m["avg_holdings"] = float(n_hold.mean()) if n_hold is not None else None
    return m


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    sector_leader_cfg = json.loads((CONFIGS_DIR / "final0706_sector_leader.json").read_text("utf-8"))

    print("US 슬리브 단독(ETF vs 대장주) 성과 비교 중 ...", flush=True)
    us_overrides_orig = us_leg_overrides(final0706_cfg)
    us_overrides_new = us_leg_overrides(sector_leader_cfg)
    sleeve_rows = []
    for label, overrides in [("ETF(원본)", us_overrides_orig), ("섹터 대장주", us_overrides_new)]:
        m = sleeve_perf("yoon1j_lp", overrides, window_start="2007-01-29")
        sleeve_rows.append({"variant": label, **m})
        hold_str = f"{m['avg_holdings']:.2f}" if m.get("avg_holdings") is not None else "-"
        print(f"  {label}: CAGR={_fmt_pct(m['cagr'])} MDD={_fmt_pct(m['max_drawdown'])} "
              f"Sharpe={_fmt_num(m['sharpe'])} 평균보유={hold_str}", flush=True)

    print("전체 파이프라인(Yoon1kHandler) 성과 비교 중 ...", flush=True)
    k_handler = Yoon1kHandler()
    navs: dict[str, pd.Series] = {}
    for label, cfg in [("ETF(원본)", final0706_cfg), ("섹터 대장주", sector_leader_cfg)]:
        raw = k_handler.load_data("PORTFOLIO", cfg, synthetic=False)
        art = k_handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        navs[label] = art.equity
        print(f"  {label} 완료", flush=True)

    base_label = "ETF(원본)"
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
        "섹터대장주_vs_원본": paired_bootstrap(
            navs["섹터 대장주"], navs[base_label], label_a="sector_leader", label_b="원본"),
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "sleeve_performance": sleeve_rows,
        "performance": perf_rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_sector_leader_study.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 섹터 대장주 치환 스터디: ETF vs 시기별 개별종목 1위",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화 기준)",
        "- 공식 핸들러로 계산(재구현 없음). US 슬리브(yoon1j_lp)의 11개 섹터 ETF를 "
        "시기별 대장주 개별종목으로 치환(`configs/strategies/final0706_sector_leader.json`), "
        "KR 슬리브·통화틸트·신용게이트는 원본과 동일.",
        "- 리더십 시대는 웹검색+근사(사용자 확인 후 진행) — 정확도 한계 있음(config _comment 참조).",
        "",
        "## 1. US 슬리브 단독 성과 (yoon1j_lp, top_k=3)",
        "",
        "| variant | CAGR | MDD | Sharpe | Total Return | 평균 보유종목수 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sleeve_rows:
        hold = r.get("avg_holdings")
        lines.append(
            f"| {r['variant']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
            f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('total_return'))} | "
            f"{'-' if hold is None else f'{hold:.2f}'} |"
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

    lines.append("## 3. 페어드 블록 부트스트랩 (섹터대장주 vs 원본, iters=8000, block=20)")
    lines.append("")
    lines.append("| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for label, b in boot.items():
        p = b["prob"]
        lines.append(
            f"| {label} | {b['period']} | {p['sharpe_a_gt_b']*100:.1f}% | "
            f"{p['cagr_a_gt_b']*100:.1f}% | {p['mdd_a_shallower']*100:.1f}% |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_sector_leader_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
