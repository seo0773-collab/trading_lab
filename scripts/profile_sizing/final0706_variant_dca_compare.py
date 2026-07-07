"""final0706 3종(원본/미국장 단일/final0706_qld) 적립식(DCA) 총수익률 비교.

기존 `dca_lumpsum_eval.py`와 동일한 근사(정수주·수수료 미반영, 이상적 월적립)를
`final0706_variant_compare.py`가 계산하는 3종 NAV에 그대로 적용한다. 초기
2,000,000원 + 매월 첫 거래일 200,000원 적립, 각 phase 구간 시작 시점부터 새로
적립을 시작한다고 가정(구간별 독립 계좌).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_variant_dca_compare.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from trading_lab.strategies.yoon1k_leverage_mix import Yoon1kLeverageMixHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
INITIAL = 2_000_000.0
MONTHLY = 200_000.0


def build_us_only_config(final0706_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    us_leg = copy.deepcopy(next(s for s in cfg["sub_strategies"] if s["currency"] == "USD"))
    us_leg["weight"] = 1.0
    cfg["sub_strategies"] = [us_leg]
    cfg.pop("currency_allocation", None)
    cfg["benchmark_symbols"] = {"USD": "SPY"}
    cfg["benchmark_label"] = "SPY(원화환산)"
    return cfg


def dca_path(nav: pd.Series) -> tuple[pd.Series, pd.Series]:
    """월초 적립식 계좌가치·누적원금 경로 (dca_lumpsum_eval.py와 동일 로직)."""
    units = INITIAL / nav.iloc[0]
    principal = INITIAL
    firsts = set(nav.groupby([nav.index.year, nav.index.month]).head(1).index)
    values, principals = [], []
    for t, price in nav.items():
        if t in firsts and t != nav.index[0]:
            units += MONTHLY / price
            principal += MONTHLY
        values.append(units * price)
        principals.append(principal)
    return pd.Series(values, index=nav.index), pd.Series(principals, index=nav.index)


def max_drawdown(series: pd.Series) -> float:
    return float((series / series.cummax() - 1.0).min())


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    final0706_qld_cfg = json.loads((CONFIGS_DIR / "final0706_qld.json").read_text("utf-8"))
    us_only_cfg = build_us_only_config(final0706_cfg)

    k_handler = Yoon1kHandler()
    lev_handler = Yoon1kLeverageMixHandler()

    print("final0706 로드 중...", flush=True)
    raw_orig = k_handler.load_data("PORTFOLIO", final0706_cfg, synthetic=False)
    nav_orig = k_handler.build_artifacts(
        raw_orig, final0706_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252).equity

    print("final0706 미국장 단일 로드 중...", flush=True)
    raw_us = k_handler.load_data("PORTFOLIO", us_only_cfg, synthetic=False)
    nav_us = k_handler.build_artifacts(
        raw_us, us_only_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252).equity

    print("final0706_qld 로드 중...", flush=True)
    raw_qld = lev_handler.load_data("PORTFOLIO", final0706_qld_cfg, synthetic=False)
    nav_qld = lev_handler.build_artifacts(
        raw_qld, final0706_qld_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252).equity

    navs = {
        "final0706 (원본 50/50)": nav_orig,
        "final0706 미국장 단일": nav_us,
        "final0706_qld (core63.5%+QLD36.5%)": nav_qld,
    }

    idx = nav_orig.index
    n = len(idx)
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    rows = []
    for phase, window in phases.items():
        for label, nav in navs.items():
            s = nav.reindex(window).ffill().dropna()
            if len(s) < 60:
                continue
            s = s / s.iloc[0]
            acct, principal = dca_path(s)
            lump_ret = float(s.iloc[-1] - 1.0)
            dca_ret = float(acct.iloc[-1] / principal.iloc[-1] - 1.0)
            rows.append({
                "phase": phase,
                "candidate": label,
                "months": round((window[-1] - window[0]).days / 30.44, 1),
                "total_contributed": float(principal.iloc[-1]),
                "lump_total_return": lump_ret,
                "dca_final_value": float(acct.iloc[-1]),
                "dca_total_return": dca_ret,
                "dca_account_mdd": max_drawdown(acct),
                "nav_mdd": max_drawdown(s),
            })

    out_json = ROOT / "reports" / "profile_sizing" / "final0706_variant_dca_compare.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    def won(v: float) -> str:
        return f"{v:,.0f}원"

    lines = [
        "# final0706 3종 적립식(DCA) 총수익률 비교",
        "",
        "- 가정: 초기 2,000,000원 + 매월 첫 거래일 200,000원 적립(정수주·수수료 미반영 이상적 근사, `dca_lumpsum_eval.py`와 동일 로직).",
        "- 각 phase는 그 구간 시작 시점부터 새로 적립을 시작한다고 가정한 독립 계좌.",
        "- `lump_total_return` = 동일 구간 일시투입(거치식) 총수익률 참고용.",
        "",
    ]
    for phase in phases:
        prows = [r for r in rows if r["phase"] == phase]
        if not prows:
            continue
        lines.append(f"## {phase} ({idx[0].date() if phase=='all' else phases[phase][0].date()}~{phases[phase][-1].date()}, 약 {prows[0]['months']}개월)")
        lines.append("")
        lines.append("| candidate | 총 납입원금 | 적립식 평가액 | 적립식 총수익률 | 적립계좌 MDD | (참고)거치식 총수익률 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in prows:
            lines.append(
                f"| {r['candidate']} | {won(r['total_contributed'])} | {won(r['dca_final_value'])} | "
                f"{pct(r['dca_total_return'])} | {pct(r['dca_account_mdd'])} | {pct(r['lump_total_return'])} |"
            )
        lines.append("")

    lines.extend(["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_variant_dca_compare.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
