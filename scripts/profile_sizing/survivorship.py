#!/usr/bin/env python
"""생존편향(survivorship bias) 점검 — profile-portfolio-v1.

현재 기본 유니버스(끝까지 살아남은 대형 우량주 30)는 결과를 낙관적으로 만들 수
있다. 닷컴·금융위기·원자재·리테일에서 크게 부진했던(그러나 데이터가 남은) 종목을
섞은 **확장 유니버스**로 재평가해, 성과가 (a) 종목 선별 덕인지 (b) 전략 덕인지 가른다.

가설: top-K 선택 전략은 점수 낮은 부진 종목을 회피하므로, 부진 종목을 다 떠안는
equal-weight buy & hold 대비 **넓은 유니버스에서 오히려 우위가 커진다**.

상장폐지로 데이터가 없는 종목(예: WBA)은 yfinance에서 빠지므로 완전한
survivorship-free는 아니다(데이터 한계). 그래도 '부진 종목 포함' 민감도는 측정된다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/survivorship.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trading_lab.strategies import get_handler  # noqa: E402

BASE30 = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "BAC", "WFC", "GS", "JNJ", "PFE", "MRK", "UNH",
    "KO", "PEP", "PG", "WMT", "MCD", "COST",
    "CAT", "DE", "HON", "GE", "XOM", "CVX", "NEE", "DIS",
]
# 닷컴/금융위기/원자재/리테일에서 크게 부진했던(데이터 존재) 종목.
LAGGARDS = [
    "INTC", "IBM", "CSCO", "ORCL", "QCOM", "HPQ",      # 닷컴·테크 정체
    "AIG", "C", "T", "VZ",                              # 금융위기·통신 정체
    "F", "GM", "X", "CLF", "MRO", "OXY", "FCX", "MOS",  # 경기순환·원자재
    "M", "KSS", "GPS", "BBY", "GIS", "KHC", "PARA",     # 리테일·소비 부진
    "MU", "WDC", "HAL", "NEM",
]


def evaluate(handler, base_cfg, universe, label) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["universe"] = universe
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    loaded = list(raw.columns.get_level_values(0).unique())
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO",
                                  phase="all", bars_per_year=252)
    m = art.metrics
    return {
        "label": label,
        "requested": len(universe),
        "loaded": len(loaded),
        "exposure": art.metadata["avg_exposure"],
        "cagr": m["cagr"], "mdd": m["max_drawdown"], "sharpe": m["sharpe"],
        "bnh_cagr": m["buy_hold_cagr"], "bnh_mdd": m["buy_hold_max_drawdown"],
        "bnh_sharpe": m["buy_hold_sharpe"],
        "sharpe_vs_bnh": (m["sharpe"] - m["buy_hold_sharpe"])
        if (m["sharpe"] is not None and m["buy_hold_sharpe"] is not None) else None,
    }


def _p(v, pct=False):
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def main() -> int:
    handler = get_handler("yoon1")
    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1.json").read_text()
    )
    print("우량주 30 평가 ...", flush=True)
    a = evaluate(handler, base_cfg, BASE30, "우량주 30")
    ext = BASE30 + [s for s in LAGGARDS if s not in BASE30]
    print(f"확장 유니버스({len(ext)}) 평가 ...", flush=True)
    b = evaluate(handler, base_cfg, ext, "확장(부진 포함)")

    lines = [
        "# profile-portfolio-v1 생존편향 점검",
        "",
        "기본 우량주 30 vs 부진·위기 폭락 종목을 섞은 확장 유니버스. phase=all, "
        "시장필터 ON, top_k=20, floor=1.0.",
        "",
        "| 유니버스 | 로드 | 전략 노출 | 전략 CAGR | 전략 MDD | 전략 Sharpe | "
        "B&H CAGR | B&H MDD | B&H Sharpe | 전략−B&H Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in (a, b):
        lines.append(
            f"| {r['label']} | {r['loaded']}/{r['requested']} | "
            f"{_p(r['exposure'], True)} | {_p(r['cagr'], True)} | "
            f"{_p(r['mdd'], True)} | {_p(r['sharpe'])} | {_p(r['bnh_cagr'], True)} | "
            f"{_p(r['bnh_mdd'], True)} | {_p(r['bnh_sharpe'])} | "
            f"**{_p(r['sharpe_vs_bnh'])}** |"
        )
    md = "\n".join(lines) + "\n"
    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "survivorship.md").write_text(md, encoding="utf-8")
    (outdir / "survivorship.json").write_text(
        json.dumps({"base": a, "extended": b}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"리포트: {outdir / 'survivorship.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
