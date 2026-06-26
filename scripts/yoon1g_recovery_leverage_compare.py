"""yoon1g 후보 — 회복 게이팅 레버리지 슬리브 + 시장필터 보강(A안).

배경: "깊은 저가권 + 회복 확인"에서만 2x 섹터를 섞는 슬리브(leverage_sleeve)가
holdout서 세션 첫 진짜 개선이었으나(CAGR+1.3%p·Sharpe↑·MDD~보존), full-cycle
(2008 포함)에선 RECOVERY가 위기 바닥의 false 반등에 몇 번 물려 소폭 열위였다
(Sharpe 0.81→0.77·MDD-17.6→-22.3%).

A안: 슬리브 게이트에 **시장필터 ON(SPY>200MA) 동시 요구**(require_market_on)를
더해 위기 바닥을 차단한다. RECOVERY(자기 국면) ∧ 시장정상(시장 추세)일 때만 2x를
태운다. additive·기본 off이므로 다른 전략엔 영향 없음.

비교: 1x(12) vs MIX 무게이팅 vs MIX RECOVERY게이팅 vs MIX RECOVERY+시장ON(A안) vs SPY.
phase all(2008 포함)·test(holdout), 2007-02~ 동일기간. 연도별 2x 점유율 진단 포함.

실행:
    PYTHONPATH=src:scripts .venv/bin/python scripts/yoon1g_recovery_leverage_compare.py
"""
from __future__ import annotations

import copy
import json

import pandas as pd

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler

ONE = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU",
       "XLB", "XLRE", "TLT", "GLD"]                       # 1x 분산 (12)
TWO = ["ROM", "UYG", "RXL", "DIG", "UXI", "UGE", "UCC", "UPW",
       "UYM", "URE"]                                       # 2x 섹터 슬리브 (10)
MIX = ONE + TWO
START = "2007-02-01"


def _run(h, base, uni, tk, *, sleeve=False, market_on=False):
    c = copy.deepcopy(base)
    c["universe"] = uni
    c["top_k"] = tk
    if sleeve:
        c["leverage_sleeve"] = {
            "enabled": True, "symbols": TWO, "regimes": ["RECOVERY"],
            "require_market_on": market_on,
        }
    raw = h.load_data("PORTFOLIO", c, synthetic=False)
    raw = raw.loc[pd.to_datetime(raw.index) >= START]
    return c, raw


def _build(h, c, raw, ph):
    return h.build_artifacts(raw, c, symbol="PORTFOLIO", phase=ph,
                             bars_per_year=252)


def run() -> str:
    h = get_handler("yoon1f")
    base = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1f.json").read_text("utf-8")
    )
    print("[load] 유니버스 로딩...", flush=True)
    variants = {
        "1x (12)": _run(h, base, ONE, 12),
        "MIX 무게이팅": _run(h, base, MIX, 12, sleeve=True, market_on=False),
        "MIX 게이팅(RECOVERY)": _run(h, base, MIX, 12, sleeve=True,
                                    market_on=False),
        "MIX 게이팅+시장ON(A)": _run(h, base, MIX, 12, sleeve=True,
                                   market_on=True),
    }
    # 무게이팅: leverage_sleeve 자체를 끄고 2x를 상시 후보로(컨트래리언 점수대로).
    c0 = copy.deepcopy(base)
    c0["universe"] = MIX
    c0["top_k"] = 12
    raw0 = h.load_data("PORTFOLIO", c0, synthetic=False)
    raw0 = raw0.loc[pd.to_datetime(raw0.index) >= START]
    variants["MIX 무게이팅"] = (c0, raw0)

    lines = ["# yoon1g 후보 — 회복 레버리지 슬리브 + 시장필터 보강(A안)\n"]
    lines.append(
        "RECOVERY(자기 국면)에서만 2x를 태우던 슬리브에 **시장필터 ON(SPY>200MA) "
        "동시 요구**를 더해 위기 바닥의 false 반등을 차단한다. 1x(12) vs 무게이팅 vs "
        "RECOVERY게이팅 vs RECOVERY+시장ON(A) vs SPY. 2007-02~ 동일기간.\n"
    )
    for ph in ("test", "all"):
        tag = "holdout, OOS" if ph == "test" else "2008 포함"
        lines.append(f"\n## phase={ph} ({tag})\n")
        lines.append("| 변형 | 노출 | CAGR | Sharpe | MDD |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        spy = None
        for name, (c, raw) in variants.items():
            a = _build(h, c, raw, ph)
            m = a.metrics
            spy = (m["buy_hold_cagr"], m["buy_hold_sharpe"],
                   m["buy_hold_max_drawdown"])
            lines.append(
                f"| {name} | {a.metadata['avg_exposure']*100:.0f}% | "
                f"{m['cagr']*100:+.1f}% | {m['sharpe']:.3f} | "
                f"{m['max_drawdown']*100:.1f}% |"
            )
        lines.append(
            f"| SPY | 100% | {spy[0]*100:+.1f}% | {spy[1]:.3f} | "
            f"{spy[2]*100:.1f}% |"
        )

    # 연도별 2x 점유율(주식 중) — RECOVERY 단독 vs RECOVERY+시장ON 비교.
    lines.append("\n## 연도별 2x 점유율(주식 중)\n")
    for name in ("MIX 게이팅(RECOVERY)", "MIX 게이팅+시장ON(A)"):
        c, raw = variants[name]
        a = _build(h, c, raw, "all")
        wave = a.extras["portfolio_wave"].copy()
        wave["yr"] = pd.to_datetime(wave["time"]).dt.year
        two = [s for s in TWO if s in wave.columns]
        one = [s for s in ONE if s in wave.columns]
        wave["two"] = wave[two].sum(axis=1)
        wave["stock"] = wave[two + one].sum(axis=1)
        g = wave.groupby("yr").apply(
            lambda d: (d["two"].sum() / d["stock"].sum())
            if d["stock"].sum() > 0 else 0.0,
            include_groups=False,
        )
        share = " ".join(f"{yr%100:02d}:{v*100:.0f}" for yr, v in g.items())
        lines.append(f"- **{name}**: `{share}` (%)")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run()
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon1g_recovery_leverage.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
