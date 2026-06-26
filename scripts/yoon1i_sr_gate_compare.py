"""yoon1i — heatmap2 HVN 지지/저항 기대값 게이트 vs yoon1b (A/B).

yoon1b 종목 점수에 EV=(저항−종가)/(저항−지지) 게이트[g_min,1]를 곱한다(블렌드).
지지 근처=매수 기대(게이트 열림), 저항 근처=매도 기대(억제). yoon1h(VA위치 교체)와
달리 percentile 점수를 유지하고 미세조정한다.

규율(결과 확인 전 잠금): holdout(test) Sharpe ≥ yoon1b. 위험: HVN/percentile이 같은
프로파일에서 나와 정보가 겹칠 수 있음(yoon1h 기각·yoon3 방어다이얼 전례).

실행: PYTHONPATH=src:scripts .venv/bin/python scripts/yoon1i_sr_gate_compare.py
"""
from __future__ import annotations

import copy
import json

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler


def _cfg(name: str) -> dict:
    return json.loads(
        (ROOT / "configs" / "strategies" / f"{name}.json").read_text("utf-8")
    )


def _with_gmin(base_i: dict, g_min: float) -> dict:
    c = copy.deepcopy(base_i)
    c["sr_gate"] = {**c["sr_gate"], "enabled": True, "g_min": g_min}
    return c


def run() -> str:
    h = get_handler("yoon1b")
    base_b = _cfg("yoon1b")
    base_i = _cfg("yoon1i")
    print("[load] megacap30 로딩...", flush=True)
    raw = h.load_data("PORTFOLIO", base_b, synthetic=False)

    variants = {
        "yoon1b (게이트 off)": base_b,
        "yoon1i SR g0.3": _with_gmin(base_i, 0.3),
        "yoon1i SR g0.5": _with_gmin(base_i, 0.5),
        "yoon1i SR g0.7": _with_gmin(base_i, 0.7),
    }

    lines = ["# yoon1i — HVN 지지/저항 기대값 게이트 vs yoon1b\n"]
    lines.append(
        "yoon1b 점수 × EV=(저항−종가)/(저항−지지) 게이트[g_min,1]. 지지근처=매수기대"
        "(열림)·저항근처=매도기대(억제). 규율: **holdout(test) Sharpe ≥ yoon1b**.\n"
    )
    cache: dict[tuple[str, str], dict] = {}
    for ph in ("validation", "test", "all"):
        tag = {"validation": "val(선정)", "test": "holdout(OOS)",
               "all": "full-cycle"}[ph]
        lines.append(f"\n## phase={ph} ({tag})\n")
        lines.append("| 변형 | 노출 | CAGR | Sharpe | MDD |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        spy = None
        for name, cfg in variants.items():
            a = h.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase=ph,
                                  bars_per_year=252)
            cache[(name, ph)] = a.metrics
            m = a.metrics
            spy = (m["buy_hold_cagr"], m["buy_hold_sharpe"],
                   m["buy_hold_max_drawdown"])
            lines.append(
                f"| {name} | {a.metadata['avg_exposure']*100:.0f}% | "
                f"{m['cagr']*100:+.1f}% | {m['sharpe']:.3f} | "
                f"{m['max_drawdown']*100:.1f}% |"
            )
        if spy and spy[0] is not None:
            lines.append(
                f"| SPY | 100% | {spy[0]*100:+.1f}% | {spy[1]:.3f} | "
                f"{spy[2]*100:.1f}% |"
            )

    base_test = cache[("yoon1b (게이트 off)", "test")]["sharpe"]
    lines.append("\n## 규율 판정 (holdout test Sharpe 대비 yoon1b)\n")
    for name in variants:
        if name.startswith("yoon1b"):
            continue
        d = cache[(name, "test")]["sharpe"] - base_test
        lines.append(
            f"- {name}: test ΔSharpe {d:+.3f} → **{'통과' if d >= 0 else '기각'}**"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run()
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon1i_sr_gate.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
