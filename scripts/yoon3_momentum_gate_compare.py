"""yoon3 후보 — 칼만 히스토그램 누적프로파일 모멘텀 게이트(블렌드).

배경: yoon1b는 저가권(컨트래리언) 점수로 종목을 고르는 방어형 사이저다. 단일종목
yoon2(칼만 MACD 타이밍)의 교훈은 "이진 진입/청산은 whipsaw·best-day-miss로 자멸,
연속 노출이 답"이었다. 그래서 macd_raw.txt의 칼만 히스토그램(kalHist)을 *값의
누적프로파일 백분위*로 자기적응 정규화하고, [g_min,1.0] 게이트로 yoon1b 점수에
**곱한다**(블렌드: 저가권 × 모멘텀). 저가권이 가리켜도 모멘텀이 자기 분포 하위면
억제, 모멘텀이 올라오면 게이트가 열려 회복 진입 타이밍을 보강한다.

규율(결과 확인 전 잠금): **val Sharpe ≥ yoon1b** 이면서 holdout(test)·full-cycle(all)
모두에서 위험조정/낙폭이 무너지지 않을 때만 채택. (칼만 이식 이력상 회의적으로 검증.)

비교: yoon1b(게이트 off) vs 모멘텀게이트 g_min∈{0.7,0.5,0.3} vs contrarian(g_min0.5).
phase val/test/all, megacap30. 주 벤치마크 SPY.

실행:
    PYTHONPATH=src:scripts .venv/bin/python scripts/yoon3_momentum_gate_compare.py
"""
from __future__ import annotations

import copy
import json

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler


def _gate(g_min: float, direction: str = "momentum") -> dict:
    return {
        "enabled": True, "g_min": g_min, "fast_len": 12, "slow_len": 26,
        "signal_len": 9, "kalman_q": 0.01, "kalman_r": 0.10,
        "kalman_base": "MACD Line", "norm_window": 252, "bin_count": 120,
        "rolling_window": 0, "z_clip": 4.0, "direction": direction,
    }


def run() -> str:
    h = get_handler("yoon1b")
    base = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1b.json").read_text("utf-8")
    )
    print("[load] megacap30 로딩...", flush=True)
    raw = h.load_data("PORTFOLIO", base, synthetic=False)

    variants: dict[str, dict] = {
        "yoon1b (게이트 off)": base,
        "게이트 momentum g0.7": {**copy.deepcopy(base), "mom_gate": _gate(0.7)},
        "게이트 momentum g0.5": {**copy.deepcopy(base), "mom_gate": _gate(0.5)},
        "게이트 momentum g0.3": {**copy.deepcopy(base), "mom_gate": _gate(0.3)},
        "게이트 contrarian g0.5": {
            **copy.deepcopy(base), "mom_gate": _gate(0.5, "contrarian")
        },
    }

    lines = ["# yoon3 후보 — 칼만 히스토그램 누적프로파일 모멘텀 게이트\n"]
    lines.append(
        "yoon1b 저가권 점수 × 칼만 히스토그램(kalHist) 누적프로파일 백분위 게이트"
        "[g_min,1.0]. momentum=백분위↑→게이트 열림(상승 모멘텀에 비중), "
        "contrarian=백분위↓→게이트 열림. 규율: **val Sharpe ≥ yoon1b** 우선.\n"
    )
    cache: dict[tuple[str, str], object] = {}
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
        if spy:
            lines.append(
                f"| SPY | 100% | {spy[0]*100:+.1f}% | {spy[1]:.3f} | "
                f"{spy[2]*100:.1f}% |"
            )

    # 규율 판정: val Sharpe 기준 yoon1b 대비.
    base_val = cache[("yoon1b (게이트 off)", "validation")]["sharpe"]
    lines.append("\n## 규율 판정 (val Sharpe 대비 yoon1b)\n")
    for name in variants:
        if name.startswith("yoon1b"):
            continue
        d = cache[(name, "validation")]["sharpe"] - base_val
        verdict = "통과" if d >= 0 else "기각"
        lines.append(f"- {name}: val ΔSharpe {d:+.3f} → **{verdict}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run()
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon3_momentum_gate.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
