"""yoon1h — POC/VA 매물대 위치 사이징 vs yoon1b percentile (A/B).

배경: yoon1b는 종목별 `cumulative_percentile`(누적 분포에서 현재가의 하위 누적비율)을
bucket 가중에 넣어 비중을 정한다. yoon1h는 같은 볼륨 프로파일에서 **POC/VA 매물대
레벨**(heatmap1 계보)을 뽑아 현재가의 **Value Area 대비 연속 위치 `va_position`**
(VAL→0 싸다·POC→0.5·VAH→1 비싸다, VA 밖은 외삽 후 클립)을 같은 bucket에 넣는다.

같은 포트폴리오 엔진·같은 유니버스·같은 bucket을 통과하므로 "위치 측정법"만의
순수 A/B다. percentile은 분포상 위치라 둔하고, VA는 실제 거래가 쌓인 지지/저항
밴드 대비 위치라 POC 회귀 근거가 분명하다는 가설.

규율(결과 확인 전 잠금): **holdout(test) Sharpe ≥ yoon1b** 이면서 full-cycle(all)
에서 위험조정/낙폭이 무너지지 않을 때만 채택. (사이징 입력 교체 이력상 보수적 검증.)

비교: yoon1b(percentile) vs yoon1h(poc_va) va_pct∈{0.60,0.70,0.80}.
phase val/test/all, megacap30. 주 벤치마크 SPY.

실행:
    PYTHONPATH=src:scripts .venv/bin/python scripts/yoon1h_poc_va_compare.py
"""
from __future__ import annotations

import copy
import json

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler


def _load_cfg(name: str) -> dict:
    return json.loads(
        (ROOT / "configs" / "strategies" / f"{name}.json").read_text("utf-8")
    )


def _with_va_pct(base_h: dict, va_pct: float) -> dict:
    cfg = copy.deepcopy(base_h)
    cfg["profile"] = {**cfg.get("profile", {}), "compute_va": True, "va_pct": va_pct}
    cfg["position_source"] = "poc_va"
    return cfg


def run() -> str:
    h = get_handler("yoon1b")  # 핸들러는 공유(profile_portfolio)
    base_b = _load_cfg("yoon1b")
    base_h = _load_cfg("yoon1h")
    print("[load] megacap30 로딩...", flush=True)
    raw = h.load_data("PORTFOLIO", base_b, synthetic=False)

    variants: dict[str, dict] = {
        "yoon1b (percentile)": base_b,
        "yoon1h va_pct0.60": _with_va_pct(base_h, 0.60),
        "yoon1h va_pct0.70": _with_va_pct(base_h, 0.70),
        "yoon1h va_pct0.80": _with_va_pct(base_h, 0.80),
    }

    lines = ["# yoon1h — POC/VA 매물대 위치 사이징 vs yoon1b percentile\n"]
    lines.append(
        "yoon1b의 사이징 입력(cumulative_percentile)을 VA 매물대 대비 연속 위치"
        "(va_position: VAL→0·POC→0.5·VAH→1, 밖은 외삽+클립)로 교체. 같은 엔진·"
        "유니버스·bucket을 통과하는 순수 A/B. 규율: **holdout(test) Sharpe ≥ "
        "yoon1b** 우선.\n"
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
        if spy and spy[0] is not None:
            lines.append(
                f"| SPY | 100% | {spy[0]*100:+.1f}% | {spy[1]:.3f} | "
                f"{spy[2]*100:.1f}% |"
            )

    # 규율 판정: holdout(test) Sharpe 기준 yoon1b 대비.
    base_test = cache[("yoon1b (percentile)", "test")]["sharpe"]
    lines.append("\n## 규율 판정 (holdout test Sharpe 대비 yoon1b)\n")
    for name in variants:
        if name.startswith("yoon1b"):
            continue
        d = cache[(name, "test")]["sharpe"] - base_test
        verdict = "통과" if d >= 0 else "기각"
        lines.append(f"- {name}: test ΔSharpe {d:+.3f} → **{verdict}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run()
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon1h_poc_va.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
