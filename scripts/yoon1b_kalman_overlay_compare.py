"""yoon1b × yoon2 칼만 오버레이 비교.

yoon2 분석에서 나온 "이진 레짐 필터가 best-day를 놓친다"는 진단을 yoon1b에 이식:
  ① 칼만 시장 레짐(market_filter.mode=kalman): SPY 200MA 이진 ×0.5 → 칼만 MACD
     z-score tanh 연속 스케일(절대 빠지지 않고 반등 시 빠른 재진입).
  ② 칼만 추세신호(trend_overlay.signal=kalman): 종목 추세강도를 칼만 평활 종가로
     계산(스케일 동일, 잔물결만 완화).
4변형(base/+시장/+추세/둘다)을 같은 유니버스·기간에서 SPY·B&H와 비교한다.
phase=test가 holdout(진짜 OOS).

실행:
    PYTHONPATH=src .venv/bin/python scripts/yoon1b_kalman_overlay_compare.py
"""
from __future__ import annotations

import argparse
import copy
import json

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler

PHASES = ["all", "test"]
VARIANTS: dict[str, dict] = {
    "base":            {},
    "+kal_market":     {"market_filter": {"mode": "kalman"}},
    "+kal_trend":      {"trend_overlay": {"signal": "kalman"}},
    "+both":           {"market_filter": {"mode": "kalman"},
                        "trend_overlay": {"signal": "kalman"}},
}


def _merge(base: dict, override: dict) -> dict:
    """nested dict 부분 병합(override의 하위 키만 덮어씀)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{v*100:+.1f}%" if pct else f"{v:.3f}"


def run(period: str) -> str:
    handler = get_handler("yoon1b")
    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1b.json").read_text("utf-8")
    )
    if period:
        base_cfg["period"] = period
    bpy = 252

    print("[load] yoon1b 유니버스 로딩...", flush=True)
    raw = handler.load_data("PORTFOLIO", base_cfg, synthetic=False)

    lines = [f"# yoon1b × 칼만 오버레이 비교 (period={base_cfg.get('period')})\n"]
    lines.append(
        "베이스=yoon1b(top_k20·monthly·gain1.25·이진 SPY200MA필터·SMA추세). "
        "변형: +kal_market(칼만 연속 시장레짐), +kal_trend(칼만 추세신호), +both. "
        "주 벤치마크=SPY. `exp`=평균 주식노출. **test=holdout(OOS).**\n"
    )

    spy_row = {}
    for phase in PHASES:
        lines.append(f"\n## phase = {phase}\n")
        lines.append(
            "| 변형 | exp | CAGR | Sharpe | MDD | vs SPY(Sharpe) |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, override in VARIANTS.items():
            cfg = _merge(base_cfg, override)
            art = handler.build_artifacts(
                raw, cfg, symbol="PORTFOLIO", phase=phase, bars_per_year=bpy,
            )
            m = art.metrics
            exp = art.metadata.get("avg_exposure")
            bench = m.get("benchmark_kind")
            spy_row[phase] = (m.get("buy_hold_cagr"), m.get("buy_hold_sharpe"),
                              m.get("buy_hold_max_drawdown"), bench)
            d_sharpe = (
                m["sharpe"] - m["buy_hold_sharpe"]
                if m.get("sharpe") is not None and m.get("buy_hold_sharpe") is not None
                else None
            )
            lines.append(
                f"| {name} | {_fmt(exp)} | {_fmt(m.get('cagr'))} | "
                f"{_fmt(m.get('sharpe'), pct=False)} | {_fmt(m.get('max_drawdown'))} | "
                f"{_fmt(d_sharpe, pct=False)} |"
            )
        c, s, mdd, bench = spy_row[phase]
        lines.append(
            f"| **{bench} (B&H)** | 100% | {_fmt(c)} | {_fmt(s, pct=False)} | "
            f"{_fmt(mdd)} | - |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="")
    args = ap.parse_args()
    report = run(args.period)
    print(report)
    out = ROOT / "reports" / "yoon2" / "yoon1b_kalman_overlay.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
