"""yoon2 확장 비교 — 진입 트리거(delta_turn|cross) × 분포 익절 사다리(off|on).

검증된 베이스 조합(`macd_zero_filter=True`, `direction=long`, `confirm_bars=2`)
위에서 2×2 격자를 같은 종목·기간에서 비교한다. 질문:
  ① 라인 크로싱(cross)이 델타 전환(delta_turn) 대비 위험조정수익을 개선하는가.
  ② in-sample 히스토델타 분포 기반 익절 사다리가 OOS에서 곡선을 다듬는가.
익절 분포는 identification 구간에서만 추정되므로 phase=validation이 진짜 OOS다.

실행:
    PYTHONPATH=src .venv/bin/python scripts/yoon2_tp_compare.py
    PYTHONPATH=src .venv/bin/python scripts/yoon2_tp_compare.py --symbols SPY NVDA --period 10y
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler

DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]
PHASES = ["all", "validation"]

# 검증된 베이스 + 2×2 격자(트리거 × 익절)
BASE = {"confirm_bars": 2, "min_hist_gap_atr": 0.0, "macd_zero_filter": True,
        "direction": "long"}
VARIANTS: dict[str, dict] = {
    "delta":     {"entry_trigger": "delta_turn", "tp_enabled": False},
    "delta+TP":  {"entry_trigger": "delta_turn", "tp_enabled": True},
    "cross":     {"entry_trigger": "cross",      "tp_enabled": False},
    "cross+TP":  {"entry_trigger": "cross",      "tp_enabled": True},
}


def buy_hold(raw, bars_per_year: int) -> dict:
    close = raw["close"].astype(float)
    ret = close.pct_change().dropna()
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    std = float(ret.std())
    return {
        "total_return": float(equity.iloc[-1]) - 1.0,
        "sharpe": (float(ret.mean()) / std * np.sqrt(bars_per_year)
                   if std > 0 else float("nan")),
        "max_drawdown": float((equity / peak - 1.0).min()),
    }


def _row(name: str, art) -> tuple[str, float]:
    m = art.metrics
    t = art.trades
    hit = f"{m['hit_rate']:.0%}" if m["hit_rate"] is not None else "-"
    shp = m["sharpe"] if m["sharpe"] is not None else float("nan")
    tp_share = (
        f"{(t['exit_reason'] == 'take_profit').mean():.0%}"
        if len(t) else "-"
    )
    mults = art.metadata.get("tp_atr_mults")
    mult_s = "/".join(f"{x:.1f}" for x in mults) if mults else "-"
    line = (
        f"| {name} | {m['trades']} | {hit} | {tp_share} | "
        f"{m['total_return']:+.1%} | {shp:.2f} | "
        f"{(m['max_drawdown'] or 0):.1%} | {mult_s} |"
    )
    return line, (shp if shp == shp else 0.0)


def run(symbols: list[str], period: str) -> str:
    handler = get_handler("yoon2")
    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "yoon2.json").read_text("utf-8")
    )
    base_cfg["period"] = period
    base_cfg.update(BASE)
    bpy = 252

    lines = [f"# yoon2 트리거×익절 비교 (period={period})\n"]
    lines.append(
        "베이스: `macd_zero_filter=True`·`direction=long`·`confirm_bars=2`. "
        "2×2 = 트리거(delta_turn|cross) × 익절 사다리(off|on). "
        "`TP%`=take_profit 청산 비중, `mults`=ATR 익절배수(33/66/90분위, "
        "identification 추정).\n"
    )
    # phase -> variant -> [sharpe...]
    summary: dict[str, dict[str, list[float]]] = {
        ph: {v: [] for v in VARIANTS} for ph in PHASES
    }

    for symbol in symbols:
        raw = handler.load_data(symbol, base_cfg, synthetic=False)
        lines.append(f"\n## {symbol}  (bars={len(raw)})\n")
        for phase in PHASES:
            bh = buy_hold(
                raw if phase == "all" else raw, bpy
            )
            lines.append(f"\n### phase = {phase}\n")
            lines.append(
                "| 변형 | 거래수 | 승률 | TP% | 총수익 | Sharpe | MDD | mults |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            if phase == "all":
                lines.append(
                    f"| **B&H** | - | - | - | {bh['total_return']:+.1%} | "
                    f"{bh['sharpe']:.2f} | {bh['max_drawdown']:.1%} | - |"
                )
            for name, override in VARIANTS.items():
                cfg = copy.deepcopy(base_cfg)
                cfg.update(override)
                art = handler.build_artifacts(
                    raw, cfg, symbol=symbol, phase=phase, bars_per_year=bpy,
                )
                line, shp = _row(name, art)
                lines.append(line)
                summary[phase][name].append(shp)

    lines.append("\n## 변형별 평균 Sharpe (종목 평균)\n")
    lines.append("| 변형 | " + " | ".join(f"{ph}" for ph in PHASES) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in PHASES) + " |")
    for name in VARIANTS:
        cells = [
            f"{np.mean(summary[ph][name]):.3f}" if summary[ph][name] else "nan"
            for ph in PHASES
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--period", default="10y")
    args = ap.parse_args()

    report = run(args.symbols, args.period)
    print(report)
    outdir = ROOT / "reports" / "yoon2"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "tp_trigger_compare.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
