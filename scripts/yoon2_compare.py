"""yoon2 실데이터 비교 배치 — 델타 전환 트리거의 노이즈 통제 효과 검증.

게이트 3~4: raw 트리거(confirm_bars=1, 필터 off)와 노이즈 통제 변형들을 같은
종목·기간에서 비교해 "선행성이 거래비용을 이기는가"를 본다. Buy & Hold도 함께.

실행:
    PYTHONPATH=src .venv/bin/python scripts/yoon2_compare.py
    PYTHONPATH=src .venv/bin/python scripts/yoon2_compare.py --symbols SPY QQQ --period 10y
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler

DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]

# 비교 변형: 이름 -> config 오버라이드
VARIANTS: dict[str, dict] = {
    "raw(both)":   {"confirm_bars": 1, "min_hist_gap_atr": 0.0, "macd_zero_filter": False, "direction": "both"},
    "zero(both)":  {"confirm_bars": 2, "min_hist_gap_atr": 0.0, "macd_zero_filter": True, "direction": "both"},
    "zero(long)":  {"confirm_bars": 2, "min_hist_gap_atr": 0.0, "macd_zero_filter": True, "direction": "long"},
    "combo(long)": {"confirm_bars": 3, "min_hist_gap_atr": 0.5, "macd_zero_filter": True, "direction": "long"},
    "raw(long)":   {"confirm_bars": 1, "min_hist_gap_atr": 0.0, "macd_zero_filter": False, "direction": "long"},
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


def run(symbols: list[str], period: str, phase: str) -> str:
    handler = get_handler("yoon2")
    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "yoon2.json").read_text("utf-8")
    )
    base_cfg["period"] = period
    bpy = 252  # 일봉

    lines: list[str] = []
    lines.append(f"# yoon2 실데이터 비교 (period={period}, phase={phase})\n")
    lines.append(
        "각 종목에 대해 raw 트리거와 노이즈 통제 변형을 비교. "
        "**핵심: 통제 후 Sharpe↑·거래수↓(whipsaw 완화)인가.**\n"
    )
    summary: dict[str, list[float]] = {v: [] for v in VARIANTS}

    for symbol in symbols:
        raw = handler.load_data(symbol, base_cfg, synthetic=False)
        bh = buy_hold(raw, bpy)
        lines.append(f"\n## {symbol}  (bars={len(raw)})\n")
        lines.append("| 변형 | 거래수 | 승률 | 총수익 | Sharpe | MDD |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| **B&H** | - | - | {bh['total_return']:+.1%} | "
            f"{bh['sharpe']:.2f} | {bh['max_drawdown']:.1%} |"
        )
        for name, override in VARIANTS.items():
            cfg = copy.deepcopy(base_cfg)
            cfg.update(override)
            art = handler.build_artifacts(
                raw, cfg, symbol=symbol, phase=phase, bars_per_year=bpy,
            )
            m = art.metrics
            hit = f"{m['hit_rate']:.0%}" if m["hit_rate"] is not None else "-"
            shp = m["sharpe"] if m["sharpe"] is not None else float("nan")
            summary[name].append(shp if shp == shp else 0.0)
            lines.append(
                f"| {name} | {m['trades']} | {hit} | "
                f"{m['total_return']:+.1%} | {shp:.2f} | "
                f"{(m['max_drawdown'] or 0):.1%} |"
            )

    lines.append("\n## 변형별 평균 Sharpe (종목 평균)\n")
    lines.append("| 변형 | 평균 Sharpe |")
    lines.append("| --- | ---: |")
    for name, vals in summary.items():
        avg = float(np.mean(vals)) if vals else float("nan")
        lines.append(f"| {name} | {avg:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--period", default="10y")
    ap.add_argument("--phase", default="all", choices=["all", "validation"])
    args = ap.parse_args()

    report = run(args.symbols, args.period, args.phase)
    print(report)
    outdir = ROOT / "reports" / "yoon2"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"compare_{args.phase}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
