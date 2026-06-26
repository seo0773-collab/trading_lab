"""종목별 3-way 비교: yoon2(단일종목 칼만MACD) vs yoon1b(단일종목 사이징) vs B&H.

"이 종목을 yoon2로 다룰지" 판단용. 같은 종목·기간으로 정렬해 위험조정수익
(Sharpe)·총수익·MDD를 나란히 본다. yoon1b는 단일종목 유니버스(top_k=1)로 돌려
포트폴리오 전략의 사이징 로직을 그 종목에 적용한 버전이다.

실행:
    PYTHONPATH=src .venv/bin/python scripts/yoon2_vs_yoon1b.py
    PYTHONPATH=src .venv/bin/python scripts/yoon2_vs_yoon1b.py --symbols SPY NVDA --period 10y
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler

DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "KO", "XOM", "JNJ"]


def _cfg(name: str, period: str) -> dict:
    cfg = json.loads(
        (ROOT / "configs" / "strategies" / f"{name}.json").read_text("utf-8")
    )
    cfg["period"] = period
    return cfg


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


def yoon2_metrics(sym, period, phase, bpy) -> tuple[dict, object]:
    h = get_handler("yoon2")
    cfg = _cfg("yoon2", period)  # 기본값 = 검증된 zero(long)
    raw = h.load_data(sym, cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol=sym, phase=phase, bars_per_year=bpy)
    return art.metrics, raw


def yoon1b_metrics(sym, period, phase, bpy) -> dict:
    h = get_handler("yoon1b")
    cfg = copy.deepcopy(_cfg("yoon1b", period))
    cfg["universe"] = [sym]   # 단일종목 유니버스
    cfg["top_k"] = 1
    raw = h.load_data(sym, cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol=sym, phase=phase, bars_per_year=bpy)
    return art.metrics


def _f(v, fmt):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return format(v, fmt)


def run(symbols: list[str], period: str, phase: str) -> str:
    bpy = 252
    lines = [f"# yoon2 vs yoon1b vs B&H — 종목별 (period={period}, phase={phase})\n"]
    lines.append(
        "yoon2=단일종목 칼만MACD(zero·long 기본값), yoon1b=단일종목 사이징"
        "(universe=[종목],top_k=1), B&H=매수후보유. **'최고'=Sharpe 1위 전략.**\n"
    )
    lines.append("| 종목 | 지표 | yoon2 | yoon1b | B&H | 최고 |")
    lines.append("| --- | --- | ---: | ---: | ---: | :--: |")

    wins = {"yoon2": 0, "yoon1b": 0, "B&H": 0}
    for sym in symbols:
        m2, raw = yoon2_metrics(sym, period, phase, bpy)
        m1 = yoon1b_metrics(sym, period, phase, bpy)
        bh = buy_hold(raw, bpy)

        sharpes = {
            "yoon2": m2.get("sharpe"), "yoon1b": m1.get("sharpe"),
            "B&H": bh.get("sharpe"),
        }
        best = max(
            sharpes, key=lambda k: (sharpes[k] if sharpes[k] is not None
                                    and sharpes[k] == sharpes[k] else -1e9)
        )
        wins[best] += 1

        lines.append(
            f"| **{sym}** | Sharpe | {_f(m2.get('sharpe'),'.2f')} | "
            f"{_f(m1.get('sharpe'),'.2f')} | {_f(bh.get('sharpe'),'.2f')} | "
            f"**{best}** |"
        )
        lines.append(
            f"| | 총수익 | {_f(m2.get('total_return'),'+.0%')} | "
            f"{_f(m1.get('total_return'),'+.0%')} | {_f(bh.get('total_return'),'+.0%')} | |"
        )
        lines.append(
            f"| | MDD | {_f(m2.get('max_drawdown'),'.0%')} | "
            f"{_f(m1.get('max_drawdown'),'.0%')} | {_f(bh.get('max_drawdown'),'.0%')} | |"
        )
        lines.append(
            f"| | 거래수 | {m2.get('trades','-')} | {m1.get('trades','-')} | - | |"
        )

    lines.append("\n## Sharpe 우위 집계\n")
    lines.append("| 전략 | 1위 종목수 |")
    lines.append("| --- | ---: |")
    for k, v in wins.items():
        lines.append(f"| {k} | {v} |")
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
    out = outdir / f"vs_yoon1b_{args.phase}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
