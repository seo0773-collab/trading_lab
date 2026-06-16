#!/usr/bin/env python
"""유니버스 성과 배치 — 실데이터로 fin-sensitivity-v1을 종목별로 백테스트하고
종합 성과를 집계한다 (finance_plan.txt §28 A 배치 스크리너).

batch.py가 예측력(IC)만 보는 연구 도구라면, 이 스크립트는 **공통 파이프라인**
(BacktestService.run)을 종목마다 그대로 호출해 실제 거래·평가자산·성과지표를 모은다.
각 run은 1 run = 1 종목 계약대로 독립 아티팩트(var/runs/...)를 남기므로 대시보드
"비교" 화면에서도 그대로 보인다. 여기서는 그 metrics.json/forecast_metadata.json/
learning_summary.json 을 읽어 종목별 비교표 + 유니버스 종합 리포트를 만든다.

실패 종목은 status=failed로 남기고 배치는 계속한다(§13 결측에도 비중단).

Usage:
    .venv/bin/python scripts/finance_sensitivity/perf_batch.py --phase all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trading_lab.paths import var_dir  # noqa: E402
from trading_lab.service import BacktestRequest, BacktestService  # noqa: E402
from trading_lab.storage import RunStore  # noqa: E402

# 섹터 분산 대형주 30종목(var/fundamentals 수집 완료분).
DEFAULT_UNIVERSE = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "UNH",
    "KO", "PEP", "PG", "WMT", "MCD", "COST",
    "CAT", "DE", "HON", "GE",
    "XOM", "CVX", "NEE", "DIS",
]


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ic_row(summary: list | None, horizon: int) -> tuple[float, float, int]:
    """learning_summary에서 (spearman_ic, direction_accuracy, samples)."""
    if not summary:
        return (float("nan"), float("nan"), 0)
    for row in summary:
        if int(row.get("horizon_days", -1)) == horizon:
            return (
                float(row.get("spearman_ic", float("nan"))),
                float(row.get("direction_accuracy", float("nan"))),
                int(row.get("samples", 0)),
            )
    return (float("nan"), float("nan"), 0)


def run_symbol(
    service: BacktestService, store: RunStore, symbol: str, phase: str,
    initial_capital: float,
) -> dict:
    try:
        run_id = service.run(BacktestRequest(
            strategy_id="fin-sensitivity-v1",
            symbol=symbol,
            phase=phase,
            chart_type="stock",
            initial_capital=initial_capital,
        ))
    except Exception as exc:  # noqa: BLE001 — 한 종목 실패가 배치를 멈추지 않게.
        return {"symbol": symbol, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:160]}

    run = store.get_run(run_id) or {}
    if run.get("status") != "succeeded":
        return {"symbol": symbol, "status": run.get("status", "unknown"),
                "error": (run.get("error") or "")[:160]}

    run_dir = var_dir() / "runs" / run["run_name"]
    metrics = _read_json(run_dir / "metrics.json") or {}
    meta = _read_json(run_dir / "forecast_metadata.json") or {}
    summary = _read_json(run_dir / "learning_summary.json")
    ic20, da20, n20 = _ic_row(summary, 20)
    ic60, da60, n60 = _ic_row(summary, 60)

    return {
        "symbol": symbol,
        "status": "ok",
        "trades": int(metrics.get("trades", 0)),
        "hit_rate": metrics.get("hit_rate"),
        "total_return": metrics.get("total_return"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "final_account_value": metrics.get("final_account_value"),
        "profit_abs": metrics.get("profit_abs"),
        "n_events": int(meta.get("n_events", 0)),
        "n_predicted": int(meta.get("n_predicted", 0)),
        "insufficient": bool(meta.get("insufficient_train_data", False)),
        "ic20": ic20, "ic60": ic60,
        "dir_acc60": da60, "samples60": n60,
        "run_name": run["run_name"],
    }


def _mean(values: list[float]) -> float:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _median(values: list[float]) -> float:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def aggregate(rows: list[dict], phase: str, horizon: int) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    traded = [r for r in ok if r["trades"] > 0]
    rets = [r["total_return"] for r in traded]
    ic60 = [r["ic60"] for r in ok]
    ic20 = [r["ic20"] for r in ok]
    return {
        "phase": phase,
        "horizon": horizon,
        "n_requested": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "n_traded": len(traded),
        "n_insufficient": sum(1 for r in ok if r["insufficient"]),
        "total_trades": int(sum(r["trades"] for r in ok)),
        "mean_total_return": _mean(rets),
        "median_total_return": _median(rets),
        "frac_return_positive": (
            float(np.mean([r > 0 for r in rets])) if rets else float("nan")
        ),
        "mean_sharpe": _mean([r["sharpe"] for r in traded]),
        "mean_max_drawdown": _mean([r["max_drawdown"] for r in traded]),
        "mean_hit_rate": _mean([r["hit_rate"] for r in traded]),
        "mean_ic20": _mean(ic20),
        "mean_ic60": _mean(ic60),
        "median_ic60": _median(ic60),
        "frac_ic60_positive": (
            float(np.mean([v > 0 for v in ic60 if not np.isnan(v)]))
            if any(not np.isnan(v) for v in ic60) else float("nan")
        ),
    }


def _fmt(value, pct=False, places=2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if pct:
        return f"{value * 100:.{places}f}%"
    return f"{value:.{places}f}"


def to_markdown(rows: list[dict], agg: dict) -> str:
    h = agg["horizon"]
    lines = [
        f"# fin-sensitivity-v1 실데이터 성과 배치 (phase={agg['phase']})",
        "",
        f"- 요청 종목: **{agg['n_requested']}** · 성공: {agg['n_ok']} · "
        f"실패: {agg['n_failed']} · 거래발생: {agg['n_traded']} · "
        f"학습부족(insufficient): {agg['n_insufficient']}",
        f"- 총 거래수: {agg['total_trades']}",
        "",
        "## 종합 성과",
        "",
        "| 지표 | 값 |",
        "| --- | --- |",
        f"| 평균 총수익률(거래종목) | {_fmt(agg['mean_total_return'], pct=True)} |",
        f"| 중앙값 총수익률 | {_fmt(agg['median_total_return'], pct=True)} |",
        f"| 수익 양수 비율 | {_fmt(agg['frac_return_positive'], pct=True, places=0)} |",
        f"| 평균 Sharpe | {_fmt(agg['mean_sharpe'])} |",
        f"| 평균 MDD | {_fmt(agg['mean_max_drawdown'], pct=True)} |",
        f"| 평균 승률 | {_fmt(agg['mean_hit_rate'], pct=True, places=0)} |",
        f"| 평균 IC20 (예측력) | {_fmt(agg['mean_ic20'], places=3)} |",
        f"| 평균 IC{h} (예측력) | {_fmt(agg['mean_ic60'], places=3)} |",
        f"| 중앙값 IC{h} | {_fmt(agg['median_ic60'], places=3)} |",
        f"| IC{h} 양수 비율 | {_fmt(agg['frac_ic60_positive'], pct=True, places=0)} |",
        "",
        "## 종목별 결과",
        "",
        "| 종목 | 상태 | 거래 | 총수익률 | Sharpe | MDD | 승률 | "
        f"IC{h} | 방향적중 | 분기수 | 예측수 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = {"ok": 0, "failed": 1}
    for r in sorted(rows, key=lambda x: (order.get(x["status"], 2),
                                         -(x.get("total_return") or -9))):
        if r["status"] != "ok":
            lines.append(
                f"| {r['symbol']} | {r['status']} | — | — | — | — | — | "
                f"— | — | — | — |"
            )
            continue
        lines.append(
            f"| {r['symbol']} | ok | {r['trades']} | "
            f"{_fmt(r['total_return'], pct=True)} | {_fmt(r['sharpe'])} | "
            f"{_fmt(r['max_drawdown'], pct=True)} | {_fmt(r['hit_rate'], pct=True, places=0)} | "
            f"{_fmt(r['ic60'], places=3)} | {_fmt(r['dir_acc60'], pct=True, places=0)} | "
            f"{r['n_events']} | {r['n_predicted']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--phase", default="all",
                        choices=["validation", "all"])
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "reports" / "finance_sensitivity",
    )
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    store = RunStore()
    service = BacktestService(store)

    rows: list[dict] = []
    n = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        row = run_symbol(service, store, symbol, args.phase, args.initial_capital)
        rows.append(row)
        note = (f"trades={row.get('trades', '-')} "
                f"ret={_fmt(row.get('total_return'), pct=True)} "
                f"ic60={_fmt(row.get('ic60'), places=3)}"
                if row["status"] == "ok" else row.get("error", row["status"]))
        print(f"PROGRESS {i}/{n} {symbol} [{row['status']}] {note} "
              f"({time.time() - t0:.1f}s)", flush=True)

    agg = aggregate(rows, args.phase, args.horizon)
    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": agg, "per_symbol": rows}
    (args.outdir / "perf_batch.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = to_markdown(rows, agg)
    (args.outdir / "perf_batch.md").write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"리포트: {args.outdir / 'perf_batch.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
