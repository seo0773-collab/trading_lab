#!/usr/bin/env python
"""유니버스 성과 배치 — profile-sizing-v1을 종목별로 백테스트하고 buy & hold 대비
성과를 집계한다.

공통 파이프라인(BacktestService.run)을 종목마다 그대로 호출하므로 각 run은
1 run = 1 종목 계약대로 독립 아티팩트(var/runs/...)를 남긴다. 여기서는 그
metrics.json(전략·B&H 성과·초과수익)을 읽어 종목별 비교표 + 유니버스 종합 리포트를
만든다. 실패 종목은 status로 남기고 배치는 계속한다.

이 전략은 데이터 적합이 없는 규칙 기반 사이징이라 과적합 우려가 없으므로, 기본
phase=all(전체 기간)로 장기 B&H 비교를 수행한다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/batch.py --phase all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trading_lab.paths import var_dir  # noqa: E402
from trading_lab.service import BacktestRequest, BacktestService  # noqa: E402
from trading_lab.storage import RunStore  # noqa: E402

# 섹터 분산 대형주 30종목(var/market_data 캐시 완료분).
DEFAULT_UNIVERSE = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "UNH",
    "KO", "PEP", "PG", "WMT", "MCD", "COST",
    "CAT", "DE", "HON", "GE",
    "XOM", "CVX", "NEE", "DIS",
]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_symbol(service, store, strategy_id, symbol, phase, initial_capital) -> dict:
    try:
        run_id = service.run(BacktestRequest(
            strategy_id=strategy_id, symbol=symbol, phase=phase,
            chart_type="stock", initial_capital=initial_capital,
        ))
    except Exception as exc:  # noqa: BLE001 — 한 종목 실패가 배치를 멈추지 않게.
        return {"symbol": symbol, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:160]}

    run = store.get_run(run_id) or {}
    if run.get("status") != "succeeded":
        return {"symbol": symbol, "status": run.get("status", "unknown"),
                "error": (run.get("error") or "")[:160]}

    run_dir = var_dir() / "runs" / run["run_name"]
    m = _read_json(run_dir / "metrics.json") or {}
    meta = _read_json(run_dir / "forecast_metadata.json") or {}
    strat = m.get("total_return")
    bnh = m.get("buy_hold_return")
    return {
        "symbol": symbol, "status": "ok",
        "trades": int(m.get("trades", 0)),
        "total_return": strat,
        "sharpe": m.get("sharpe"),
        "max_drawdown": m.get("max_drawdown"),
        "cagr": m.get("cagr"),
        "volatility": m.get("volatility"),
        "bnh_return": bnh,
        "bnh_sharpe": m.get("buy_hold_sharpe"),
        "bnh_max_drawdown": m.get("buy_hold_max_drawdown"),
        "excess": m.get("excess_return_vs_bnh"),
        "avg_exposure": meta.get("avg_exposure"),
        "n_bars": int(meta.get("n_bars", 0)),
        "run_name": run["run_name"],
    }


def _clean(values):
    return np.array([v for v in values if v is not None and not _isnan(v)], dtype=float)


def _isnan(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def _mean(values) -> float:
    arr = _clean(values)
    return float(arr.mean()) if arr.size else float("nan")


def _median(values) -> float:
    arr = _clean(values)
    return float(np.median(arr)) if arr.size else float("nan")


def aggregate(rows, phase) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    strat_ret = [r["total_return"] for r in ok]
    bnh_ret = [r["bnh_return"] for r in ok]
    excess = [r["excess"] for r in ok]
    beat = [1.0 if (r["excess"] is not None and r["excess"] > 0) else 0.0 for r in ok]
    # 위험조정: MDD 개선(전략 MDD가 B&H보다 얕은 비율).
    dd_better = [
        1.0 for r in ok
        if r["max_drawdown"] is not None and r["bnh_max_drawdown"] is not None
        and r["max_drawdown"] > r["bnh_max_drawdown"]  # 둘 다 음수, 큰 쪽이 얕음
    ]
    return {
        "phase": phase,
        "n_requested": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "mean_strategy_return": _mean(strat_ret),
        "median_strategy_return": _median(strat_ret),
        "mean_bnh_return": _mean(bnh_ret),
        "median_bnh_return": _median(bnh_ret),
        "mean_excess": _mean(excess),
        "median_excess": _median(excess),
        "frac_beat_bnh": float(np.mean(beat)) if beat else float("nan"),
        "mean_strategy_sharpe": _mean([r["sharpe"] for r in ok]),
        "mean_bnh_sharpe": _mean([r["bnh_sharpe"] for r in ok]),
        "mean_strategy_mdd": _mean([r["max_drawdown"] for r in ok]),
        "mean_bnh_mdd": _mean([r["bnh_max_drawdown"] for r in ok]),
        "frac_mdd_improved": (len(dd_better) / len(ok)) if ok else float("nan"),
        "mean_exposure": _mean([r["avg_exposure"] for r in ok]),
        "total_trades": int(sum(r["trades"] for r in ok)),
    }


def _fmt(v, pct=False, places=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.{places}f}%" if pct else f"{v:.{places}f}"


def to_markdown(rows, agg, strategy_id) -> str:
    a = agg
    lines = [
        f"# {strategy_id} 유니버스 성과 — Buy&Hold 대비 (phase={a['phase']})",
        "",
        f"- 요청 종목: **{a['n_requested']}** · 성공: {a['n_ok']} · 실패: {a['n_failed']}",
        f"- 총 거래수: {a['total_trades']} · 평균 익스포저(주식 비중): "
        f"{_fmt(a['mean_exposure'], pct=True, places=0)}",
        "",
        "## 종합 (전략 vs Buy&Hold)",
        "",
        "| 지표 | 전략 | Buy&Hold | 차이 |",
        "| --- | ---: | ---: | ---: |",
        f"| 평균 총수익률 | {_fmt(a['mean_strategy_return'], pct=True)} | "
        f"{_fmt(a['mean_bnh_return'], pct=True)} | "
        f"{_fmt(a['mean_excess'], pct=True)} |",
        f"| 중앙값 총수익률 | {_fmt(a['median_strategy_return'], pct=True)} | "
        f"{_fmt(a['median_bnh_return'], pct=True)} | "
        f"{_fmt(a['median_excess'], pct=True)} |",
        f"| 평균 Sharpe | {_fmt(a['mean_strategy_sharpe'], places=3)} | "
        f"{_fmt(a['mean_bnh_sharpe'], places=3)} | — |",
        f"| 평균 MDD | {_fmt(a['mean_strategy_mdd'], pct=True)} | "
        f"{_fmt(a['mean_bnh_mdd'], pct=True)} | — |",
        "",
        f"- **B&H 총수익률을 이긴 종목 비율**: {_fmt(a['frac_beat_bnh'], pct=True, places=0)}",
        f"- **B&H보다 낙폭(MDD)이 얕았던 종목 비율**: "
        f"{_fmt(a['frac_mdd_improved'], pct=True, places=0)}",
        "",
        "## 종목별 결과",
        "",
        "| 종목 | 상태 | 거래 | 익스포저 | 전략수익 | B&H수익 | 초과 | "
        "전략Sharpe | B&H Sharpe | 전략MDD | B&H MDD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = {"ok": 0}
    for r in sorted(rows, key=lambda x: (order.get(x["status"], 1),
                                         -(x.get("excess") or -9))):
        if r["status"] != "ok":
            lines.append(f"| {r['symbol']} | {r['status']} | — | — | — | — | — | "
                         "— | — | — | — |")
            continue
        lines.append(
            f"| {r['symbol']} | ok | {r['trades']} | "
            f"{_fmt(r['avg_exposure'], pct=True, places=0)} | "
            f"{_fmt(r['total_return'], pct=True)} | {_fmt(r['bnh_return'], pct=True)} | "
            f"{_fmt(r['excess'], pct=True)} | {_fmt(r['sharpe'], places=2)} | "
            f"{_fmt(r['bnh_sharpe'], places=2)} | {_fmt(r['max_drawdown'], pct=True)} | "
            f"{_fmt(r['bnh_max_drawdown'], pct=True)} |"
        )
    return "\n".join(lines) + "\n"


def run_strategy(service, store, strategy_id, symbols, phase, initial_capital) -> dict:
    rows, n = [], len(symbols)
    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        row = run_symbol(service, store, strategy_id, symbol, phase, initial_capital)
        rows.append(row)
        note = (f"ret={_fmt(row.get('total_return'), pct=True)} "
                f"bnh={_fmt(row.get('bnh_return'), pct=True)} "
                f"excess={_fmt(row.get('excess'), pct=True)}"
                if row["status"] == "ok" else row.get("error", row["status"]))
        print(f"  [{strategy_id}] {i}/{n} {symbol} [{row['status']}] {note} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return {"strategy_id": strategy_id, "rows": rows,
            "summary": aggregate(rows, phase)}


def compare_markdown(results: list[dict], phase: str) -> str:
    """전략별 종합을 한 표로 비교(Buy&Hold 공통 기준)."""
    bnh = results[0]["summary"]
    lines = [
        f"# profile-sizing 변형 비교 — Buy&Hold 대비 (phase={phase})",
        "",
        f"- 유니버스 30종목 공통 · Buy&Hold 평균 총수익률 "
        f"{_fmt(bnh['mean_bnh_return'], pct=True)} · 평균 MDD "
        f"{_fmt(bnh['mean_bnh_mdd'], pct=True)} · 평균 Sharpe "
        f"{_fmt(bnh['mean_bnh_sharpe'], places=3)}",
        "",
        "| 전략 | 평균 익스포저 | 평균 총수익 | 중앙값 총수익 | 평균 Sharpe | "
        "평균 MDD | B&H 수익 우위 | MDD 개선 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        a = r["summary"]
        lines.append(
            f"| {r['strategy_id']} | {_fmt(a['mean_exposure'], pct=True, places=0)} | "
            f"{_fmt(a['mean_strategy_return'], pct=True)} | "
            f"{_fmt(a['median_strategy_return'], pct=True)} | "
            f"{_fmt(a['mean_strategy_sharpe'], places=3)} | "
            f"{_fmt(a['mean_strategy_mdd'], pct=True)} | "
            f"{_fmt(a['frac_beat_bnh'], pct=True, places=0)} | "
            f"{_fmt(a['frac_mdd_improved'], pct=True, places=0)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--strategies",
                        default="profile-sizing-v1,profile-sizing-trend-v1,"
                                "profile-sizing-exp-v1")
    parser.add_argument("--phase", default="all", choices=["validation", "all"])
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    store = RunStore()
    service = BacktestService(store)
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for strategy_id in strategies:
        print(f"\n=== {strategy_id} ({len(symbols)} symbols) ===", flush=True)
        res = run_strategy(service, store, strategy_id, symbols, args.phase,
                           args.initial_capital)
        results.append(res)
        slug = strategy_id.replace("/", "_")
        (args.outdir / f"perf_{slug}.json").write_text(
            json.dumps({"summary": res["summary"], "per_symbol": res["rows"]},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (args.outdir / f"perf_{slug}.md").write_text(
            to_markdown(res["rows"], res["summary"], strategy_id), encoding="utf-8"
        )

    if len(results) > 1:
        cmp_md = compare_markdown(results, args.phase)
        (args.outdir / "compare.md").write_text(cmp_md, encoding="utf-8")
        print("\n" + cmp_md)
        print(f"비교 리포트: {args.outdir / 'compare.md'}")
    else:
        print(f"\n리포트: {args.outdir / ('perf_' + strategies[0] + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
