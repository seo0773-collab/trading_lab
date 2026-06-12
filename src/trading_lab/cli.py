from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .paths import ROOT, database_path, ensure_runtime_dirs
from .service import BacktestRequest, BacktestService
from .storage import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize runtime directories and database")

    ui = subparsers.add_parser("ui", help="start the Streamlit dashboard")
    ui.add_argument("streamlit_args", nargs=argparse.REMAINDER)

    backtest = subparsers.add_parser("backtest", help="run a synchronous backtest")
    backtest.add_argument("--strategy", default="h72-price-v1")
    backtest.add_argument("--symbol", default="BTC-USD")
    backtest.add_argument("--phase", choices=("validation", "all"), default="validation")
    backtest.add_argument("--chart-type", choices=("crypto", "stock", "random"))
    backtest.add_argument("--chart-detail")
    backtest.add_argument("--bars-per-year", type=int, default=8760)
    backtest.add_argument("--initial-capital", type=float, default=10_000.0)
    backtest.add_argument("--csv", type=Path)
    backtest.add_argument("--synthetic", action="store_true")

    runs = subparsers.add_parser("runs", help="list recent runs")
    runs.add_argument("--limit", type=int, default=20)

    show = subparsers.add_parser("show", help="show a run as JSON")
    show.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        ensure_runtime_dirs()
        RunStore()
        print(f"database: {database_path()}")
        return 0

    if args.command == "ui":
        environment = os.environ.copy()
        source = str(ROOT / "src")
        environment["PYTHONPATH"] = (
            source if not environment.get("PYTHONPATH")
            else f"{source}:{environment['PYTHONPATH']}"
        )
        command = [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "src" / "trading_lab" / "ui" / "app.py"),
            *args.streamlit_args,
        ]
        return subprocess.call(command, cwd=ROOT, env=environment)

    store = RunStore()
    if args.command == "runs":
        for run in store.list_runs(args.limit):
            print(
                run.get("run_name") or run["run_id"],
                run["status"],
                run["strategy_id"],
                run["symbol"],
                run["created_at"],
            )
        return 0

    if args.command == "show":
        run = store.get_run(args.run_id)
        if run is None:
            print(f"run not found: {args.run_id}", file=sys.stderr)
            return 1
        print(json.dumps(run, indent=2, default=str))
        return 0

    service = BacktestService(store)
    run_id = service.run(BacktestRequest(
        strategy_id=args.strategy,
        symbol=args.symbol,
        phase=args.phase,
        chart_type=args.chart_type,
        chart_detail=args.chart_detail,
        bars_per_year=args.bars_per_year,
        initial_capital=args.initial_capital,
        csv_path=args.csv,
        synthetic=args.synthetic,
    ))
    run = store.get_run(run_id)
    print(json.dumps(run, indent=2, default=str))
    return 0 if run and run["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

