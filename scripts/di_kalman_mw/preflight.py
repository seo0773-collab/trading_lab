#!/usr/bin/env python
"""Validate real-data readiness without running a trading backtest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from di_kalman_mw.config import combo_config, sufficiency_rule  # noqa: E402
from di_kalman_mw.extreme_transition import (  # noqa: E402
    build_pattern_dataset, completed_instances_for_split,
)
from di_kalman_mw.run import load_data  # noqa: E402

MIN_COMPLETE_PATTERNS = {
    "1h": 500,
    "4h": 300,
    "1d": 100,
}


def _timeframe(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def inspect_dataset(path: Path) -> dict:
    timeframe = _timeframe(path)
    df = load_data(path)
    cfg = combo_config("A")
    dataset = build_pattern_dataset(df, cfg, cfg.patterns.strict)
    completed = completed_instances_for_split(
        dataset.instances, dataset.labels, "train"
    )
    train_index = df.index[dataset.labels == "train"]
    years = 0.0
    if len(train_index) >= 2:
        years = (
            (train_index[-1] - train_index[0]).total_seconds()
            / (365.25 * 24 * 3600.0)
        )
    rule = sufficiency_rule(timeframe)
    min_patterns = MIN_COMPLETE_PATTERNS.get(timeframe, 100)
    period_ok = years >= float(rule["min_years"])
    patterns_ok = len(completed) >= min_patterns
    return {
        "path": str(path),
        "timeframe": timeframe,
        "rows": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "train_rows": len(train_index),
        "train_years": round(years, 2),
        "complete_train_p5_patterns": len(completed),
        "minimum_train_years": rule["min_years"],
        "minimum_complete_patterns": min_patterns,
        "period_ok": period_ok,
        "patterns_ok": patterns_ok,
        "ready": period_ok and patterns_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "di_kalman_mw" / "preflight.json",
    )
    args = parser.parse_args(argv)

    files = sorted(args.data_dir.glob("*USDT_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no OHLCV parquet files in {args.data_dir}")
    reports = [inspect_dataset(path) for path in files]
    summary = {
        "datasets": reports,
        "ready_count": sum(item["ready"] for item in reports),
        "total_count": len(reports),
        "all_ready": all(item["ready"] for item in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    for item in reports:
        print(
            f"{Path(item['path']).name}: ready={item['ready']} "
            f"train_years={item['train_years']} "
            f"complete_p5={item['complete_train_p5_patterns']}"
        )
    print(
        f"ready={summary['ready_count']}/{summary['total_count']} "
        f"report={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
