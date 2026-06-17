from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_frame(path: str) -> pd.DataFrame:
    artifact = Path(path)
    if artifact.suffix == ".parquet":
        return pd.read_parquet(artifact)
    return pd.read_csv(artifact, index_col=0, parse_dates=True)


def load_json(path: str | None) -> Any:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_path(run: dict[str, Any], kind: str | None) -> str | None:
    if not kind:
        return None
    return next(
        (item["path"] for item in run.get("artifacts", []) if item["kind"] == kind),
        None,
    )


def load_json_frame(run: dict[str, Any], kind: str | None) -> pd.DataFrame:
    path = artifact_path(run, kind)
    records = load_json(path) if path and Path(path).exists() else []
    return pd.DataFrame(records or [])


def run_inputs(run: dict[str, Any]) -> tuple[dict[str, Any], float]:
    config = load_json(artifact_path(run, "config"))
    manifest = load_json(artifact_path(run, "manifest"))
    metrics = run.get("metrics") or {}
    initial_capital = float(
        metrics.get("initial_capital", manifest.get("initial_capital", 10_000.0))
    )
    return config, initial_capital


def run_label(run: dict[str, Any]) -> str:
    if run.get("run_name"):
        return f"{run['run_name']} · {run['status']}"
    return (
        f"{run['created_at']} | {run['symbol']} | {run['status']} | "
        f"{run['run_id'][:8]}"
    )
