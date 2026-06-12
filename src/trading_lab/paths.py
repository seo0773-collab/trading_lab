from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def var_dir() -> Path:
    return Path(os.environ.get("TRADING_LAB_VAR", ROOT / "var")).resolve()


def runs_dir() -> Path:
    return var_dir() / "runs"


def database_path() -> Path:
    return var_dir() / "trading_lab.sqlite3"


def ensure_runtime_dirs() -> None:
    for path in (var_dir(), runs_dir(), var_dir() / "logs", var_dir() / "data"):
        path.mkdir(parents=True, exist_ok=True)

