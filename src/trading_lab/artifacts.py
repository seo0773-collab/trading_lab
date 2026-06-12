from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import runs_dir


class ArtifactWriter:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.directory = runs_dir() / run_id
        self.directory.mkdir(parents=True, exist_ok=False)

    def write_json(self, name: str, value: Any) -> Path:
        path = self.directory / name
        path.write_text(
            json.dumps(value, indent=2, default=str, allow_nan=True),
            encoding="utf-8",
        )
        return path

    def write_text(self, name: str, value: str) -> Path:
        path = self.directory / name
        path.write_text(value, encoding="utf-8")
        return path

    def write_frame(self, name: str, frame: pd.DataFrame) -> Path:
        parquet = self.directory / f"{name}.parquet"
        try:
            frame.to_parquet(parquet)
            return parquet
        except ImportError:
            csv = self.directory / f"{name}.csv"
            frame.to_csv(csv, index=True)
            return csv

