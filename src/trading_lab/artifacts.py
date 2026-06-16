from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import runs_dir


class ArtifactWriter:
    def __init__(self, run_id: str, *, dir_name: str | None = None):
        self.run_id = run_id
        # 사람이 읽을 수 있는 결과 폴더명(run_name)으로 저장하되, 미지정 시
        # run_id로 폴백한다. run_number가 앞에 붙어 고유성이 보장된다.
        self.directory = runs_dir() / (dir_name or run_id)
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

