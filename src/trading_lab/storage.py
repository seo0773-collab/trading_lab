from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import RUN_STATUSES, RunRecord, utc_now
from .paths import database_path, ensure_runtime_dirs


SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    live_eligible INTEGER NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_number INTEGER,
    run_name TEXT,
    strategy_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    symbol TEXT NOT NULL,
    chart_type TEXT,
    chart_detail TEXT,
    phase TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, kind, path)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    broker_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    broker_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    payload_json TEXT,
    created_at TEXT NOT NULL
);
"""


class RunStore:
    def __init__(self, path: Path | None = None):
        ensure_runtime_dirs()
        self.path = path or database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            migrations = {
                "run_number": "ALTER TABLE runs ADD COLUMN run_number INTEGER",
                "run_name": "ALTER TABLE runs ADD COLUMN run_name TEXT",
                "chart_type": "ALTER TABLE runs ADD COLUMN chart_type TEXT",
                "chart_detail": "ALTER TABLE runs ADD COLUMN chart_detail TEXT",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)

    def register_strategy(
        self, strategy_id: str, version: str, description: str,
        *, enabled: bool, live_eligible: bool,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO strategies
                    (strategy_id, version, enabled, live_eligible, description)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    version=excluded.version,
                    enabled=excluded.enabled,
                    live_eligible=excluded.live_eligible,
                    description=excluded.description
                """,
                (strategy_id, version, int(enabled), int(live_eligible), description),
            )

    def create_run(self, record: RunRecord) -> None:
        if record.status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {record.status}")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, run_number, run_name, strategy_id, mode, status,
                    symbol, chart_type, chart_detail, phase,
                    created_at, started_at, finished_at, error, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id, record.run_number, record.run_name,
                    record.strategy_id, record.mode, record.status,
                    record.symbol, record.chart_type, record.chart_detail,
                    record.phase, record.created_at,
                    record.started_at, record.finished_at, record.error,
                    json.dumps(record.metrics) if record.metrics is not None else None,
                ),
            )

    def next_run_number(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(
                    COALESCE(run_number, (SELECT COUNT(*) FROM runs))
                ) AS current_number
                FROM runs
                """
            ).fetchone()
        return int(row["current_number"] or 0) + 1

    def update_status(
        self, run_id: str, status: str, *, error: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {status}")
        started_at = utc_now() if status == "running" else None
        finished_at = utc_now() if status in {"succeeded", "failed", "cancelled"} else None
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs SET
                    status=?,
                    started_at=COALESCE(started_at, ?),
                    finished_at=COALESCE(?, finished_at),
                    error=?,
                    metrics_json=COALESCE(?, metrics_json)
                WHERE run_id=?
                """,
                (
                    status, started_at, finished_at, error,
                    json.dumps(metrics) if metrics is not None else None,
                    run_id,
                ),
            )

    def add_artifact(self, run_id: str, kind: str, path: Path) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO artifacts (run_id, kind, path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, kind, str(path), utc_now()),
            )

    def add_event(
        self, run_id: str, event_type: str, message: str,
        *, level: str = "info", payload: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events
                    (run_id, level, event_type, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, level, event_type, message,
                    json.dumps(payload) if payload is not None else None,
                    utc_now(),
                ),
            )

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            artifacts = connection.execute(
                "SELECT kind, path, created_at FROM artifacts WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
            events = connection.execute(
                """
                SELECT level, event_type, message, payload_json, created_at
                FROM events WHERE run_id=? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        result = self._run_row(row)
        result["artifacts"] = [dict(item) for item in artifacts]
        result["events"] = [
            {
                **dict(item),
                "payload": json.loads(item["payload_json"])
                if item["payload_json"] else None,
            }
            for item in events
        ]
        return result

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        metrics_json = result.pop("metrics_json")
        result["metrics"] = json.loads(metrics_json) if metrics_json else None
        return result

