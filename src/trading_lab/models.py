from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


RUN_STATUSES = {"created", "running", "succeeded", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    strategy_id: str
    mode: str
    status: str
    symbol: str
    phase: str
    created_at: str
    run_number: int | None = None
    run_name: str | None = None
    chart_type: str | None = None
    chart_detail: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

