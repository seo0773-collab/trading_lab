from __future__ import annotations

import json
from pathlib import Path

from trading_lab.strategies import get_strategy


TF_OPTIONS = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
]
PERIOD_OPTIONS = [
    "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y",
    "10y", "ytd", "max",
]


def strategy_config_dict(strategy_id: str) -> dict:
    path = get_strategy(strategy_id).config_path
    return json.loads(Path(path).read_text(encoding="utf-8"))
