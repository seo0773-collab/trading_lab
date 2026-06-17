from __future__ import annotations

from typing import Any

import pandas as pd


def metric_text(value: Any, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%" if percent else f"{float(value):.2f}"


def money_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.2f}"
