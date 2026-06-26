from __future__ import annotations

from typing import Any

import pandas as pd


def metric_text(value: Any, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%" if percent else f"{float(value):.2f}"


def money_text(value: Any, symbol: str = "$", decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{symbol}{float(value):,.{decimals}f}"


# base_currency → (기호, 소수자리, number_input step, 기본 초기계좌)
_CURRENCY = {
    "KRW": ("₩", 0, 1_000_000.0, 10_000_000.0),
    "USD": ("$", 2, 1_000.0, 10_000.0),
    "EUR": ("€", 2, 1_000.0, 10_000.0),
    "JPY": ("¥", 0, 100_000.0, 1_000_000.0),
}


def currency_spec(base_currency: Any) -> tuple[str, int, float, float]:
    """전략 config의 base_currency로 (기호, 소수자리, 입력 step, 기본 초기계좌)."""
    return _CURRENCY.get(str(base_currency or "USD").upper(), ("$", 2, 1_000.0, 10_000.0))
