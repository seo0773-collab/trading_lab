from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_lab.paths import ROOT


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    version: str
    description: str
    config_path: Path
    enabled: bool
    live_eligible: bool


_STRATEGIES = {
    "h72-price-v1": StrategyDefinition(
        strategy_id="h72-price-v1",
        version="1",
        description="Adaptive Kalman 72-bar PRICE direction strategy",
        config_path=ROOT / "configs" / "strategies" / "h72_price_v1.json",
        enabled=True,
        live_eligible=False,
    ),
}


def list_strategies() -> list[StrategyDefinition]:
    return list(_STRATEGIES.values())


def get_strategy(strategy_id: str) -> StrategyDefinition:
    try:
        return _STRATEGIES[strategy_id]
    except KeyError as exc:
        raise KeyError(f"unknown strategy: {strategy_id}") from exc

