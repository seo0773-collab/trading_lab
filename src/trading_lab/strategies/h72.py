"""Handler for the h72-price-v1 strategy.

Thin wrapper over the existing research_adapter pipeline so behaviour is
unchanged; it only repackages the outputs as StrategyArtifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_lab.research_adapter import (
    build_forecast,
    execute_strategy,
    load_market_data,
)
from trading_lab.strategies.base import StrategyArtifacts


class H72Handler:
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        return load_market_data(
            symbol, config, csv_path=csv_path, synthetic=synthetic
        )

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        forecast, metadata = build_forecast(raw, config)
        execution, metrics = execute_strategy(
            forecast,
            config,
            symbol=symbol,
            phase=phase,
            bars_per_year=bars_per_year,
        )
        return StrategyArtifacts(
            forecast=forecast,
            trades=execution.trades,
            equity=execution.equity,
            metrics=metrics,
            metadata=metadata,
            horizon=int(config.get("horizon", 0)),
        )
