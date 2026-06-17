"""Strategy handler protocol decoupling BacktestService from any one pipeline.

A handler loads market data and turns it into a uniform ``StrategyArtifacts``
bundle that the common dashboard renders. This lets every strategy run through
the shared "새 백테스트 → 결과" flow instead of a bespoke page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass
class StrategyArtifacts:
    """Uniform output every strategy handler must produce.

    - ``forecast``: time-indexed frame with at least ``close`` (OHLC optional)
      plus indicator columns. ``presentation.indicator_series`` auto-discovers
      every non-OHLC numeric column for the waveform panel.
    - ``trades``: canonical schema consumed by ``presentation``:
      ``direction`` (1/-1), ``entry_time``, ``entry_price``, ``exit_time``,
      ``exit_price``, ``net_return`` (fraction), ``exit_reason``; optional
      ``stop_loss_price`` / ``take_profit_price`` / ``entry_reason``.
    - ``equity``: 1.0-based cumulative growth series.
    - ``metrics``: at least ``trades``, ``hit_rate``, ``total_return``,
      ``sharpe``, ``max_drawdown`` (dashboard metric row + service enrichment).
    - ``benchmark``: optional 1.0-based buy & hold growth series for account
      chart comparison, mainly for portfolio handlers where one price column is
      not enough to derive a fair benchmark in the UI.
    - ``extras``: optional render hints, e.g. ``{"split_breakdown": DataFrame}``
      shown as an expander in the result view.
    """

    forecast: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    horizon: int = 0
    benchmark: pd.Series | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class StrategyHandler(Protocol):
    """Per-strategy data loading + artifact construction."""

    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        ...

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        ...
