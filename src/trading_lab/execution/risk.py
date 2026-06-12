from __future__ import annotations

from abc import ABC, abstractmethod

from .broker import OrderRequest, Position


class RiskPolicy(ABC):
    @abstractmethod
    def validate(self, request: OrderRequest, positions: list[Position]) -> None:
        """Raise an exception when an order violates configured risk limits."""
        raise NotImplementedError
