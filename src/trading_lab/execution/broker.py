from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class OrderStatus(StrEnum):
    CREATED = "created"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_price: float


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, request: OrderRequest) -> str:
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> list[Position]:
        raise NotImplementedError


class DisabledBrokerAdapter(BrokerAdapter):
    def submit_order(self, request: OrderRequest) -> str:
        raise RuntimeError(
            "live trading is disabled: no broker adapter is configured"
        )

    def positions(self) -> list[Position]:
        return []

