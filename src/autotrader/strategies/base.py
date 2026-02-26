"""Base interface for trading strategies."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autotrader.signals.base import Signal


class OrderUrgency(StrEnum):
    """Order execution urgency."""

    PASSIVE = "passive"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"


@dataclass
class ProposedOrder:
    """An order proposed by a strategy, pending risk approval."""

    strategy: str
    ticker: str
    side: str  # "yes" or "no"
    price_cents: int
    quantity: int
    urgency: OrderUrgency = OrderUrgency.PASSIVE
    rationale: str = ""
    max_time_in_force_seconds: int | None = None


class Strategy(abc.ABC):
    """Abstract base class for trading strategies."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique name of this strategy."""

    @property
    @abc.abstractmethod
    def enabled(self) -> bool:
        """Whether this strategy is currently active."""

    @property
    @abc.abstractmethod
    def target_series(self) -> list[str]:
        """Series tickers this strategy operates on."""

    @abc.abstractmethod
    async def initialize(self, market_data: Any, state: Any) -> None:
        """Initialize the strategy with market data and persisted state."""

    @abc.abstractmethod
    async def on_signal(self, signal: Signal) -> list[ProposedOrder]:
        """React to a signal from a signal source."""

    @abc.abstractmethod
    async def on_market_update(self, data: Any) -> list[ProposedOrder]:
        """React to market data updates (price changes, trades)."""

    @abc.abstractmethod
    async def on_fill(self, fill_data: Any) -> None:
        """Handle notification of a fill on one of this strategy's orders."""

    @abc.abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return serializable state for persistence."""

    @abc.abstractmethod
    async def teardown(self) -> None:
        """Clean up resources."""
