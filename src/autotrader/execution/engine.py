"""Order execution engine.

Bridges risk-approved :class:`ProposedOrder` objects to the Kalshi API
(live mode) or a local paper-trading simulator (paper mode).

Responsibilities:
- Convert ``ProposedOrder`` → API ``OrderRequest``
- Generate unique client order IDs
- Submit orders and track their lifecycle
- Paper-trade simulation (instant fills at proposed price)
- Emit fill callbacks for strategy position tracking
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from autotrader.api.client import KalshiAPIClient, KalshiAPIError, OrderRequest
from autotrader.config.models import ExecutionMode as ExecutionMode
from autotrader.utils.fees import FeeCalculator

if TYPE_CHECKING:
    from autotrader.strategies.base import ProposedOrder

logger = structlog.get_logger("autotrader.execution.engine")


# ── Types ────────────────────────────────────────────────────────────────


class OrderStatus(StrEnum):
    """Lifecycle status of a tracked order."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    RESTING = "resting"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TrackedOrder:
    """An order tracked through its lifecycle by the engine."""

    client_order_id: str
    kalshi_order_id: str | None
    ticker: str
    side: str
    price_cents: int
    quantity: int
    filled_quantity: int
    status: OrderStatus
    strategy: str
    urgency: str
    rationale: str
    is_paper: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime
    filled_at: datetime.datetime | None = None
    cancelled_at: datetime.datetime | None = None


@dataclass
class ExecutionResult:
    """Result of submitting a single order."""

    success: bool
    order: TrackedOrder
    error: str = ""


@dataclass
class FillEvent:
    """A fill to be reported back to the strategy."""

    ticker: str
    side: str
    action: str  # "buy"
    count: int
    price_cents: int
    fee_cents: int
    is_taker: bool
    is_paper: bool
    client_order_id: str
    kalshi_fill_id: str | None = None
    filled_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


# Type alias for fill callbacks: strategy.on_fill receives fill dicts
FillCallback = Callable[[dict[str, Any]], Any]


# ── Execution Engine ─────────────────────────────────────────────────────


class ExecutionEngine:
    """Submits orders to Kalshi or simulates them in paper mode.

    Usage::

        engine = ExecutionEngine(mode=ExecutionMode.PAPER)
        engine.on_fill(my_callback)

        for order in risk_approved_orders:
            result = await engine.submit(order)
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        api_client: KalshiAPIClient | None = None,
        fee_calculator: FeeCalculator | None = None,
    ) -> None:
        self._mode = mode
        self._api = api_client
        self._fee_calc = fee_calculator or FeeCalculator()
        self._fill_callbacks: list[FillCallback] = []
        self._orders: dict[str, TrackedOrder] = {}  # client_order_id → TrackedOrder

        # Terminal orders older than this many entries are evicted to prevent
        # unbounded memory growth during long-running 24/7 operation.
        self._max_terminal_orders = 500

        if mode == ExecutionMode.LIVE and api_client is None:
            raise ValueError("Live mode requires an api_client")

    # ── Public API ────────────────────────────────────────────────────

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def orders(self) -> dict[str, TrackedOrder]:
        """All tracked orders keyed by client_order_id."""
        return dict(self._orders)

    def on_fill(self, callback: FillCallback) -> None:
        """Register a callback to be invoked on each fill."""
        self._fill_callbacks.append(callback)

    async def submit(self, proposed: ProposedOrder) -> ExecutionResult:
        """Submit a risk-approved order for execution.

        In paper mode, fills are simulated instantly.
        In live mode, the order is sent to the Kalshi API.
        """
        client_order_id = self._generate_order_id(proposed)
        now = datetime.datetime.utcnow()

        tracked = TrackedOrder(
            client_order_id=client_order_id,
            kalshi_order_id=None,
            ticker=proposed.ticker,
            side=proposed.side,
            price_cents=proposed.price_cents,
            quantity=proposed.quantity,
            filled_quantity=0,
            status=OrderStatus.PENDING,
            strategy=proposed.strategy,
            urgency=str(proposed.urgency),
            rationale=proposed.rationale,
            is_paper=self._mode == ExecutionMode.PAPER,
            created_at=now,
            updated_at=now,
        )
        self._orders[client_order_id] = tracked

        if self._mode == ExecutionMode.PAPER:
            result = self._execute_paper(tracked)
        else:
            result = self._execute_live(tracked)
        self._evict_terminal_orders()
        return result

    async def submit_batch(self, orders: list[ProposedOrder]) -> list[ExecutionResult]:
        """Submit multiple orders. Returns results in the same order."""
        return [await self.submit(o) for o in orders]

    async def cancel(self, client_order_id: str) -> bool:
        """Cancel a resting order. Returns True if successful."""
        tracked = self._orders.get(client_order_id)
        if tracked is None:
            logger.warning("cancel_unknown_order", client_order_id=client_order_id)
            return False

        if tracked.status not in (OrderStatus.SUBMITTED, OrderStatus.RESTING, OrderStatus.PARTIAL):
            logger.warning(
                "cancel_invalid_status",
                client_order_id=client_order_id,
                status=tracked.status,
            )
            return False

        if self._mode == ExecutionMode.PAPER:
            tracked.status = OrderStatus.CANCELLED
            tracked.cancelled_at = datetime.datetime.utcnow()
            tracked.updated_at = tracked.cancelled_at
            logger.info("paper_order_cancelled", client_order_id=client_order_id)
            return True

        # Live cancel
        if tracked.kalshi_order_id is None:
            logger.error("cancel_no_kalshi_id", client_order_id=client_order_id)
            return False
        try:
            self._api.cancel_order(tracked.kalshi_order_id)  # type: ignore[union-attr]
            tracked.status = OrderStatus.CANCELLED
            tracked.cancelled_at = datetime.datetime.utcnow()
            tracked.updated_at = tracked.cancelled_at
            logger.info(
                "live_order_cancelled",
                client_order_id=client_order_id,
                kalshi_order_id=tracked.kalshi_order_id,
            )
            return True
        except KalshiAPIError as e:
            logger.error("cancel_failed", client_order_id=client_order_id, error=str(e))
            return False

    def get_open_orders(self) -> list[TrackedOrder]:
        """Return all orders that are still open (submitted/resting/partial)."""
        return [
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.RESTING, OrderStatus.PARTIAL)
        ]

    def get_orders_for_strategy(self, strategy: str) -> list[TrackedOrder]:
        """Return all orders for a given strategy."""
        return [o for o in self._orders.values() if o.strategy == strategy]

    # ── Paper execution ───────────────────────────────────────────────

    def _execute_paper(self, tracked: TrackedOrder) -> ExecutionResult:
        """Simulate an instant fill at the proposed price."""
        now = datetime.datetime.utcnow()

        # Calculate fees
        fee_result = self._fee_calc.taker_fee(tracked.price_cents, tracked.quantity)

        # Instant full fill
        tracked.status = OrderStatus.FILLED
        tracked.filled_quantity = tracked.quantity
        tracked.filled_at = now
        tracked.updated_at = now

        logger.info(
            "paper_order_filled",
            client_order_id=tracked.client_order_id,
            ticker=tracked.ticker,
            side=tracked.side,
            price=tracked.price_cents,
            quantity=tracked.quantity,
            fee=fee_result.total_fee_cents,
        )

        # Emit fill event
        fill = FillEvent(
            ticker=tracked.ticker,
            side=tracked.side,
            action="buy",
            count=tracked.quantity,
            price_cents=tracked.price_cents,
            fee_cents=fee_result.total_fee_cents,
            is_taker=True,
            is_paper=True,
            client_order_id=tracked.client_order_id,
            filled_at=now,
        )
        self._emit_fill(fill)

        return ExecutionResult(success=True, order=tracked)

    # ── Live execution ────────────────────────────────────────────────

    def _execute_live(self, tracked: TrackedOrder) -> ExecutionResult:
        """Submit order to the Kalshi API."""
        if self._api is None:
            tracked.status = OrderStatus.REJECTED
            tracked.updated_at = datetime.datetime.utcnow()
            return ExecutionResult(success=False, order=tracked, error="No API client")

        # Build the API request
        request = self._build_order_request(tracked)

        try:
            result = self._api.place_order(request)
            order_data = result.get("order", {})
            tracked.kalshi_order_id = order_data.get("order_id")
            tracked.status = OrderStatus.SUBMITTED
            tracked.updated_at = datetime.datetime.utcnow()

            logger.info(
                "live_order_submitted",
                client_order_id=tracked.client_order_id,
                kalshi_order_id=tracked.kalshi_order_id,
                ticker=tracked.ticker,
            )
            return ExecutionResult(success=True, order=tracked)

        except KalshiAPIError as e:
            tracked.status = OrderStatus.REJECTED
            tracked.updated_at = datetime.datetime.utcnow()
            logger.error(
                "live_order_rejected",
                client_order_id=tracked.client_order_id,
                error=str(e),
                status_code=e.status_code,
            )
            return ExecutionResult(success=False, order=tracked, error=str(e))

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_order_request(self, tracked: TrackedOrder) -> OrderRequest:
        """Convert a tracked order into a Kalshi API OrderRequest."""
        if tracked.side == "yes":
            return OrderRequest(
                ticker=tracked.ticker,
                side="yes",
                action="buy",
                type="limit",
                count=tracked.quantity,
                yes_price=tracked.price_cents,
                client_order_id=tracked.client_order_id,
            )
        else:
            return OrderRequest(
                ticker=tracked.ticker,
                side="no",
                action="buy",
                type="limit",
                count=tracked.quantity,
                no_price=tracked.price_cents,
                client_order_id=tracked.client_order_id,
            )

    @staticmethod
    def _generate_order_id(proposed: ProposedOrder) -> str:
        """Generate a unique client order ID."""
        short_uuid = uuid.uuid4().hex[:12]
        return f"{proposed.strategy}-{proposed.ticker}-{short_uuid}"

    _TERMINAL_STATUSES = frozenset({OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED})

    def _evict_terminal_orders(self) -> None:
        """Remove old terminal orders to prevent unbounded memory growth."""
        terminal = [
            oid for oid, o in self._orders.items() if o.status in self._TERMINAL_STATUSES
        ]
        excess = len(terminal) - self._max_terminal_orders
        if excess <= 0:
            return
        # Sort by updated_at so we evict the oldest first
        terminal.sort(key=lambda oid: self._orders[oid].updated_at)
        for oid in terminal[:excess]:
            del self._orders[oid]

    def _emit_fill(self, fill: FillEvent) -> None:
        """Notify all registered callbacks of a fill."""
        fill_dict: dict[str, Any] = {
            "ticker": fill.ticker,
            "side": fill.side,
            "action": fill.action,
            "count": fill.count,
            "price_cents": fill.price_cents,
            "fee_cents": fill.fee_cents,
            "is_taker": fill.is_taker,
            "is_paper": fill.is_paper,
            "client_order_id": fill.client_order_id,
            "kalshi_fill_id": fill.kalshi_fill_id,
            "filled_at": fill.filled_at.isoformat(),
        }
        for cb in self._fill_callbacks:
            try:
                cb(fill_dict)
            except Exception:
                logger.exception("fill_callback_error", client_order_id=fill.client_order_id)
