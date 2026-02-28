"""Risk management gate for the autotrader.

Every :class:`ProposedOrder` produced by a strategy passes through the
:class:`RiskManager` before reaching the execution engine.  The manager
runs a sequence of independent checks; if **any** check fails the order
is rejected and a :class:`~autotrader.state.models.RiskEvent` is logged.

Checks implemented
------------------
1. **kill_switch**        — reject everything when the kill switch is on.
2. **position_per_contract** — per-strategy max position in a single contract.
3. **position_per_event**    — per-strategy max exposure across an event.
4. **daily_loss**            — per-strategy max realised + unrealised loss for
   the current day.
5. **portfolio_exposure**    — global max percentage of balance at risk.
6. **price_sanity**          — reject orders with prices outside 1–99 ¢.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from autotrader.config.models import RiskConfig
    from autotrader.strategies.base import ProposedOrder

logger = structlog.get_logger("autotrader.risk.manager")


# ── Result types ─────────────────────────────────────────────────────────


class RiskVerdict(StrEnum):
    """Outcome of a risk check."""

    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of a single risk check."""

    check_name: str
    verdict: RiskVerdict
    reason: str = ""


@dataclass(frozen=True)
class RiskDecision:
    """Aggregate risk decision for a proposed order."""

    order: ProposedOrder
    verdict: RiskVerdict
    results: list[RiskCheckResult] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED

    @property
    def rejection_reasons(self) -> list[str]:
        return [r.reason for r in self.results if r.verdict == RiskVerdict.REJECTED]


# ── Portfolio state ──────────────────────────────────────────────────────


@dataclass
class PositionInfo:
    """Snapshot of a single contract position."""

    ticker: str
    event_ticker: str
    quantity: int  # Positive = long YES
    avg_cost_cents: float = 0.0
    unrealized_pnl_cents: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state fed into the risk manager.

    The run-loop is responsible for building this from the database /
    exchange state and passing it in on each evaluation cycle.
    """

    balance_cents: int = 0
    positions: list[PositionInfo] = field(default_factory=list)
    daily_realized_pnl_cents: dict[str, int] = field(default_factory=dict)  # strategy → cents
    daily_unrealized_pnl_cents: dict[str, int] = field(default_factory=dict)  # strategy → cents


# ── Risk Manager ─────────────────────────────────────────────────────────


class RiskManager:
    """Validates proposed orders against risk limits.

    Usage::

        rm = RiskManager(config)
        rm.update_portfolio(snapshot)

        for order in proposed_orders:
            decision = rm.evaluate(order)
            if decision.approved:
                send_to_execution(order)
            else:
                log_rejection(decision)
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._portfolio = PortfolioSnapshot()
        self._kill_switch = config.global_config.kill_switch_enabled

    # ── Public API ────────────────────────────────────────────────────

    def update_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Refresh the portfolio state used by risk checks."""
        self._portfolio = snapshot

    def evaluate(self, order: ProposedOrder) -> RiskDecision:
        """Run all risk checks against a proposed order.

        Returns a :class:`RiskDecision` with the aggregate verdict.
        Any single rejection causes the whole order to be rejected.
        """
        results: list[RiskCheckResult] = [
            self._check_kill_switch(),
            self._check_price_sanity(order),
            self._check_position_per_contract(order),
            self._check_position_per_event(order),
            self._check_daily_loss(order),
            self._check_portfolio_exposure(order),
        ]

        rejected = any(r.verdict == RiskVerdict.REJECTED for r in results)
        verdict = RiskVerdict.REJECTED if rejected else RiskVerdict.APPROVED

        decision = RiskDecision(order=order, verdict=verdict, results=results)

        if rejected:
            logger.warning(
                "order_rejected",
                strategy=order.strategy,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                reasons=decision.rejection_reasons,
            )
        else:
            logger.debug(
                "order_approved",
                strategy=order.strategy,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
            )

        return decision

    def evaluate_batch(self, orders: list[ProposedOrder]) -> list[RiskDecision]:
        """Evaluate a batch of proposed orders."""
        return [self.evaluate(o) for o in orders]

    def activate_kill_switch(self, reason: str = "") -> None:
        """Activate the kill switch — all future orders will be rejected."""
        self._kill_switch = True
        logger.critical("kill_switch_activated", reason=reason)

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch."""
        self._kill_switch = False
        logger.info("kill_switch_deactivated")

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    @property
    def portfolio(self) -> PortfolioSnapshot:
        return self._portfolio

    def serialize_order(self, order: ProposedOrder) -> dict[str, Any]:
        """Convert a ProposedOrder to a dict for RiskEvent persistence."""
        return {
            "strategy": order.strategy,
            "ticker": order.ticker,
            "side": order.side,
            "price_cents": order.price_cents,
            "quantity": order.quantity,
            "urgency": str(order.urgency),
            "rationale": order.rationale,
        }

    # ── Individual checks ─────────────────────────────────────────────

    def _check_kill_switch(self) -> RiskCheckResult:
        if self._kill_switch:
            return RiskCheckResult(
                check_name="kill_switch",
                verdict=RiskVerdict.REJECTED,
                reason="Kill switch is active — all orders blocked",
            )
        return RiskCheckResult(check_name="kill_switch", verdict=RiskVerdict.APPROVED)

    def _check_price_sanity(self, order: ProposedOrder) -> RiskCheckResult:
        if order.price_cents < 1 or order.price_cents > 99:
            return RiskCheckResult(
                check_name="price_sanity",
                verdict=RiskVerdict.REJECTED,
                reason=f"Price {order.price_cents}c outside valid range 1-99",
            )
        if order.quantity < 1:
            return RiskCheckResult(
                check_name="price_sanity",
                verdict=RiskVerdict.REJECTED,
                reason=f"Quantity {order.quantity} must be >= 1",
            )
        return RiskCheckResult(check_name="price_sanity", verdict=RiskVerdict.APPROVED)

    def _check_position_per_contract(self, order: ProposedOrder) -> RiskCheckResult:
        strategy_config = self._config.per_strategy.get(order.strategy)
        if strategy_config is None:
            return RiskCheckResult(check_name="position_per_contract", verdict=RiskVerdict.APPROVED)

        max_pos = strategy_config.max_position_per_contract
        current_pos = self._current_position_for_ticker(order.ticker)
        signed_delta = self._signed_position_delta(order)

        new_pos = current_pos + signed_delta
        if abs(new_pos) > max_pos:
            return RiskCheckResult(
                check_name="position_per_contract",
                verdict=RiskVerdict.REJECTED,
                reason=(
                    f"Would exceed per-contract limit: "
                    f"current={current_pos}, delta={signed_delta}, projected={new_pos}, max={max_pos:.0f}"
                ),
            )
        return RiskCheckResult(check_name="position_per_contract", verdict=RiskVerdict.APPROVED)

    def _check_position_per_event(self, order: ProposedOrder) -> RiskCheckResult:
        strategy_config = self._config.per_strategy.get(order.strategy)
        if strategy_config is None:
            return RiskCheckResult(check_name="position_per_event", verdict=RiskVerdict.APPROVED)

        max_event_pos = strategy_config.max_position_per_event
        event_ticker = self._event_ticker_for(order.ticker)
        current_event_pos = self._current_position_for_event(event_ticker)
        signed_delta = self._signed_position_delta(order)

        new_event_pos = current_event_pos + signed_delta
        if abs(new_event_pos) > max_event_pos:
            return RiskCheckResult(
                check_name="position_per_event",
                verdict=RiskVerdict.REJECTED,
                reason=(
                    f"Would exceed per-event limit: "
                    f"event={event_ticker}, current={current_event_pos}, "
                    f"delta={signed_delta}, projected={new_event_pos}, max={max_event_pos:.0f}"
                ),
            )
        return RiskCheckResult(check_name="position_per_event", verdict=RiskVerdict.APPROVED)

    def _check_daily_loss(self, order: ProposedOrder) -> RiskCheckResult:
        strategy_config = self._config.per_strategy.get(order.strategy)
        if strategy_config is None:
            return RiskCheckResult(check_name="daily_loss", verdict=RiskVerdict.APPROVED)

        max_loss = strategy_config.max_strategy_loss
        realized = self._portfolio.daily_realized_pnl_cents.get(order.strategy, 0)
        unrealized = self._portfolio.daily_unrealized_pnl_cents.get(order.strategy, 0)
        total_pnl_dollars = (realized + unrealized) / 100.0

        # If already in a loss exceeding the limit, reject new orders
        if total_pnl_dollars < 0 and abs(total_pnl_dollars) >= max_loss:
            return RiskCheckResult(
                check_name="daily_loss",
                verdict=RiskVerdict.REJECTED,
                reason=(f"Strategy daily loss limit reached: pnl=${total_pnl_dollars:.2f}, max_loss=${max_loss:.2f}"),
            )
        return RiskCheckResult(check_name="daily_loss", verdict=RiskVerdict.APPROVED)

    def _check_portfolio_exposure(self, order: ProposedOrder) -> RiskCheckResult:
        max_pct = self._config.global_config.max_portfolio_exposure_pct
        balance = self._portfolio.balance_cents

        if balance <= 0:
            return RiskCheckResult(
                check_name="portfolio_exposure",
                verdict=RiskVerdict.REJECTED,
                reason=(f"Cannot evaluate portfolio exposure: invalid balance={balance}c"),
            )

        current_exposure = self._total_exposure_cents()
        current_pos = self._current_position_for_ticker(order.ticker)
        new_pos = current_pos + self._signed_position_delta(order)
        current_contract_exposure = self._ticker_exposure_cents(order.ticker)
        current_abs_pos = abs(current_pos)
        new_abs_pos = abs(new_pos)

        if current_abs_pos == 0:
            new_contract_exposure = new_abs_pos * order.price_cents
        elif new_abs_pos <= current_abs_pos:
            # Reduce existing exposure at existing blended carrying cost.
            new_contract_exposure = round(current_contract_exposure * (new_abs_pos / current_abs_pos))
        else:
            # Existing carrying exposure plus incremental size at the new order price.
            added_abs_pos = new_abs_pos - current_abs_pos
            new_contract_exposure = current_contract_exposure + (added_abs_pos * order.price_cents)

        new_exposure = current_exposure - current_contract_exposure + new_contract_exposure
        exposure_pct = new_exposure / balance

        if exposure_pct > max_pct:
            return RiskCheckResult(
                check_name="portfolio_exposure",
                verdict=RiskVerdict.REJECTED,
                reason=(
                    f"Would exceed portfolio exposure limit: "
                    f"new_exposure={new_exposure}c ({exposure_pct:.1%}), "
                    f"max={max_pct:.0%}, balance={balance}c"
                ),
            )
        return RiskCheckResult(check_name="portfolio_exposure", verdict=RiskVerdict.APPROVED)

    # ── Portfolio queries ─────────────────────────────────────────────

    def _current_position_for_ticker(self, ticker: str) -> int:
        """Sum of all position quantities for a given contract ticker."""
        return sum(p.quantity for p in self._portfolio.positions if p.ticker == ticker)

    def _event_ticker_for(self, ticker: str) -> str:
        """Find the event ticker for a contract.

        Looks up in portfolio positions first; if not found, uses the
        contract ticker itself as a fallback.
        """
        for p in self._portfolio.positions:
            if p.ticker == ticker:
                return p.event_ticker
        return ticker

    def _current_position_for_event(self, event_ticker: str) -> int:
        """Sum of all position quantities across an event."""
        return sum(p.quantity for p in self._portfolio.positions if p.event_ticker == event_ticker)

    def _total_exposure_cents(self) -> int:
        """Total gross economic exposure across contracts.

        Positions are tracked as signed YES-equivalent quantities
        (positive=long YES, negative=long NO). Exposure is gross, so a
        long YES and long NO position both contribute positively.
        """
        return sum(abs(p.quantity) * int(p.avg_cost_cents) for p in self._portfolio.positions)

    def _ticker_exposure_cents(self, ticker: str) -> int:
        """Gross economic exposure attributable to a single ticker."""
        return sum(abs(p.quantity) * int(p.avg_cost_cents) for p in self._portfolio.positions if p.ticker == ticker)

    def _signed_position_delta(self, order: ProposedOrder) -> int:
        """Convert an order into signed YES-equivalent position delta.

        Follows the same convention as ``LeaderboardAlphaStrategy.on_fill``:
        buying YES increases position, buying NO decreases position.
        """
        return order.quantity if order.side.lower() == "yes" else -order.quantity
