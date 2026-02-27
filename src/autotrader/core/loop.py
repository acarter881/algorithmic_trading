"""Main run loop that orchestrates all autotrader components.

The :class:`TradingLoop` ties together:

- **ArenaMonitor** — polls the leaderboard for signals
- **LeaderboardAlphaStrategy** — converts signals into trade proposals
- **RiskManager** — gates proposals against risk limits
- **ExecutionEngine** — submits approved orders (paper or live)

The loop runs on a configurable interval (default: arena poll interval)
and supports graceful shutdown via :meth:`stop`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from autotrader.execution.engine import ExecutionEngine, ExecutionMode
from autotrader.risk.manager import PortfolioSnapshot, PositionInfo, RiskManager
from autotrader.signals.arena_monitor import ArenaMonitor
from autotrader.strategies.leaderboard_alpha import LeaderboardAlphaStrategy
from autotrader.utils.fees import FeeCalculator

if TYPE_CHECKING:
    from autotrader.config.models import AppConfig
    from autotrader.strategies.base import ProposedOrder

logger = structlog.get_logger("autotrader.core.loop")


class TradingLoop:
    """Async run loop that orchestrates the full trading pipeline.

    Usage::

        loop = TradingLoop(config)
        await loop.initialize()
        await loop.run()       # blocks until stop() is called
        await loop.shutdown()
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._running = False
        self._tick_count = 0

        # Components — created in initialize()
        self._monitor: ArenaMonitor | None = None
        self._strategy: LeaderboardAlphaStrategy | None = None
        self._risk: RiskManager | None = None
        self._engine: ExecutionEngine | None = None
        self._fee_calc = FeeCalculator()

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(self, market_data: dict[str, Any] | None = None) -> None:
        """Create and initialize all components."""
        # Arena monitor
        self._monitor = ArenaMonitor(config=self._config.arena_monitor)
        await self._monitor.initialize()

        # Strategy
        self._strategy = LeaderboardAlphaStrategy(
            config=self._config.leaderboard_alpha,
            fee_calculator=self._fee_calc,
        )
        await self._strategy.initialize(market_data or {}, None)

        # Risk manager
        self._risk = RiskManager(config=self._config.risk)

        # Execution engine
        mode = ExecutionMode.LIVE if self._config.kalshi.environment.value == "production" else ExecutionMode.PAPER
        self._engine = ExecutionEngine(mode=mode, fee_calculator=self._fee_calc)

        # Wire fill callbacks: engine fills → strategy position tracking
        self._engine.on_fill(self._on_fill)

        logger.info(
            "trading_loop_initialized",
            mode=mode.value,
            poll_interval=self._config.arena_monitor.poll_interval_seconds,
            target_series=self._config.leaderboard_alpha.target_series,
        )

    async def run(self) -> None:
        """Start the poll loop.  Blocks until :meth:`stop` is called."""
        if self._monitor is None:
            raise RuntimeError("Call initialize() before run()")

        self._running = True
        interval = self._config.arena_monitor.poll_interval_seconds
        logger.info("trading_loop_started", interval_seconds=interval)

        while self._running:
            await self._tick()
            await asyncio.sleep(interval)

        logger.info("trading_loop_stopped", total_ticks=self._tick_count)

    def stop(self) -> None:
        """Signal the loop to stop after the current tick."""
        self._running = False
        logger.info("trading_loop_stop_requested")

    async def shutdown(self) -> None:
        """Tear down all components."""
        if self._monitor:
            await self._monitor.teardown()
        if self._strategy:
            await self._strategy.teardown()
        logger.info("trading_loop_shutdown")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def strategy(self) -> LeaderboardAlphaStrategy | None:
        return self._strategy

    @property
    def risk_manager(self) -> RiskManager | None:
        return self._risk

    @property
    def execution_engine(self) -> ExecutionEngine | None:
        return self._engine

    # ── Core tick ─────────────────────────────────────────────────────

    async def _tick(self) -> None:
        """Execute one poll → signal → propose → risk → execute cycle.

        All exceptions are caught and logged so that the loop continues.
        """
        self._tick_count += 1
        try:
            await self._tick_inner()
        except Exception:
            logger.exception("tick_error", tick=self._tick_count)

    async def _tick_inner(self) -> None:
        """Inner tick logic — exceptions bubble up to _tick()."""
        assert self._monitor is not None
        assert self._strategy is not None
        assert self._risk is not None
        assert self._engine is not None

        # 1. Poll for signals
        signals = await self._monitor.poll()
        if not signals:
            logger.debug("tick_no_signals", tick=self._tick_count)
            return

        logger.info(
            "tick_signals_received",
            tick=self._tick_count,
            count=len(signals),
            types=[s.event_type for s in signals],
        )

        # 2. Feed signals to strategy → collect proposals
        all_proposals: list[ProposedOrder] = []
        for signal in signals:
            proposals = await self._strategy.on_signal(signal)
            all_proposals.extend(proposals)

        if not all_proposals:
            logger.debug("tick_no_proposals", tick=self._tick_count)
            return

        logger.info(
            "tick_proposals_generated",
            tick=self._tick_count,
            count=len(all_proposals),
            tickers=[p.ticker for p in all_proposals],
        )

        # 3. Risk-check each proposal
        approved: list[ProposedOrder] = []
        for proposal in all_proposals:
            decision = self._risk.evaluate(proposal)
            if decision.approved:
                approved.append(proposal)
            else:
                logger.info(
                    "proposal_risk_rejected",
                    tick=self._tick_count,
                    ticker=proposal.ticker,
                    reasons=decision.rejection_reasons,
                )

        if not approved:
            logger.debug("tick_all_rejected", tick=self._tick_count)
            return

        logger.info(
            "tick_orders_approved",
            tick=self._tick_count,
            count=len(approved),
        )

        # 4. Execute approved orders
        results = await self._engine.submit_batch(approved)
        for result in results:
            if result.success:
                logger.info(
                    "order_executed",
                    tick=self._tick_count,
                    client_order_id=result.order.client_order_id,
                    ticker=result.order.ticker,
                    status=result.order.status,
                )
            else:
                logger.error(
                    "order_execution_failed",
                    tick=self._tick_count,
                    ticker=result.order.ticker,
                    error=result.error,
                )

    # ── Fill callback ─────────────────────────────────────────────────

    def _on_fill(self, fill_data: dict[str, Any]) -> None:
        """Handle a fill from the execution engine — update strategy state."""
        if self._strategy is None:
            return
        # Run the async on_fill synchronously within the event loop
        # (this callback is invoked synchronously by the engine)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._strategy.on_fill(fill_data))
        except RuntimeError:
            # No running loop — skip (happens in tests without async context)
            pass

    # ── Portfolio snapshot (for risk manager refresh) ─────────────────

    def build_portfolio_snapshot(self, balance_cents: int = 0) -> PortfolioSnapshot:
        """Build a portfolio snapshot from the strategy's current state.

        In a full production system this would query the database and API.
        For now it uses the strategy's internal contract views.
        """
        positions: list[PositionInfo] = []
        if self._strategy:
            for ticker, contract in self._strategy.contracts.items():
                if contract.position != 0:
                    positions.append(
                        PositionInfo(
                            ticker=ticker,
                            event_ticker=ticker.rsplit("-", 1)[0] if "-" in ticker else ticker,
                            quantity=abs(contract.position),
                            avg_cost_cents=float(contract.last_price or contract.yes_ask),
                        )
                    )
        return PortfolioSnapshot(balance_cents=balance_cents, positions=positions)
