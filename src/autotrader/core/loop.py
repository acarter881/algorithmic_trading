"""Main run loop that orchestrates all autotrader components.

The :class:`TradingLoop` ties together:

- **ArenaMonitor** — polls the leaderboard for signals
- **LeaderboardAlphaStrategy** — converts signals into trade proposals
- **RiskManager** — gates proposals against risk limits
- **ExecutionEngine** — submits approved orders (paper or live)
- **TradingRepository** — persists orders, fills, signals, risk events
- **DiscordAlerter** — sends real-time notifications

The loop runs on a configurable interval (default: arena poll interval)
and supports graceful shutdown via :meth:`stop`.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import structlog

from autotrader.api.client import KalshiAPIClient, MarketInfo
from autotrader.execution.engine import ExecutionEngine, ExecutionMode
from autotrader.monitoring.discord import DiscordAlerter
from autotrader.risk.manager import PortfolioSnapshot, PositionInfo, RiskManager
from autotrader.signals.arena_monitor import ArenaMonitor
from autotrader.state.repository import TradingRepository
from autotrader.strategies.leaderboard_alpha import LeaderboardAlphaStrategy
from autotrader.utils.fees import FeeCalculator

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from autotrader.config.models import AppConfig
    from autotrader.strategies.base import ProposedOrder

logger = structlog.get_logger("autotrader.core.loop")


class TradingLoop:
    """Async run loop that orchestrates the full trading pipeline.

    Usage::

        loop = TradingLoop(config)
        await loop.initialize(session_factory=sf)
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
        self._repo: TradingRepository | None = None
        self._alerter: DiscordAlerter | None = None
        self._market_data_client: KalshiAPIClient | None = None
        self._fee_calc = FeeCalculator()

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(
        self,
        market_data: dict[str, Any] | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        """Create and initialize all components."""
        # Arena monitor
        self._monitor = ArenaMonitor(config=self._config.arena_monitor)
        await self._monitor.initialize()

        # Strategy
        self._strategy = LeaderboardAlphaStrategy(
            config=self._config.leaderboard_alpha,
            fee_calculator=self._fee_calc,
        )
        is_paper_mode = self._config.kalshi.environment.value != "production"

        # Database repository (optional — gracefully degrades if not provided)
        state_payload: dict[str, Any] | None = None
        if session_factory is not None:
            self._repo = TradingRepository(session_factory)
            state_payload = {
                "positions": self._repo.get_net_positions_by_ticker(self._strategy.name, is_paper=is_paper_mode),
            }

        if market_data is None:
            market_data = self._bootstrap_market_data()
        await self._strategy.initialize(market_data, state_payload)

        # Discord alerter (initialized early so startup validation can alert)
        self._alerter = DiscordAlerter(self._config.discord)
        await self._alerter.initialize()

        await self._validate_startup_markets(is_paper_mode=is_paper_mode)

        # Market data refresh client (runtime quote updates for tracked tickers)
        self._market_data_client = KalshiAPIClient(self._config.kalshi)
        try:
            private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
            self._market_data_client.connect(private_key_pem=private_key_pem)
        except Exception:
            logger.exception("market_data_client_init_failed")
            self._market_data_client = None

        # Risk manager
        self._risk = RiskManager(config=self._config.risk)

        # Execution engine
        mode = ExecutionMode.PAPER if is_paper_mode else ExecutionMode.LIVE
        if mode == ExecutionMode.LIVE:
            client = KalshiAPIClient(self._config.kalshi)
            private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
            client.connect(private_key_pem=private_key_pem)
            self._engine = ExecutionEngine(mode=mode, api_client=client, fee_calculator=self._fee_calc)
        else:
            self._engine = ExecutionEngine(mode=mode, fee_calculator=self._fee_calc)

        # Wire fill callbacks: engine fills → strategy position tracking
        self._engine.on_fill(self._on_fill)

        logger.info(
            "trading_loop_initialized",
            mode=mode.value,
            poll_interval=self._config.arena_monitor.poll_interval_seconds,
            target_series=self._config.leaderboard_alpha.target_series,
            persistence=self._repo is not None,
            discord=self._alerter.enabled,
        )

        # Record startup event
        self._persist_system_event("startup", {"mode": mode.value}, "info")
        await self._send_system_alert("Autotrader Started", f"Mode: {mode.value}")

    async def _validate_startup_markets(self, *, is_paper_mode: bool) -> None:
        """Validate that configured target series have at least one tradable contract."""
        if self._strategy is None:
            return

        expected_series = list(self._config.leaderboard_alpha.target_series)
        loaded_contracts = self._strategy.contracts

        series_market_counts = {
            series: sum(1 for ticker in loaded_contracts if ticker.startswith(f"{series}-"))
            for series in expected_series
        }
        missing_series = [series for series, count in series_market_counts.items() if count < 1]
        if not missing_series:
            return

        mode = "paper" if is_paper_mode else "production"
        details = {
            "mode": mode,
            "expected_series": expected_series,
            "series_market_counts": series_market_counts,
            "loaded_contracts": len(loaded_contracts),
            "missing_series": missing_series,
        }
        logger.error("no_tradable_markets_loaded", **details)
        self._persist_system_event("no_tradable_markets_loaded", details, severity="critical")

        message = (
            f"Expected at least one open market per configured series. Missing series: {', '.join(missing_series)}"
        )
        if not is_paper_mode:
            raise RuntimeError(f"no_tradable_markets_loaded: {message}")

        # Paper mode degrades gracefully but emits repeated high-severity alerts.
        for _ in range(3):
            await self._send_error_alert("no_tradable_markets_loaded", message)
            await self._send_system_alert("CRITICAL: no_tradable_markets_loaded", message)

    def _bootstrap_market_data(self) -> dict[str, Any]:
        """Discover active markets for configured target series at startup."""
        client = KalshiAPIClient(self._config.kalshi)
        markets: list[MarketInfo] = []

        try:
            private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
            client.connect(private_key_pem=private_key_pem)

            for series_ticker in self._config.leaderboard_alpha.target_series:
                cursor: str | None = None
                while True:
                    series_markets, cursor = client.get_markets(
                        series_ticker=series_ticker,
                        status="open",
                        limit=200,
                        cursor=cursor,
                    )
                    markets.extend(series_markets)
                    if not cursor or not series_markets:
                        break

            logger.info(
                "market_data_bootstrapped",
                series=self._config.leaderboard_alpha.target_series,
                markets=len(markets),
            )
        except Exception:
            logger.exception(
                "market_data_bootstrap_failed",
                series=self._config.leaderboard_alpha.target_series,
            )
            return {"markets": []}

        return {
            "markets": [
                {
                    "ticker": market.ticker,
                    "title": market.title,
                    "subtitle": market.subtitle,
                    "yes_bid": market.yes_bid,
                    "yes_ask": market.yes_ask,
                    "last_price": market.last_price,
                }
                for market in markets
            ]
        }

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
        self._persist_system_event("shutdown", {"ticks": self._tick_count}, "info")
        await self._send_system_alert("Autotrader Stopped", f"Total ticks: {self._tick_count}")

        if self._monitor:
            await self._monitor.teardown()
        if self._strategy:
            await self._strategy.teardown()
        if self._alerter:
            await self._alerter.teardown()
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

    @property
    def repository(self) -> TradingRepository | None:
        return self._repo

    @property
    def alerter(self) -> DiscordAlerter | None:
        return self._alerter

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
            self._persist_system_event("tick_error", {"tick": self._tick_count}, "error")
            await self._send_error_alert("tick_error", f"Tick {self._tick_count} failed — see logs")

    async def _tick_inner(self) -> None:
        """Inner tick logic — exceptions bubble up to _tick()."""
        assert self._monitor is not None
        assert self._strategy is not None
        assert self._risk is not None
        assert self._engine is not None

        # 0. Refresh tracked market quotes every tick so strategy uses current prices.
        await self._refresh_market_data()

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

        # Persist signals
        for sig in signals:
            self._persist_signal(sig)

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

        # Refresh portfolio state before evaluating this tick's proposals.
        snapshot = self.build_portfolio_snapshot()
        self._risk.update_portfolio(snapshot)

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
                # Persist each risk rejection
                for check_result in decision.results:
                    if check_result.verdict.value == "rejected":
                        self._persist_risk_event(
                            check_result.check_name,
                            self._risk.serialize_order(proposal),
                            check_result.reason,
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
            # Persist order regardless of success
            self._persist_order(result.order)

            if result.success:
                logger.info(
                    "order_executed",
                    tick=self._tick_count,
                    client_order_id=result.order.client_order_id,
                    ticker=result.order.ticker,
                    status=result.order.status,
                )
                # Send Discord trade alert
                await self._send_trade_alert(result.order)
            else:
                logger.error(
                    "order_execution_failed",
                    tick=self._tick_count,
                    ticker=result.order.ticker,
                    error=result.error,
                )
                await self._send_error_alert(
                    "order_execution_failed",
                    f"Ticker: {result.order.ticker}, Error: {result.error}",
                )

    async def _refresh_market_data(self) -> None:
        """Refresh quotes for all strategy-tracked tickers and route into strategy."""
        if self._strategy is None or self._market_data_client is None:
            return

        for ticker in self._strategy.contracts:
            try:
                market = self._market_data_client.get_market(ticker)
                await self._strategy.on_market_update(
                    {
                        "ticker": ticker,
                        "yes_bid": market.yes_bid,
                        "yes_ask": market.yes_ask,
                        "last_price": market.last_price,
                    }
                )
            except Exception:
                logger.warning("market_data_refresh_failed", ticker=ticker, tick=self._tick_count)

    # ── Fill callback ─────────────────────────────────────────────────

    def _on_fill(self, fill_data: dict[str, Any]) -> None:
        """Handle a fill from the execution engine — update strategy + persist."""
        # Persist fill
        if self._repo:
            strategy = (
                fill_data.get("client_order_id", "").split("-")[0] if fill_data.get("client_order_id") else "unknown"
            )
            self._repo.record_fill(fill_data, strategy)

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

    def build_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Build a portfolio snapshot from the strategy's current state.

        Uses exchange/account balance when available, strategy-tracked positions,
        and persisted daily P&L from the repository (if configured).
        """
        balance_cents = self._current_balance_cents()
        positions: list[PositionInfo] = []
        if self._strategy:
            for ticker, contract in self._strategy.contracts.items():
                if contract.position != 0:
                    positions.append(
                        PositionInfo(
                            ticker=ticker,
                            event_ticker=ticker.rsplit("-", 1)[0] if "-" in ticker else ticker,
                            quantity=contract.position,
                            avg_cost_cents=float(contract.last_price or contract.yes_ask),
                        )
                    )

        daily_realized: dict[str, int] = {}
        daily_unrealized: dict[str, int] = {}
        if self._repo and self._strategy:
            pnl = self._repo.get_daily_pnl(self._strategy.name)
            if pnl is not None:
                daily_realized[self._strategy.name] = pnl.realized_pnl_cents
                daily_unrealized[self._strategy.name] = pnl.unrealized_pnl_cents

        return PortfolioSnapshot(
            balance_cents=balance_cents,
            positions=positions,
            daily_realized_pnl_cents=daily_realized,
            daily_unrealized_pnl_cents=daily_unrealized,
        )

    def _current_balance_cents(self) -> int:
        """Best-effort account balance lookup.

        Live mode uses the exchange API. Paper mode falls back to a configurable
        notional account balance.
        """
        if self._engine and self._engine.mode == ExecutionMode.LIVE and self._engine._api is not None:
            try:
                return self._engine._api.get_balance().balance
            except Exception:
                logger.warning("balance_fetch_failed", tick=self._tick_count)
                return 0

        try:
            return int(os.environ.get("AUTOTRADER_PAPER_BALANCE_CENTS", "100000"))
        except ValueError:
            logger.warning("invalid_paper_balance_env", value=os.environ.get("AUTOTRADER_PAPER_BALANCE_CENTS"))
            return 100_000

    # ── Persistence helpers ───────────────────────────────────────────

    def _persist_order(self, tracked: Any) -> None:
        if self._repo:
            self._repo.record_order(tracked)

    def _persist_signal(self, signal: Any) -> None:
        if self._repo:
            self._repo.record_signal(
                signal_source=signal.source,
                event_type=signal.event_type,
                data=signal.data,
                relevant_series=list(signal.relevant_series),
                urgency=str(signal.urgency),
            )

    def _persist_risk_event(self, check_name: str, order_data: dict[str, Any], reason: str) -> None:
        if self._repo:
            self._repo.record_risk_event(check_name, order_data, reason)

    def _persist_system_event(
        self, event_type: str, details: dict[str, Any] | None = None, severity: str = "info"
    ) -> None:
        if self._repo:
            self._repo.record_system_event(event_type, details, severity)

    # ── Discord alert helpers ─────────────────────────────────────────

    async def _send_trade_alert(self, tracked: Any) -> None:
        if self._alerter:
            await self._alerter.send_trade_alert(
                ticker=tracked.ticker,
                side=tracked.side,
                quantity=tracked.quantity,
                price_cents=tracked.price_cents,
                strategy=tracked.strategy,
                is_paper=tracked.is_paper,
            )

    async def _send_error_alert(self, error_type: str, details: str) -> None:
        if self._alerter:
            await self._alerter.send_error_alert(error_type, details)

    async def _send_system_alert(self, title: str, message: str) -> None:
        if self._alerter:
            await self._alerter.send_system_alert(title, message)
