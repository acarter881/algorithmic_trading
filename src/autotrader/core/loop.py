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
import contextlib
import os
from typing import TYPE_CHECKING, Any

import structlog

from autotrader.api.client import KalshiAPIClient, MarketInfo
from autotrader.api.websocket import Channel, KalshiWebSocketClient
from autotrader.execution.engine import ExecutionEngine, ExecutionMode
from autotrader.monitoring.discord import DiscordAlerter
from autotrader.monitoring.metrics import metrics
from autotrader.risk.manager import PortfolioSnapshot, PositionInfo, RiskManager
from autotrader.signals.arena_monitor import ArenaMonitor, ArenaMonitorFailureThresholdError
from autotrader.state.repository import TradingRepository
from autotrader.strategies.leaderboard_alpha import LeaderboardAlphaStrategy
from autotrader.utils.fees import FeeCalculator

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from autotrader.config.models import AppConfig
    from autotrader.strategies.base import ProposedOrder, Strategy

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
        self._strategies: dict[str, Strategy] = {}
        self._risk: RiskManager | None = None
        self._engine: ExecutionEngine | None = None
        self._repo: TradingRepository | None = None
        self._alerter: DiscordAlerter | None = None
        self._api_client: KalshiAPIClient | None = None
        self._market_data_client: KalshiAPIClient | None = None
        self._ws_client: KalshiWebSocketClient | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._fee_calc = FeeCalculator()
        self._ticker_event_map: dict[str, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(
        self,
        market_data: dict[str, Any] | None = None,
        session_factory: sessionmaker[Session] | None = None,
        api_client: KalshiAPIClient | None = None,
    ) -> None:
        """Create and initialize all components.

        If *market_data* is not provided, the loop will attempt to
        discover tradable markets from the Kalshi API for each series
        in the strategy's ``target_series`` list.  If discovery fails
        (e.g. missing credentials), the loop starts with no contracts
        and logs a warning.

        An *api_client* may be injected for testing; otherwise one is
        created from the application config.
        """
        # Arena monitor
        self._monitor = ArenaMonitor(config=self._config.arena_monitor)
        await self._monitor.initialize()

        # Kalshi API client — used for market discovery and live execution.
        # When injected (e.g. in tests), it is used directly for discovery.
        self._api_client = api_client

        # Auto-discover markets when none are provided
        if market_data is None and self._api_client is not None:
            market_data = self._discover_markets(self._api_client)

        # Strategy — primary (leaderboard_alpha) is always created.
        # Additional strategies can be registered via register_strategy().
        primary = LeaderboardAlphaStrategy(
            config=self._config.leaderboard_alpha,
            fee_calculator=self._fee_calc,
        )
        self._strategies[primary.name] = primary
        is_paper_mode = self._config.kalshi.environment.value != "production"

        # Database repository (optional — gracefully degrades if not provided)
        if session_factory is not None:
            self._repo = TradingRepository(session_factory)

        if market_data is None:
            market_data = self._bootstrap_market_data()
        self._ticker_event_map = self._extract_ticker_event_map(market_data)

        # Initialize all registered strategies
        for strat in self._strategies.values():
            state_payload: dict[str, Any] = {}
            if self._repo is not None:
                state_payload["positions"] = self._repo.get_net_positions_by_ticker(strat.name, is_paper=is_paper_mode)
            state_payload["ticker_event_map"] = dict(self._ticker_event_map)
            await strat.initialize(market_data, state_payload)

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
            live_client = self._api_client
            if live_client is None:
                live_client = KalshiAPIClient(self._config.kalshi)
                private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
                live_client.connect(private_key_pem=private_key_pem)
            self._engine = ExecutionEngine(mode=mode, api_client=live_client, fee_calculator=self._fee_calc)
        else:
            self._engine = ExecutionEngine(mode=mode, fee_calculator=self._fee_calc)

        # Wire fill callbacks: engine fills → strategy position tracking
        self._engine.on_fill(self._on_fill)

        # WebSocket client for real-time market data streaming
        if self._config.kalshi.websocket_enabled:
            self._init_websocket()

        logger.info(
            "trading_loop_initialized",
            mode=mode.value,
            poll_interval=self._config.arena_monitor.poll_interval_seconds,
            strategies=list(self._strategies.keys()),
            contracts_loaded=sum(len(s.contracts) for s in self._strategies.values() if hasattr(s, "contracts")),
            persistence=self._repo is not None,
            discord=self._alerter.enabled,
        )

        primary = self._primary_strategy
        if primary is not None and hasattr(primary, "contracts") and not primary.contracts:
            logger.warning(
                "no_tradable_markets_loaded",
                target_series=self._config.leaderboard_alpha.target_series,
                hint="Check API credentials and that target series have open markets",
            )

        total_contracts = sum(len(s.contracts) for s in self._strategies.values() if hasattr(s, "contracts"))
        # Record startup event
        self._persist_system_event(
            "startup",
            {"mode": mode.value, "contracts_loaded": total_contracts},
            "info",
        )
        await self._send_system_alert("Autotrader Started", f"Mode: {mode.value}")

    async def _validate_startup_markets(self, *, is_paper_mode: bool) -> None:
        """Validate that configured target series have at least one tradable contract."""
        primary = self._primary_strategy
        if primary is None or not hasattr(primary, "contracts"):
            return

        expected_series = list(self._config.leaderboard_alpha.target_series)
        loaded_contracts = primary.contracts

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
                    "event_ticker": market.event_ticker,
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
        reconcile_interval = self._config.risk.global_config.reconciliation_interval_seconds
        ticks_per_reconcile = max(1, reconcile_interval // interval) if reconcile_interval > 0 else 0

        # Paper mode tracks positions purely in-memory — reconciling
        # against the exchange would overwrite paper positions with zeros.
        is_paper = self._engine is not None and self._engine.mode == ExecutionMode.PAPER
        if is_paper:
            ticks_per_reconcile = 0

        logger.info(
            "trading_loop_started",
            interval_seconds=interval,
            reconcile_every_ticks=ticks_per_reconcile or "disabled",
            websocket=self._ws_client is not None,
        )

        # Start WebSocket streaming in the background (if configured).
        await self._start_websocket()

        # Reconcile on startup so that positions are accurate from tick 1.
        # Skipped in paper mode since exchange state doesn't reflect paper trades.
        if not is_paper:
            await self._reconcile_positions()

        while self._running:
            await self._tick()
            if ticks_per_reconcile and self._tick_count % ticks_per_reconcile == 0:
                await self._reconcile_positions()
            await asyncio.sleep(interval)

        # Stop WebSocket when the loop exits.
        await self._stop_websocket()

        logger.info("trading_loop_stopped", total_ticks=self._tick_count)

    def stop(self) -> None:
        """Signal the loop to stop after the current tick."""
        self._running = False
        logger.info("trading_loop_stop_requested")

    async def shutdown(self) -> None:
        """Tear down all components."""
        self._persist_system_event("shutdown", {"ticks": self._tick_count}, "info")
        await self._send_system_alert("Autotrader Stopped", f"Total ticks: {self._tick_count}")

        await self._stop_websocket()
        if self._monitor:
            await self._monitor.teardown()
        for strat in self._strategies.values():
            await strat.teardown()
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
        """Return the primary strategy (backward compatibility)."""
        s = self._strategies.get("leaderboard_alpha")
        if isinstance(s, LeaderboardAlphaStrategy):
            return s
        return None

    @property
    def strategies(self) -> dict[str, Strategy]:
        """Return all registered strategies."""
        return self._strategies

    @property
    def _primary_strategy(self) -> Strategy | None:
        """Internal helper — returns the first registered strategy."""
        if not self._strategies:
            return None
        return next(iter(self._strategies.values()))

    def register_strategy(self, strategy: Strategy) -> None:
        """Register an additional strategy (call before initialize)."""
        self._strategies[strategy.name] = strategy

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

    @property
    def ws_client(self) -> KalshiWebSocketClient | None:
        return self._ws_client

    # ── Market discovery ────────────────────────────────────────────────

    def _discover_markets(self, api: KalshiAPIClient) -> dict[str, Any]:
        """Discover open markets for every series in ``target_series``.

        Returns a dict in the shape the strategy expects::

            {"markets": [{"ticker": ..., "subtitle": ..., ...}, ...]}
        """
        target_series = self._config.leaderboard_alpha.target_series
        all_markets: list[dict[str, Any]] = []

        for series in target_series:
            try:
                series_markets: list[MarketInfo] = []
                cursor: str | None = None
                while True:
                    page, cursor = api.get_markets(
                        series_ticker=series,
                        status="open",
                        limit=200,
                        cursor=cursor,
                    )
                    series_markets.extend(page)
                    if not cursor or not page:
                        break
                for m in series_markets:
                    all_markets.append(self._market_info_to_dict(m))
                logger.info(
                    "markets_discovered",
                    series=series,
                    count=len(series_markets),
                    tickers=[m.ticker for m in series_markets],
                )
            except Exception:
                logger.warning("market_discovery_failed", series=series)

        logger.info("market_discovery_complete", total=len(all_markets))
        return {"markets": all_markets}

    @staticmethod
    def _market_info_to_dict(m: MarketInfo) -> dict[str, Any]:
        """Convert a :class:`MarketInfo` to the dict format the strategy expects."""
        return {
            "ticker": m.ticker,
            "event_ticker": m.event_ticker,
            "series_ticker": m.series_ticker,
            "title": m.title,
            "subtitle": m.subtitle,
            "status": m.status,
            "yes_bid": m.yes_bid,
            "yes_ask": m.yes_ask,
            "last_price": m.last_price,
        }

    # ── Core tick ─────────────────────────────────────────────────────

    async def _tick(self) -> None:
        """Execute one poll → signal → propose → risk → execute cycle.

        All exceptions are caught and logged so that the loop continues.
        """
        self._tick_count += 1
        metrics.increment("ticks_total")
        try:
            await self._tick_inner()
        except Exception:
            metrics.increment("tick_errors_total")
            logger.exception("tick_error", tick=self._tick_count)
            self._persist_system_event("tick_error", {"tick": self._tick_count}, "error")
            await self._send_error_alert("tick_error", f"Tick {self._tick_count} failed — see logs")

    async def _tick_inner(self) -> None:
        """Inner tick logic — exceptions bubble up to _tick()."""
        assert self._monitor is not None
        assert self._strategies
        assert self._risk is not None
        assert self._engine is not None

        # 0. Refresh tracked market quotes every tick so strategy uses current prices.
        await self._refresh_market_data()

        # 1. Poll for signals
        try:
            signals = await self._monitor.poll()
        except ArenaMonitorFailureThresholdError as failure:
            await self._handle_arena_monitor_failure_threshold(failure)
            return

        if not signals:
            logger.debug("tick_no_signals", tick=self._tick_count)
            return

        metrics.increment("signals_total", len(signals))
        logger.info(
            "tick_signals_received",
            tick=self._tick_count,
            count=len(signals),
            types=[s.event_type for s in signals],
        )

        # Persist signals
        for sig in signals:
            self._persist_signal(sig)

        # 2. Feed signals to all strategies → collect proposals
        all_proposals: list[ProposedOrder] = []
        for signal in signals:
            for strat in self._strategies.values():
                proposals = await strat.on_signal(signal)
                all_proposals.extend(proposals)

        if not all_proposals:
            logger.debug("tick_no_proposals", tick=self._tick_count)
            return

        metrics.increment("proposals_total", len(all_proposals))
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
                metrics.increment("risk_approved_total")
            else:
                metrics.increment("risk_rejected_total")
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
                metrics.increment("orders_filled_total")
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
                metrics.increment("orders_failed_total")
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

    async def _handle_arena_monitor_failure_threshold(
        self,
        failure: ArenaMonitorFailureThresholdError,
    ) -> None:
        """Protective response when Arena monitor repeatedly fails."""
        details = {
            "consecutive_failures": failure.consecutive_failures,
            "max_consecutive_failures": failure.max_consecutive_failures,
            "urls_attempted": failure.urls_attempted,
            "tick": self._tick_count,
        }

        logger.critical("arena_failure_threshold_triggered", **details)

        if self._risk:
            self._risk.activate_kill_switch(
                reason=(
                    "Arena monitor failure threshold exceeded: "
                    f"{failure.consecutive_failures}/{failure.max_consecutive_failures}; "
                    f"urls={failure.urls_attempted}"
                )
            )

        self.stop()
        self._persist_system_event("arena_failure_threshold_exceeded", details, severity="critical")

        message = (
            "Arena monitor failed consecutively and triggered protective shutdown. "
            f"consecutive_failures={failure.consecutive_failures}, "
            f"max_consecutive_failures={failure.max_consecutive_failures}, "
            f"urls_attempted={failure.urls_attempted}"
        )
        await self._send_error_alert("arena_failure_threshold_exceeded", message)
        await self._send_system_alert("CRITICAL: Arena monitor failure threshold exceeded", message)

    async def _refresh_market_data(self) -> None:
        """Refresh quotes for all strategy-tracked tickers and route into strategy.

        When WebSocket streaming is active and connected, REST polling is skipped
        because the ticker channel pushes updates in real time.
        """
        if not self._strategies or self._market_data_client is None:
            return

        # Skip REST polling when WebSocket is streaming prices
        if self._ws_client is not None and self._ws_client.connected:
            return

        # Collect tickers from all strategies
        all_tickers: set[str] = set()
        for strat in self._strategies.values():
            if hasattr(strat, "contracts"):
                all_tickers.update(strat.contracts.keys())

        for ticker in all_tickers:
            try:
                market = self._market_data_client.get_market(ticker)
                update = {
                    "ticker": ticker,
                    "yes_bid": market.yes_bid,
                    "yes_ask": market.yes_ask,
                    "last_price": market.last_price,
                    "event_ticker": market.event_ticker,
                }
                for strat in self._strategies.values():
                    await strat.on_market_update(update)
                if market.event_ticker:
                    self._ticker_event_map[ticker] = market.event_ticker
            except Exception:
                logger.warning("market_data_refresh_failed", ticker=ticker, tick=self._tick_count)

    # ── Position reconciliation ──────────────────────────────────────

    async def _reconcile_positions(self) -> None:
        """Reconcile in-memory strategy positions against the Kalshi API.

        Fetches positions from the exchange and compares them to the
        strategy's internal tracking.  Discrepancies are logged and the
        strategy's positions are corrected to match the exchange.

        Runs periodically (configurable via ``reconciliation_interval_seconds``)
        and once on startup to recover from any prior unclean shutdown.
        """
        api = self._api_client or self._market_data_client
        if api is None or not self._strategies:
            return

        # Collect all tracked tickers across strategies
        all_tracked: set[str] = set()
        for strat in self._strategies.values():
            if hasattr(strat, "contracts"):
                all_tracked.update(strat.contracts.keys())

        try:
            exchange_positions: dict[str, int] = {}
            cursor: str | None = None
            while True:
                page, cursor = api.get_positions(limit=200, cursor=cursor)
                for pos in page:
                    if pos.ticker in all_tracked:
                        exchange_positions[pos.ticker] = pos.position
                if not cursor or not page:
                    break

            # Compare against strategy in-memory state (across all strategies)
            mismatches: list[dict[str, object]] = []
            for strat in self._strategies.values():
                if not hasattr(strat, "contracts"):
                    continue
                for ticker, contract in strat.contracts.items():
                    exchange_qty = exchange_positions.get(ticker, 0)
                    if contract.position != exchange_qty:
                        mismatches.append(
                            {
                                "ticker": ticker,
                                "strategy": strat.name,
                                "strategy_position": contract.position,
                                "exchange_position": exchange_qty,
                            }
                        )
                        # Correct the strategy's position to match the exchange
                        contract.position = exchange_qty

            if mismatches:
                logger.warning(
                    "position_reconciliation_mismatches",
                    count=len(mismatches),
                    mismatches=mismatches,
                    tick=self._tick_count,
                )
                self._persist_system_event(
                    "position_reconciliation_mismatch",
                    {"mismatches": mismatches, "tick": self._tick_count},
                    severity="warning",
                )
                await self._send_error_alert(
                    "position_reconciliation_mismatch",
                    f"{len(mismatches)} position(s) corrected: "
                    + ", ".join(
                        f"{m['ticker']}: {m['strategy_position']}→{m['exchange_position']}" for m in mismatches
                    ),
                )
            else:
                logger.debug(
                    "position_reconciliation_ok",
                    tracked_tickers=len(all_tracked),
                    tick=self._tick_count,
                )
        except Exception:
            logger.exception("position_reconciliation_failed", tick=self._tick_count)

    # ── WebSocket streaming ────────────────────────────────────────────

    def _init_websocket(self) -> None:
        """Create and configure the WebSocket client for real-time streaming."""
        private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM", "")
        is_demo = self._config.kalshi.environment.value != "production"
        self._ws_client = KalshiWebSocketClient(
            api_key_id=self._config.kalshi.api_key_id,
            private_key_pem=private_key_pem,
            is_demo=is_demo,
        )

        # Subscribe to ticker channel for all tracked markets across all strategies
        tickers: list[str] = []
        for strat in self._strategies.values():
            if hasattr(strat, "contracts"):
                tickers.extend(strat.contracts.keys())
        if tickers:
            self._ws_client.subscribe(Channel.TICKER, tickers=tickers)

        # Subscribe to private channels when authenticated
        if self._config.kalshi.api_key_id and private_key_pem:
            self._ws_client.subscribe(Channel.FILL)
            self._ws_client.subscribe(Channel.USER_ORDERS)

        # Register message handlers
        self._ws_client.on_message("ticker", self._on_ws_ticker)
        self._ws_client.on_message("fill", self._on_ws_fill)

        logger.info(
            "websocket_initialized",
            tickers=len(tickers),
            authenticated=bool(self._config.kalshi.api_key_id and private_key_pem),
        )

    async def _start_websocket(self) -> None:
        """Start the WebSocket connection as a background task."""
        if self._ws_client is None:
            return
        self._ws_task = asyncio.create_task(self._ws_client.connect())
        logger.info("websocket_task_started")

    async def _stop_websocket(self) -> None:
        """Gracefully disconnect the WebSocket and cancel its task."""
        if self._ws_client is not None:
            await self._ws_client.disconnect()
        if self._ws_task is not None and not self._ws_task.done():
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task
            self._ws_task = None
        logger.info("websocket_stopped")

    async def _on_ws_ticker(self, message: dict[str, Any]) -> None:
        """Handle a WebSocket ticker message — route to all strategies."""
        if not self._strategies:
            return
        msg = message.get("msg", {})
        if not isinstance(msg, dict):
            return
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        # Check if any strategy tracks this ticker
        relevant = any(hasattr(s, "contracts") and ticker in s.contracts for s in self._strategies.values())
        if not relevant:
            return

        update: dict[str, Any] = {"ticker": ticker}
        for key in ("yes_bid", "yes_ask", "last_price"):
            value = msg.get(key)
            if isinstance(value, int) and value > 0:
                update[key] = value
        event_ticker = msg.get("event_ticker")
        if isinstance(event_ticker, str) and event_ticker:
            update["event_ticker"] = event_ticker
            self._ticker_event_map[ticker] = event_ticker
        for strat in self._strategies.values():
            await strat.on_market_update(update)

    async def _on_ws_fill(self, message: dict[str, Any]) -> None:
        """Handle a WebSocket fill message — route to fill callback."""
        msg = message.get("msg", {})
        if isinstance(msg, dict):
            self._on_fill(msg)

    # ── Fill callback ─────────────────────────────────────────────────

    def _on_fill(self, fill_data: dict[str, Any]) -> None:
        """Handle a fill from the execution engine — update strategy + persist."""
        # Determine which strategy this fill belongs to from client_order_id prefix
        strategy_name = (
            fill_data.get("client_order_id", "").split("-")[0] if fill_data.get("client_order_id") else "unknown"
        )

        # Persist fill
        if self._repo:
            self._repo.record_fill(fill_data, strategy_name)

        if not self._strategies:
            return

        # Route to the matching strategy, or all strategies if unknown
        target_strategies = (
            [self._strategies[strategy_name]] if strategy_name in self._strategies else list(self._strategies.values())
        )

        # Run the async on_fill within the event loop
        # (this callback is invoked synchronously by the engine)
        try:
            loop = asyncio.get_running_loop()
            for strat in target_strategies:
                loop.create_task(strat.on_fill(fill_data))
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
        for strat in self._strategies.values():
            if not hasattr(strat, "contracts"):
                continue
            for ticker, contract in strat.contracts.items():
                if contract.position != 0:
                    positions.append(
                        PositionInfo(
                            ticker=ticker,
                            event_ticker=self._resolve_event_ticker(ticker),
                            quantity=contract.position,
                            avg_cost_cents=float(contract.last_price or contract.yes_ask),
                        )
                    )

        daily_realized: dict[str, int] = {}
        daily_unrealized: dict[str, int] = {}
        if self._repo:
            for strat in self._strategies.values():
                pnl = self._repo.get_daily_pnl(strat.name)
                if pnl is not None:
                    daily_realized[strat.name] = pnl.realized_pnl_cents
                    daily_unrealized[strat.name] = pnl.unrealized_pnl_cents

        return PortfolioSnapshot(
            balance_cents=balance_cents,
            positions=positions,
            daily_realized_pnl_cents=daily_realized,
            daily_unrealized_pnl_cents=daily_unrealized,
            ticker_event_map=dict(self._ticker_event_map),
        )

    def _extract_ticker_event_map(self, market_data: dict[str, Any]) -> dict[str, str]:
        """Build a ticker → event_ticker map from market payloads."""
        ticker_event_map: dict[str, str] = {}
        for market in market_data.get("markets", []):
            if not isinstance(market, dict):
                continue
            ticker = market.get("ticker")
            event_ticker = market.get("event_ticker")
            if isinstance(ticker, str) and ticker and isinstance(event_ticker, str) and event_ticker:
                ticker_event_map[ticker] = event_ticker
        return ticker_event_map

    def _resolve_event_ticker(self, ticker: str) -> str:
        """Resolve event ticker using in-memory map or strategy metadata."""
        mapped = self._ticker_event_map.get(ticker)
        if mapped:
            return mapped
        for strat in self._strategies.values():
            if hasattr(strat, "resolve_event_ticker"):
                resolved = strat.resolve_event_ticker(ticker)
                if resolved:
                    self._ticker_event_map[ticker] = resolved
                    return resolved
        return ticker.rsplit("-", 1)[0] if "-" in ticker else ticker

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
