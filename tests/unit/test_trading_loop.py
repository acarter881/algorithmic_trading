"""Unit tests for the trading loop and Discord alerter."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autotrader.api.client import KalshiAPIClient, KalshiAPIError, MarketInfo
from autotrader.config.models import (
    AppConfig,
    ArenaMonitorConfig,
    DiscordConfig,
    KalshiConfig,
    LeaderboardAlphaConfig,
)
from autotrader.core.loop import TradingLoop
from autotrader.execution.engine import ExecutionMode
from autotrader.monitoring.discord import DiscordAlerter
from autotrader.signals.base import Signal, SignalUrgency
from autotrader.state.models import Base, Fill, Order, RiskEvent, SystemEvent
from autotrader.state.models import Signal as SignalRow
from autotrader.strategies.base import OrderUrgency, ProposedOrder

# ── Helpers ──────────────────────────────────────────────────────────────


def _config(discord_enabled: bool = False) -> AppConfig:
    return AppConfig(
        arena_monitor=ArenaMonitorConfig(poll_interval_seconds=1),
        leaderboard_alpha=LeaderboardAlphaConfig(target_series=["KXTOPMODEL"]),
        discord=DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test/test" if discord_enabled else "",
            enabled=discord_enabled,
        ),
    )


def _session_factory() -> sessionmaker:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _signal(event_type: str = "ranking_change") -> Signal:
    return Signal(
        source="arena_monitor",
        timestamp=datetime.datetime(2026, 2, 27, 12, 0, 0),
        event_type=event_type,
        data={"model_name": "GPT-5", "old_rank_ub": 3, "new_rank_ub": 1},
        relevant_series=["KXTOPMODEL"],
        urgency=SignalUrgency.HIGH,
    )


def _proposal() -> ProposedOrder:
    return ProposedOrder(
        strategy="leaderboard_alpha",
        ticker="KXTOPMODEL-GPT5",
        side="yes",
        price_cents=50,
        quantity=5,
        urgency=OrderUrgency.AGGRESSIVE,
        rationale="test",
    )


# ── TradingLoop ──────────────────────────────────────────────────────────


class TestTradingLoopInit:
    async def test_initialize_creates_components(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()
        assert loop.strategy is not None
        assert loop.risk_manager is not None
        assert loop.execution_engine is not None
        assert loop.alerter is not None
        await loop.shutdown()

    async def test_default_paper_mode(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()
        assert loop.execution_engine is not None
        assert loop.execution_engine.mode == ExecutionMode.PAPER
        await loop.shutdown()

    async def test_not_running_initially(self) -> None:
        loop = TradingLoop(_config())
        assert not loop.running
        assert loop.tick_count == 0

    async def test_initialize_with_session_factory(self) -> None:
        loop = TradingLoop(_config())
        sf = _session_factory()
        await loop.initialize(session_factory=sf)
        assert loop.repository is not None
        await loop.shutdown()

    async def test_initialize_without_session_factory(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()
        assert loop.repository is None
        await loop.shutdown()


class TestTradingLoopTick:
    async def test_tick_with_no_signals(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        # Mock monitor to return no signals
        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await loop._tick()
        assert loop.tick_count == 1
        await loop.shutdown()

    async def test_tick_with_signals_and_proposals(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        # Mock monitor to return a signal
        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        # Mock strategy to return a proposal
        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[_proposal()])  # type: ignore[method-assign]

        await loop._tick()
        assert loop.tick_count == 1

        # Check that execution engine received orders
        assert loop.execution_engine is not None
        assert len(loop.execution_engine.orders) > 0
        await loop.shutdown()

    async def test_tick_risk_rejection(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        # Propose an order with invalid price (will be rejected by risk)
        bad_proposal = ProposedOrder(
            strategy="leaderboard_alpha",
            ticker="KXTOPMODEL-GPT5",
            side="yes",
            price_cents=0,  # Invalid — will fail price_sanity
            quantity=5,
            urgency=OrderUrgency.PASSIVE,
        )
        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[bad_proposal])  # type: ignore[method-assign]

        await loop._tick()

        # No orders should reach execution
        assert loop.execution_engine is not None
        assert len(loop.execution_engine.orders) == 0
        await loop.shutdown()

    async def test_tick_no_proposals(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await loop._tick()
        assert loop.execution_engine is not None
        assert len(loop.execution_engine.orders) == 0
        await loop.shutdown()

    async def test_tick_exception_does_not_crash(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(side_effect=RuntimeError("fetch error"))  # type: ignore[method-assign]

        # _tick catches exceptions, so this should not raise
        await loop._tick()
        await loop.shutdown()


class TestTradingLoopRunStop:
    async def test_stop_terminates_run(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[])  # type: ignore[method-assign]

        # Stop after a short delay
        async def _stop_soon() -> None:
            await asyncio.sleep(0.05)
            loop.stop()

        await asyncio.gather(loop.run(), _stop_soon())
        assert not loop.running
        assert loop.tick_count >= 1
        await loop.shutdown()

    async def test_run_requires_initialize(self) -> None:
        loop = TradingLoop(_config())
        try:
            await loop.run()
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "initialize" in str(e)


class TestPortfolioSnapshot:
    async def test_build_snapshot_empty(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()
        snap = loop.build_portfolio_snapshot(balance_cents=50000)
        assert snap.balance_cents == 50000
        assert len(snap.positions) == 0
        await loop.shutdown()


# ── Persistence integration ──────────────────────────────────────────────


class TestTradingLoopPersistence:
    async def test_tick_persists_signals(self) -> None:
        sf = _session_factory()
        loop = TradingLoop(_config())
        await loop.initialize(session_factory=sf)

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await loop._tick()

        with sf() as session:
            signals = session.query(SignalRow).all()
            assert len(signals) == 1
            assert signals[0].source == "arena_monitor"
            assert signals[0].event_type == "ranking_change"

        await loop.shutdown()

    async def test_tick_persists_orders_and_fills(self) -> None:
        sf = _session_factory()
        loop = TradingLoop(_config())
        await loop.initialize(session_factory=sf)

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[_proposal()])  # type: ignore[method-assign]

        await loop._tick()

        with sf() as session:
            orders = session.query(Order).all()
            assert len(orders) == 1
            assert orders[0].ticker == "KXTOPMODEL-GPT5"
            assert orders[0].status == "filled"  # paper mode = instant fill

            fills = session.query(Fill).all()
            assert len(fills) == 1
            assert fills[0].ticker == "KXTOPMODEL-GPT5"

        await loop.shutdown()

    async def test_tick_persists_risk_rejections(self) -> None:
        sf = _session_factory()
        loop = TradingLoop(_config())
        await loop.initialize(session_factory=sf)

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        bad_proposal = ProposedOrder(
            strategy="leaderboard_alpha",
            ticker="T1",
            side="yes",
            price_cents=0,  # fails price_sanity
            quantity=5,
        )
        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[bad_proposal])  # type: ignore[method-assign]

        await loop._tick()

        with sf() as session:
            events = session.query(RiskEvent).all()
            assert len(events) >= 1
            assert any("price_sanity" in e.check_name for e in events)

        await loop.shutdown()

    async def test_startup_shutdown_events_persisted(self) -> None:
        sf = _session_factory()
        loop = TradingLoop(_config())
        await loop.initialize(session_factory=sf)
        await loop.shutdown()

        with sf() as session:
            events = session.query(SystemEvent).all()
            types = [e.event_type for e in events]
            assert "startup" in types
            assert "shutdown" in types

    async def test_tick_error_persists_system_event(self) -> None:
        sf = _session_factory()
        loop = TradingLoop(_config())
        await loop.initialize(session_factory=sf)

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        await loop._tick()

        with sf() as session:
            events = session.query(SystemEvent).all()
            error_events = [e for e in events if e.event_type == "tick_error"]
            assert len(error_events) == 1
            assert error_events[0].severity == "error"

        await loop.shutdown()


# ── Discord alerting integration ─────────────────────────────────────────


class TestTradingLoopAlerting:
    async def test_alerter_initialized_in_loop(self) -> None:
        loop = TradingLoop(_config(discord_enabled=True))
        await loop.initialize()
        assert loop.alerter is not None
        assert loop.alerter.enabled
        await loop.shutdown()

    async def test_trade_alert_sent_on_execution(self) -> None:
        loop = TradingLoop(_config(discord_enabled=True))
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[_proposal()])  # type: ignore[method-assign]

        # Mock the alerter's send_trade_alert
        assert loop._alerter is not None
        loop._alerter.send_trade_alert = AsyncMock()  # type: ignore[method-assign]
        loop._alerter.send_error_alert = AsyncMock()  # type: ignore[method-assign]

        await loop._tick()

        loop._alerter.send_trade_alert.assert_called_once()
        call_kwargs = loop._alerter.send_trade_alert.call_args[1]
        assert call_kwargs["ticker"] == "KXTOPMODEL-GPT5"
        assert call_kwargs["side"] == "yes"
        assert call_kwargs["is_paper"] is True

        await loop.shutdown()

    async def test_error_alert_sent_on_tick_error(self) -> None:
        loop = TradingLoop(_config(discord_enabled=True))
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        assert loop._alerter is not None
        loop._alerter.send_error_alert = AsyncMock()  # type: ignore[method-assign]
        loop._alerter.send_system_alert = AsyncMock()  # type: ignore[method-assign]

        await loop._tick()

        loop._alerter.send_error_alert.assert_called_once()

        await loop.shutdown()

    async def test_system_alerts_on_startup_shutdown(self) -> None:
        loop = TradingLoop(_config(discord_enabled=True))
        await loop.initialize()

        # After initialize(), replace alerter methods with mocks to capture calls
        assert loop._alerter is not None
        mock_alert = AsyncMock()
        loop._alerter.send_system_alert = mock_alert  # type: ignore[method-assign]

        # shutdown sends a system alert
        await loop.shutdown()

        mock_alert.assert_called_once()
        call_args = mock_alert.call_args
        assert "Stopped" in call_args[0][0]


# ── DiscordAlerter (standalone tests) ───────────────────────────────────


class TestDiscordAlerter:
    def _discord_config(self, enabled: bool = True) -> DiscordConfig:
        return DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test/test",
            enabled=enabled,
            alert_on_trades=True,
            alert_on_signals=True,
            alert_on_errors=True,
            large_trade_threshold=50.0,
        )

    async def test_disabled_alerter(self) -> None:
        alerter = DiscordAlerter(self._discord_config(enabled=False))
        assert not alerter.enabled
        await alerter.initialize()
        await alerter.teardown()

    async def test_enabled_alerter(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        assert alerter.enabled
        await alerter.initialize()
        assert alerter._client is not None
        await alerter.teardown()
        assert alerter._client is None

    async def test_send_trade_alert(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        await alerter.initialize()
        assert alerter._client is not None

        # Mock the HTTP client
        alerter._client.post = AsyncMock(return_value=MagicMock(status_code=204))  # type: ignore[method-assign]

        await alerter.send_trade_alert(
            ticker="KXTOPMODEL-GPT5",
            side="yes",
            quantity=5,
            price_cents=50,
            strategy="leaderboard_alpha",
        )
        alerter._client.post.assert_called_once()
        await alerter.teardown()

    async def test_send_signal_alert(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        await alerter.initialize()
        assert alerter._client is not None
        alerter._client.post = AsyncMock(return_value=MagicMock(status_code=204))  # type: ignore[method-assign]

        await alerter.send_signal_alert("new_leader", {"new_leader": "GPT-5"})
        alerter._client.post.assert_called_once()
        await alerter.teardown()

    async def test_send_error_alert(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        await alerter.initialize()
        assert alerter._client is not None
        alerter._client.post = AsyncMock(return_value=MagicMock(status_code=204))  # type: ignore[method-assign]

        await alerter.send_error_alert("api_failure", "Connection refused")
        alerter._client.post.assert_called_once()
        await alerter.teardown()

    async def test_send_system_alert(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        await alerter.initialize()
        assert alerter._client is not None
        alerter._client.post = AsyncMock(return_value=MagicMock(status_code=204))  # type: ignore[method-assign]

        await alerter.send_system_alert("Startup", "Autotrader started")
        alerter._client.post.assert_called_once()
        await alerter.teardown()

    async def test_large_trade_alert(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        await alerter.initialize()
        assert alerter._client is not None
        alerter._client.post = AsyncMock(return_value=MagicMock(status_code=204))  # type: ignore[method-assign]

        # 50c * 150 = $75 > $50 threshold
        await alerter.send_trade_alert(ticker="T1", side="yes", quantity=150, price_cents=50, strategy="test")
        call_args = alerter._client.post.call_args
        payload = call_args[1]["json"]
        assert "LARGE TRADE" in payload["embeds"][0]["title"]
        await alerter.teardown()

    async def test_http_error_does_not_propagate(self) -> None:
        alerter = DiscordAlerter(self._discord_config())
        await alerter.initialize()
        assert alerter._client is not None
        alerter._client.post = AsyncMock(side_effect=Exception("network error"))  # type: ignore[method-assign]

        # Should not raise
        await alerter.send_trade_alert(ticker="T1", side="yes", quantity=5, price_cents=50, strategy="test")
        await alerter.teardown()

    async def test_no_send_when_disabled_flag(self) -> None:
        config = DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test/test",
            enabled=True,
            alert_on_trades=False,
            alert_on_signals=False,
            alert_on_errors=False,
        )
        alerter = DiscordAlerter(config)
        await alerter.initialize()
        assert alerter._client is not None
        alerter._client.post = AsyncMock(return_value=MagicMock(status_code=204))  # type: ignore[method-assign]

        await alerter.send_trade_alert("T1", "yes", 5, 50, "test")
        await alerter.send_signal_alert("test", {})
        await alerter.send_error_alert("test", "test")
        alerter._client.post.assert_not_called()
        await alerter.teardown()


# ── Market discovery ──────────────────────────────────────────────────


def _mock_api_client(markets: list[MarketInfo] | None = None) -> KalshiAPIClient:
    """Create a mock KalshiAPIClient that returns the given markets."""
    client = MagicMock(spec=KalshiAPIClient)
    if markets is None:
        markets = [
            MarketInfo(
                ticker="KXTOPMODEL-GPT5",
                event_ticker="KXTOPMODEL",
                series_ticker="KXTOPMODEL",
                title="Which AI model will be #1?",
                subtitle="GPT-5",
                status="open",
                yes_bid=45,
                yes_ask=50,
                no_bid=50,
                no_ask=55,
                last_price=48,
                volume=1000,
                volume_24h=200,
                open_time="2026-01-01T00:00:00Z",
                close_time="2026-12-31T23:59:59Z",
                expiration_time="2026-12-31T23:59:59Z",
            ),
            MarketInfo(
                ticker="KXTOPMODEL-CLAUDE5",
                event_ticker="KXTOPMODEL",
                series_ticker="KXTOPMODEL",
                title="Which AI model will be #1?",
                subtitle="Claude 5",
                status="open",
                yes_bid=30,
                yes_ask=35,
                no_bid=65,
                no_ask=70,
                last_price=33,
                volume=800,
                volume_24h=150,
                open_time="2026-01-01T00:00:00Z",
                close_time="2026-12-31T23:59:59Z",
                expiration_time="2026-12-31T23:59:59Z",
            ),
        ]
    client.get_markets.return_value = (markets, None)
    return client


class TestMarketDiscovery:
    async def test_auto_discovers_markets_when_none_provided(self) -> None:
        api = _mock_api_client()
        loop = TradingLoop(_config())
        await loop.initialize(api_client=api)

        assert loop.strategy is not None
        assert len(loop.strategy.contracts) == 2
        assert "KXTOPMODEL-GPT5" in loop.strategy.contracts
        assert "KXTOPMODEL-CLAUDE5" in loop.strategy.contracts

        # Verify contract details were populated
        gpt5 = loop.strategy.contracts["KXTOPMODEL-GPT5"]
        assert gpt5.model_name == "GPT-5"
        assert gpt5.yes_bid == 45
        assert gpt5.yes_ask == 50
        assert gpt5.last_price == 48

        await loop.shutdown()

    async def test_explicit_market_data_skips_discovery(self) -> None:
        api = _mock_api_client()
        explicit = {"markets": [{"ticker": "MANUAL-1", "subtitle": "ManualModel", "yes_bid": 10, "yes_ask": 15, "last_price": 12}]}
        loop = TradingLoop(_config())
        await loop.initialize(market_data=explicit, api_client=api)

        assert loop.strategy is not None
        assert "MANUAL-1" in loop.strategy.contracts
        # API should NOT have been called for discovery
        api.get_markets.assert_not_called()

        await loop.shutdown()

    async def test_discovery_survives_api_failure(self) -> None:
        api = MagicMock(spec=KalshiAPIClient)
        api.get_markets.side_effect = KalshiAPIError("Connection refused", status_code=503)

        loop = TradingLoop(_config())
        await loop.initialize(api_client=api)

        # Should initialize with no contracts rather than crash
        assert loop.strategy is not None
        assert len(loop.strategy.contracts) == 0

        await loop.shutdown()

    async def test_discovery_queries_all_target_series(self) -> None:
        api = _mock_api_client(markets=[])
        config = _config()
        config.leaderboard_alpha.target_series = ["KXTOPMODEL", "KXLLM1"]
        loop = TradingLoop(config)
        await loop.initialize(api_client=api)

        # Should have queried both series
        assert api.get_markets.call_count == 2
        calls = [c.kwargs for c in api.get_markets.call_args_list]
        queried_series = {c["series_ticker"] for c in calls}
        assert queried_series == {"KXTOPMODEL", "KXLLM1"}

        await loop.shutdown()

    async def test_api_client_creation_failure_is_graceful(self) -> None:
        loop = TradingLoop(_config())
        # Patch _create_api_client to simulate failure
        with patch.object(TradingLoop, "_create_api_client", return_value=None):
            await loop.initialize()

        # Should still start — just with no contracts
        assert loop.strategy is not None
        assert len(loop.strategy.contracts) == 0

        await loop.shutdown()

    async def test_discovered_contracts_receive_signals(self) -> None:
        """End-to-end: discovered contracts can match arena signals to tickers."""
        api = _mock_api_client()
        loop = TradingLoop(_config())
        await loop.initialize(api_client=api)

        assert loop._monitor is not None
        # Simulate a ranking_change signal for GPT-5 moving to #1
        sig = Signal(
            source="arena_monitor",
            timestamp=datetime.datetime(2026, 3, 1, 12, 0, 0),
            event_type="ranking_change",
            data={
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "rank": 1,
                "rank_ub": 1,
                "score": 1300.0,
            },
            relevant_series=["KXTOPMODEL"],
            urgency=SignalUrgency.HIGH,
        )
        loop._monitor.poll = AsyncMock(return_value=[sig])  # type: ignore[method-assign]

        await loop._tick()

        # Strategy should have generated a proposal and the engine should have orders
        assert loop.execution_engine is not None
        assert len(loop.execution_engine.orders) > 0

        # Verify the order targets the right contract
        order = list(loop.execution_engine.orders.values())[0]
        assert order.ticker == "KXTOPMODEL-GPT5"
        assert order.side == "yes"

        await loop.shutdown()

    async def test_market_info_to_dict(self) -> None:
        m = MarketInfo(
            ticker="T1",
            event_ticker="E1",
            series_ticker="S1",
            title="Title",
            subtitle="Sub",
            status="open",
            yes_bid=40,
            yes_ask=45,
            no_bid=55,
            no_ask=60,
            last_price=42,
            volume=100,
            volume_24h=50,
            open_time="2026-01-01",
            close_time="2026-12-31",
            expiration_time="2026-12-31",
        )
        d = TradingLoop._market_info_to_dict(m)
        assert d["ticker"] == "T1"
        assert d["subtitle"] == "Sub"
        assert d["yes_bid"] == 40
        assert d["yes_ask"] == 45
        assert d["last_price"] == 42
