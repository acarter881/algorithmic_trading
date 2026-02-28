"""Unit tests for the trading loop and Discord alerter."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import ANY, AsyncMock, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autotrader.config.models import (
    AppConfig,
    ArenaMonitorConfig,
    DiscordConfig,
    Environment,
    KalshiConfig,
    LeaderboardAlphaConfig,
)
from autotrader.core.loop import TradingLoop
from autotrader.execution.engine import ExecutionMode
from autotrader.monitoring.discord import DiscordAlerter
from autotrader.risk.manager import PortfolioSnapshot, PositionInfo
from autotrader.signals.base import Signal, SignalUrgency
from autotrader.state.models import Base, Fill, Order, RiskEvent, SystemEvent
from autotrader.state.models import Signal as SignalRow
from autotrader.state.repository import TradingRepository
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

    async def test_initialize_live_mode_configures_kalshi_client(self, monkeypatch) -> None:
        mock_client = MagicMock()
        monkeypatch.setattr("autotrader.core.loop.KalshiAPIClient", MagicMock(return_value=mock_client))

        config = _config()
        config.kalshi = KalshiConfig(environment=Environment.PRODUCTION)

        loop = TradingLoop(config)
        await loop.initialize()

        assert mock_client.connect.call_count >= 1
        mock_client.connect.assert_any_call(private_key_pem=ANY)
        assert loop.execution_engine is not None
        assert loop.execution_engine.mode == ExecutionMode.LIVE
        assert loop.execution_engine._api is mock_client
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

    async def test_initialize_restores_positions_from_persistence(self) -> None:
        sf = _session_factory()
        repo = TradingRepository(sf)
        repo.record_fill(
            {
                "ticker": "KXTOPMODEL-GPT5",
                "side": "yes",
                "action": "buy",
                "count": 100,
                "price_cents": 40,
                "fee_cents": 0,
                "is_taker": True,
                "is_paper": True,
                "client_order_id": "leaderboard_alpha-restart-seed",
                "kalshi_fill_id": "fill-restart-seed",
                "filled_at": datetime.datetime.combine(datetime.date.today(), datetime.time(12, 0)).isoformat(),
            },
            strategy="leaderboard_alpha",
        )

        market_data = {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-GPT5",
                    "title": "Top model",
                    "subtitle": "GPT-5",
                    "yes_bid": 44,
                    "yes_ask": 46,
                    "last_price": 45,
                }
            ]
        }

        loop = TradingLoop(_config())
        await loop.initialize(market_data=market_data, session_factory=sf)

        assert loop.strategy is not None
        assert loop.strategy.contracts["KXTOPMODEL-GPT5"].position == 100

        snapshot = loop.build_portfolio_snapshot()
        assert len(snapshot.positions) == 1
        assert snapshot.positions[0].ticker == "KXTOPMODEL-GPT5"
        assert snapshot.positions[0].quantity == 100

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]
        loop.strategy.on_signal = AsyncMock(return_value=[_proposal()])  # type: ignore[method-assign]

        await loop._tick()

        assert loop.execution_engine is not None
        assert len(loop.execution_engine.orders) == 0
        await loop.shutdown()

    async def test_initialize_bootstraps_market_data_for_strategy(self, monkeypatch) -> None:
        market_a = MagicMock(
            ticker="KXTOPMODEL-GPT5",
            title="Top model",
            subtitle="GPT-5",
            yes_bid=44,
            yes_ask=46,
            last_price=45,
        )
        market_b = MagicMock(
            ticker="KXLLM1-CLAUDE",
            title="LLM #1",
            subtitle="Claude",
            yes_bid=39,
            yes_ask=41,
            last_price=40,
        )

        mock_client = MagicMock()
        mock_client.get_markets.side_effect = [
            ([market_a], None),
            ([market_b], None),
        ]
        monkeypatch.setattr("autotrader.core.loop.KalshiAPIClient", MagicMock(return_value=mock_client))

        config = _config()
        config.leaderboard_alpha.target_series = ["KXTOPMODEL", "KXLLM1"]

        loop = TradingLoop(config)
        await loop.initialize()

        assert loop.strategy is not None
        assert loop.strategy._resolve_ticker("GPT-5") == "KXTOPMODEL-GPT5"
        assert loop.strategy._resolve_ticker("Claude") == "KXLLM1-CLAUDE"

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

    async def test_tick_refreshes_portfolio_before_risk_evaluation(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[_proposal()])  # type: ignore[method-assign]

        snapshot = PortfolioSnapshot(balance_cents=123_456)
        loop.build_portfolio_snapshot = MagicMock(return_value=snapshot)  # type: ignore[method-assign]

        assert loop._risk is not None
        loop._risk.update_portfolio = MagicMock(wraps=loop._risk.update_portfolio)  # type: ignore[method-assign]
        original_evaluate = loop._risk.evaluate

        def _evaluate_with_assert(order: ProposedOrder):
            loop._risk.update_portfolio.assert_called_once_with(snapshot)
            assert loop._risk.portfolio is snapshot
            return original_evaluate(order)

        loop._risk.evaluate = MagicMock(side_effect=_evaluate_with_assert)  # type: ignore[method-assign]

        await loop._tick()

        loop.build_portfolio_snapshot.assert_called_once()
        loop._risk.update_portfolio.assert_called_once_with(snapshot)
        assert loop._risk.evaluate.call_count == 1
        await loop.shutdown()

    async def test_tick_uses_refreshed_snapshot_for_exposure_limits(self) -> None:
        loop = TradingLoop(_config())
        await loop.initialize()

        assert loop._monitor is not None
        loop._monitor.poll = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]

        assert loop._strategy is not None
        loop._strategy.on_signal = AsyncMock(return_value=[_proposal()])  # type: ignore[method-assign]

        assert loop._risk is not None
        loop._risk.update_portfolio(PortfolioSnapshot(balance_cents=100_000))

        refreshed_snapshot = PortfolioSnapshot(
            balance_cents=10_000,
            positions=[
                PositionInfo(
                    ticker="KXTOPMODEL-GPT5",
                    event_ticker="KXTOPMODEL",
                    quantity=180,
                    avg_cost_cents=50,
                )
            ],
        )
        loop.build_portfolio_snapshot = MagicMock(return_value=refreshed_snapshot)  # type: ignore[method-assign]

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
        snap = loop.build_portfolio_snapshot()
        assert snap.balance_cents > 0
        assert len(snap.positions) == 0
        await loop.shutdown()

    async def test_build_snapshot_preserves_signed_position_quantity(self) -> None:
        market_data = {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-GPT5",
                    "title": "Top model",
                    "subtitle": "GPT-5",
                    "yes_bid": 44,
                    "yes_ask": 46,
                    "last_price": 45,
                }
            ]
        }

        loop = TradingLoop(_config())
        await loop.initialize(market_data=market_data)
        assert loop.strategy is not None
        loop.strategy.contracts["KXTOPMODEL-GPT5"].position = -95

        snap = loop.build_portfolio_snapshot()

        assert len(snap.positions) == 1
        assert snap.positions[0].ticker == "KXTOPMODEL-GPT5"
        assert snap.positions[0].quantity == -95
        await loop.shutdown()

    async def test_build_snapshot_includes_persisted_daily_pnl(self) -> None:
        sf = _session_factory()
        loop = TradingLoop(_config())
        await loop.initialize(session_factory=sf)

        assert loop.repository is not None
        loop.repository.record_fill(
            {
                "ticker": "KXTOPMODEL-GPT5",
                "side": "yes",
                "action": "buy",
                "count": 5,
                "price_cents": 40,
                "fee_cents": 0,
                "is_taker": True,
                "is_paper": True,
                "client_order_id": "leaderboard_alpha-open",
                "kalshi_fill_id": "fill-open",
                "filled_at": datetime.datetime.combine(datetime.date.today(), datetime.time(12, 0)).isoformat(),
            },
            strategy="leaderboard_alpha",
        )
        loop.repository.record_fill(
            {
                "ticker": "KXTOPMODEL-GPT5",
                "side": "no",
                "action": "buy",
                "count": 5,
                "price_cents": 30,
                "fee_cents": 0,
                "is_taker": True,
                "is_paper": True,
                "client_order_id": "leaderboard_alpha-close",
                "kalshi_fill_id": "fill-close",
                "filled_at": datetime.datetime.combine(datetime.date.today(), datetime.time(12, 1)).isoformat(),
            },
            strategy="leaderboard_alpha",
        )

        snap = loop.build_portfolio_snapshot()

        assert snap.daily_realized_pnl_cents.get("leaderboard_alpha") == 150
        assert snap.daily_unrealized_pnl_cents.get("leaderboard_alpha") == 0
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
