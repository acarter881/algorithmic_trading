"""Unit tests for the trading loop and Discord alerter."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock

from autotrader.config.models import (
    AppConfig,
    ArenaMonitorConfig,
    DiscordConfig,
    LeaderboardAlphaConfig,
)
from autotrader.core.loop import TradingLoop
from autotrader.execution.engine import ExecutionMode
from autotrader.monitoring.discord import DiscordAlerter
from autotrader.signals.base import Signal, SignalUrgency
from autotrader.strategies.base import OrderUrgency, ProposedOrder

# ── Helpers ──────────────────────────────────────────────────────────────


def _config() -> AppConfig:
    return AppConfig(
        arena_monitor=ArenaMonitorConfig(poll_interval_seconds=1),
        leaderboard_alpha=LeaderboardAlphaConfig(target_series=["KXTOPMODEL"]),
    )


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


# ── DiscordAlerter ───────────────────────────────────────────────────────


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
