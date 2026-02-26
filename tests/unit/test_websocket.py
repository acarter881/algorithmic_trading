"""Unit tests for the Kalshi WebSocket client."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from autotrader.api.websocket import (
    PUBLIC_CHANNELS,
    Channel,
    KalshiWebSocketClient,
    Subscription,
)


class TestChannels:
    def test_public_channels(self) -> None:
        assert Channel.TICKER in PUBLIC_CHANNELS
        assert Channel.TRADE in PUBLIC_CHANNELS
        assert Channel.MARKET_LIFECYCLE in PUBLIC_CHANNELS

    def test_private_channels(self) -> None:
        assert Channel.ORDERBOOK_DELTA not in PUBLIC_CHANNELS
        assert Channel.FILL not in PUBLIC_CHANNELS
        assert Channel.USER_ORDERS not in PUBLIC_CHANNELS


class TestSubscription:
    def test_default_subscription(self) -> None:
        sub = Subscription(channel=Channel.TICKER)
        assert sub.channel == Channel.TICKER
        assert sub.tickers == []
        assert sub.sid is None

    def test_subscription_with_tickers(self) -> None:
        sub = Subscription(channel=Channel.ORDERBOOK_DELTA, tickers=["KXTOPMODEL-26FEB28-CLAUDE"])
        assert len(sub.tickers) == 1


class TestWebSocketClientInit:
    def test_demo_url(self) -> None:
        client = KalshiWebSocketClient(is_demo=True)
        assert "demo-api.kalshi.co" in client._ws_url

    def test_prod_url(self) -> None:
        client = KalshiWebSocketClient(is_demo=False)
        assert "trading-api.kalshi.com" in client._ws_url

    def test_not_connected_initially(self) -> None:
        client = KalshiWebSocketClient()
        assert not client.connected

    def test_cmd_counter_increments(self) -> None:
        client = KalshiWebSocketClient()
        id1 = client._next_cmd_id()
        id2 = client._next_cmd_id()
        assert id2 == id1 + 1


class TestAuthHeaders:
    def test_no_headers_without_credentials(self) -> None:
        client = KalshiWebSocketClient(api_key_id="", private_key_pem="")
        headers = client._generate_auth_headers()
        assert headers == {}

    def test_headers_with_credentials(self) -> None:
        # Generate a test RSA key
        from cryptography.hazmat.primitives.asymmetric import rsa as rsa_mod

        private_key = rsa_mod.generate_private_key(public_exponent=65537, key_size=2048)
        from cryptography.hazmat.primitives import serialization

        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        client = KalshiWebSocketClient(api_key_id="test-key", private_key_pem=pem)
        headers = client._generate_auth_headers()

        assert "KALSHI-ACCESS-KEY" in headers
        assert headers["KALSHI-ACCESS-KEY"] == "test-key"
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        # Timestamp should be a numeric string (milliseconds)
        assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()


class TestSubscriptionManagement:
    def test_subscribe_adds_to_list(self) -> None:
        client = KalshiWebSocketClient()
        client.subscribe(Channel.TICKER, tickers=["T1", "T2"])
        assert len(client._subscriptions) == 1
        assert client._subscriptions[0].channel == Channel.TICKER
        assert client._subscriptions[0].tickers == ["T1", "T2"]

    def test_multiple_subscriptions(self) -> None:
        client = KalshiWebSocketClient()
        client.subscribe(Channel.TICKER)
        client.subscribe(Channel.TRADE)
        client.subscribe(Channel.FILL)
        assert len(client._subscriptions) == 3

    def test_on_message_registers_handler(self) -> None:
        client = KalshiWebSocketClient()

        async def handler(msg: dict[str, Any]) -> None:
            pass

        client.on_message("ticker", handler)
        assert "ticker" in client._handlers
        assert len(client._handlers["ticker"]) == 1

    def test_on_message_multiple_handlers(self) -> None:
        client = KalshiWebSocketClient()

        async def handler1(msg: dict[str, Any]) -> None:
            pass

        async def handler2(msg: dict[str, Any]) -> None:
            pass

        client.on_message("ticker", handler1)
        client.on_message("ticker", handler2)
        assert len(client._handlers["ticker"]) == 2


class TestSendSubscriptions:
    @pytest.mark.asyncio
    async def test_sends_subscription_json(self) -> None:
        client = KalshiWebSocketClient()
        client.subscribe(Channel.TICKER, tickers=["T1"])

        mock_ws = AsyncMock()
        client._ws = mock_ws

        await client._send_subscriptions()

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["cmd"] == "subscribe"
        assert sent["params"]["channels"] == ["ticker"]
        assert sent["params"]["market_tickers"] == ["T1"]

    @pytest.mark.asyncio
    async def test_no_tickers_omits_market_tickers(self) -> None:
        client = KalshiWebSocketClient()
        client.subscribe(Channel.TRADE)

        mock_ws = AsyncMock()
        client._ws = mock_ws

        await client._send_subscriptions()

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert "market_tickers" not in sent["params"]

    @pytest.mark.asyncio
    async def test_noop_when_no_ws(self) -> None:
        client = KalshiWebSocketClient()
        client.subscribe(Channel.TICKER)
        # _ws is None, should not raise
        await client._send_subscriptions()


class TestUpdateSubscription:
    @pytest.mark.asyncio
    async def test_update_add_markets(self) -> None:
        client = KalshiWebSocketClient()
        mock_ws = AsyncMock()
        client._ws = mock_ws

        await client.update_subscription(sid=1, tickers=["NEW-T1"], action="add_markets")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["cmd"] == "update_subscription"
        assert sent["params"]["sid"] == 1
        assert sent["params"]["action"] == "add_markets"
        assert sent["params"]["market_tickers"] == ["NEW-T1"]

    @pytest.mark.asyncio
    async def test_update_delete_markets(self) -> None:
        client = KalshiWebSocketClient()
        mock_ws = AsyncMock()
        client._ws = mock_ws

        await client.update_subscription(sid=2, tickers=["OLD-T1"], action="delete_markets")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["params"]["action"] == "delete_markets"


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_by_sids(self) -> None:
        client = KalshiWebSocketClient()
        mock_ws = AsyncMock()
        client._ws = mock_ws

        await client.unsubscribe(sids=[1, 2, 3])

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["cmd"] == "unsubscribe"
        assert sent["params"]["sids"] == [1, 2, 3]


class TestMessageDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_ok_confirmation(self) -> None:
        """Subscription confirmations (type=ok) should update subscription SIDs."""
        client = KalshiWebSocketClient()
        client.subscribe(Channel.TICKER)

        message = {
            "id": 1,
            "type": "ok",
            "msg": [{"channel": "ticker", "sid": 42}],
        }
        await client._dispatch_message(message)
        assert client._subscriptions[0].sid == 42

    @pytest.mark.asyncio
    async def test_dispatch_data_message(self) -> None:
        """Data messages should be routed to the correct handler."""
        client = KalshiWebSocketClient()
        received: list[dict[str, Any]] = []

        async def handler(msg: dict[str, Any]) -> None:
            received.append(msg)

        client.on_message("ticker", handler)

        message = {
            "type": "ticker",
            "sid": 1,
            "seq": 100,
            "msg": {"market_ticker": "T1", "price": 65},
        }
        await client._dispatch_message(message)

        assert len(received) == 1
        assert received[0]["msg"]["price"] == 65

    @pytest.mark.asyncio
    async def test_dispatch_catchall_handler(self) -> None:
        """Catch-all handler (*) should receive all data messages."""
        client = KalshiWebSocketClient()
        received: list[dict[str, Any]] = []

        async def handler(msg: dict[str, Any]) -> None:
            received.append(msg)

        client.on_message("*", handler)

        await client._dispatch_message({"type": "ticker", "msg": {}})
        await client._dispatch_message({"type": "trade", "msg": {}})
        await client._dispatch_message({"type": "fill", "msg": {}})

        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_dispatch_error_message(self) -> None:
        """Error messages should be logged, not dispatched to handlers."""
        client = KalshiWebSocketClient()
        received: list[dict[str, Any]] = []

        async def handler(msg: dict[str, Any]) -> None:
            received.append(msg)

        client.on_message("error", handler)

        message = {"type": "error", "msg": "Rate limited", "code": 6}
        await client._dispatch_message(message)

        # Error messages are logged but not dispatched to user handlers
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_isolated(self) -> None:
        """A handler exception should not prevent other handlers from running."""
        client = KalshiWebSocketClient()
        received: list[str] = []

        async def bad_handler(msg: dict[str, Any]) -> None:
            raise ValueError("test error")

        async def good_handler(msg: dict[str, Any]) -> None:
            received.append("ok")

        client.on_message("ticker", bad_handler)
        client.on_message("ticker", good_handler)

        await client._dispatch_message({"type": "ticker", "msg": {}})
        assert received == ["ok"]

    @pytest.mark.asyncio
    async def test_dispatch_orderbook_snapshot(self) -> None:
        """orderbook_snapshot messages should be dispatched to handlers."""
        client = KalshiWebSocketClient()
        received: list[dict[str, Any]] = []

        async def handler(msg: dict[str, Any]) -> None:
            received.append(msg)

        client.on_message("orderbook_snapshot", handler)

        message = {
            "type": "orderbook_snapshot",
            "sid": 1,
            "msg": {
                "market_ticker": "T1",
                "yes": [[65, 100], [64, 200]],
                "no": [[35, 150]],
            },
        }
        await client._dispatch_message(message)
        assert len(received) == 1


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self) -> None:
        client = KalshiWebSocketClient()
        client._running = True
        client._connected = True
        client._ws = AsyncMock()

        await client.disconnect()

        assert not client._running
        assert not client._connected
        assert client._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_close_error(self) -> None:
        """Should not raise even if ws.close() fails."""
        client = KalshiWebSocketClient()
        client._running = True
        client._connected = True
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = Exception("close failed")
        client._ws = mock_ws

        await client.disconnect()
        assert not client._connected
