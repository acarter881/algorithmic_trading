"""Kalshi WebSocket client with auto-reconnect and channel subscription.

Kalshi WebSocket provides real-time streaming of:

Public channels (no auth required):
- ticker: Price/volume/open interest updates
- trade: Public trade feed
- market_lifecycle_v2: Market state transitions

Private channels (auth required):
- orderbook_delta: Orderbook snapshots and incremental deltas
- fill: Your fill notifications
- user_orders: Order lifecycle events
- market_positions: Position updates

Authentication is done via HTTP headers on the WebSocket handshake.
Kalshi sends ping frames every ~10 seconds; the websockets library
handles pong responses automatically.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = structlog.get_logger("autotrader.api.websocket")

# Type alias for message handler callbacks
MessageHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class Channel(StrEnum):
    """Available WebSocket subscription channels."""

    # Public
    TICKER = "ticker"
    TRADE = "trade"
    MARKET_LIFECYCLE = "market_lifecycle_v2"

    # Private (require authentication)
    ORDERBOOK_DELTA = "orderbook_delta"
    FILL = "fill"
    USER_ORDERS = "user_orders"
    MARKET_POSITIONS = "market_positions"
    ORDER_GROUP_UPDATES = "order_group_updates"


# Channels that don't require authentication
PUBLIC_CHANNELS = {Channel.TICKER, Channel.TRADE, Channel.MARKET_LIFECYCLE}


@dataclass
class Subscription:
    """A channel subscription with optional market tickers."""

    channel: Channel
    tickers: list[str] = field(default_factory=list)  # Empty = all tickers for channel
    sid: int | None = None  # Server-assigned subscription ID, set on confirmation


class KalshiWebSocketClient:
    """WebSocket client for Kalshi real-time data.

    Provides auto-reconnect, heartbeat management, and channel subscriptions.
    Uses the websockets library for the underlying connection.

    The websockets library automatically handles Kalshi's ping/pong heartbeat
    frames (sent every ~10 seconds by the server).
    """

    PROD_WS_URL = "wss://trading-api.kalshi.com/trade-api/ws/v2"
    DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    # Reconnect parameters
    MAX_RECONNECT_DELAY = 60
    INITIAL_RECONNECT_DELAY = 1

    def __init__(
        self,
        api_key_id: str = "",
        private_key_pem: str = "",
        is_demo: bool = True,
    ) -> None:
        self._api_key_id = api_key_id
        self._private_key_pem = private_key_pem
        self._ws_url = self.DEMO_WS_URL if is_demo else self.PROD_WS_URL
        self._is_demo = is_demo

        self._ws: Any = None  # websockets connection
        self._subscriptions: list[Subscription] = []
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._running = False
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
        self._connected = False
        self._cmd_counter = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def _next_cmd_id(self) -> int:
        """Generate a unique command ID."""
        self._cmd_counter += 1
        return self._cmd_counter

    # ── Authentication ───────────────────────────────────────────────────

    def _generate_auth_headers(self) -> dict[str, str]:
        """Generate Kalshi RSA-PSS authentication headers for WebSocket handshake.

        Signature format: "{timestamp_ms}GET/trade-api/ws/v2"
        Signed with RSA-PSS SHA-256, salt_length = SHA256 digest size (32 bytes).
        """
        if not self._api_key_id or not self._private_key_pem:
            return {}

        timestamp_ms = str(int(time.time() * 1000))
        msg = f"{timestamp_ms}GET/trade-api/ws/v2"

        private_key = serialization.load_pem_private_key(
            self._private_key_pem.encode(),
            password=None,
        )
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise TypeError("Private key must be RSA")

        signature = private_key.sign(
            msg.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )
        sig_b64 = base64.b64encode(signature).decode()

        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    # ── Subscription Management ──────────────────────────────────────────

    def subscribe(self, channel: Channel, tickers: list[str] | None = None) -> None:
        """Add a channel subscription. Takes effect on next connect or immediately if connected."""
        sub = Subscription(channel=channel, tickers=tickers or [])
        self._subscriptions.append(sub)
        logger.info("subscription_added", channel=channel.value, tickers=tickers)

    def on_message(self, channel: str, handler: MessageHandler) -> None:
        """Register a handler for messages on a specific channel.

        Use channel="*" to receive all messages (catch-all).
        Use the message type string (e.g., "orderbook_snapshot", "orderbook_delta",
        "ticker", "trade", "fill") to match specific message types.
        """
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

    async def _send_subscriptions(self) -> None:
        """Send all pending subscription commands to the WebSocket."""
        if not self._ws:
            return
        for sub in self._subscriptions:
            cmd: dict[str, Any] = {
                "id": self._next_cmd_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": [sub.channel.value],
                },
            }
            if sub.tickers:
                cmd["params"]["market_tickers"] = sub.tickers
            await self._ws.send(json.dumps(cmd))
            logger.debug("subscription_sent", channel=sub.channel.value, tickers=sub.tickers)

    async def update_subscription(
        self,
        sid: int,
        tickers: list[str],
        action: str = "add_markets",
    ) -> None:
        """Add or remove market tickers from an existing subscription.

        Args:
            sid: Server-assigned subscription ID.
            tickers: Market tickers to add or remove.
            action: "add_markets" or "delete_markets".
        """
        if not self._ws:
            return
        cmd = {
            "id": self._next_cmd_id(),
            "cmd": "update_subscription",
            "params": {
                "sid": sid,
                "market_tickers": tickers,
                "action": action,
            },
        }
        await self._ws.send(json.dumps(cmd))
        logger.debug("subscription_updated", sid=sid, action=action, tickers=tickers)

    async def unsubscribe(self, sids: list[int]) -> None:
        """Unsubscribe from subscriptions by their server-assigned IDs."""
        if not self._ws:
            return
        cmd = {
            "id": self._next_cmd_id(),
            "cmd": "unsubscribe",
            "params": {
                "sids": sids,
            },
        }
        await self._ws.send(json.dumps(cmd))
        logger.debug("unsubscribe_sent", sids=sids)

    # ── Connection Lifecycle ─────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish WebSocket connection and start message loop.

        This is the main entry point. It handles:
        - Connection with RSA-PSS authentication headers
        - Sending subscriptions
        - Message routing to registered handlers
        - Auto-reconnect with exponential backoff on disconnection

        Kalshi's ping/pong heartbeat is handled automatically by the websockets library.
        """
        import websockets

        self._running = True
        while self._running:
            try:
                headers = self._generate_auth_headers()
                logger.info("websocket_connecting", url=self._ws_url, authenticated=bool(headers))

                async with websockets.connect(
                    self._ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
                    logger.info("websocket_connected")

                    # Re-send subscriptions on connect/reconnect
                    await self._send_subscriptions()

                    # Message loop
                    async for raw_message in ws:
                        try:
                            if isinstance(raw_message, bytes):
                                raw_message = raw_message.decode("utf-8")
                            message = json.loads(raw_message)
                            await self._dispatch_message(message)
                        except json.JSONDecodeError:
                            logger.warning("websocket_invalid_json", raw=str(raw_message)[:200])

            except asyncio.CancelledError:
                logger.info("websocket_cancelled")
                self._running = False
                break
            except Exception as e:
                self._connected = False
                self._ws = None
                if not self._running:
                    break
                logger.warning(
                    "websocket_disconnected",
                    error=str(e),
                    reconnect_delay=self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

        self._connected = False
        self._ws = None

    async def disconnect(self) -> None:
        """Gracefully disconnect the WebSocket."""
        self._running = False
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._connected = False
        self._ws = None
        logger.info("websocket_disconnected_gracefully")

    # ── Message Dispatch ─────────────────────────────────────────────────

    async def _dispatch_message(self, message: dict[str, Any]) -> None:
        """Route an incoming message to registered handlers.

        Message envelope: {"type": "<msg_type>", "sid": <int>, "seq": <int>, "msg": {...}}
        Subscription confirmations: {"id": <cmd_id>, "type": "ok", "msg": [...]}
        Errors: {"id": <cmd_id>, "type": "error", "msg": "...", "code": <int>}
        """
        msg_type = message.get("type", "")

        # Handle subscription confirmations: {"type": "ok", "msg": [{"channel": ..., "sid": ...}]}
        if msg_type == "ok":
            confirmed = message.get("msg", [])
            if isinstance(confirmed, list):
                for entry in confirmed:
                    ch = entry.get("channel", "")
                    sid = entry.get("sid")
                    # Update our subscription records with the server-assigned SID
                    for sub in self._subscriptions:
                        if sub.channel.value == ch and sub.sid is None:
                            sub.sid = sid
                            break
                    logger.debug("subscription_confirmed", channel=ch, sid=sid)
            return

        if msg_type == "error":
            logger.error("websocket_server_error", message=message)
            return

        # Data messages — dispatch by message type
        handlers = self._handlers.get(msg_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception:
                logger.exception("websocket_handler_error", msg_type=msg_type)

        # Also dispatch to catch-all handlers
        all_handlers = self._handlers.get("*", [])
        for handler in all_handlers:
            try:
                await handler(message)
            except Exception:
                logger.exception("websocket_catchall_handler_error", msg_type=msg_type)
