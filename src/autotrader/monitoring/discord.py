"""Discord webhook alerter for trade notifications and system events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from autotrader.config.models import DiscordConfig

logger = structlog.get_logger("autotrader.monitoring.discord")


class DiscordAlerter:
    """Sends alerts to a Discord channel via webhook.

    Supports trade alerts, signal alerts, error alerts, and custom messages.
    All sends are fire-and-forget — failures are logged but never propagated.
    """

    def __init__(self, config: DiscordConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if not self._config.enabled or not self._config.webhook_url:
            logger.info("discord_alerter_disabled")
            return
        self._client = httpx.AsyncClient(timeout=10.0)
        logger.info("discord_alerter_initialized")

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.webhook_url)

    # ── Alert methods ─────────────────────────────────────────────────

    async def send_trade_alert(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price_cents: int,
        strategy: str,
        is_paper: bool = True,
    ) -> None:
        if not self._config.alert_on_trades:
            return
        mode = "PAPER" if is_paper else "LIVE"
        cost_dollars = price_cents * quantity / 100
        embed = {
            "title": f"{'BUY' if side == 'yes' else 'SELL'} {ticker}",
            "color": 0x00FF00 if side == "yes" else 0xFF6600,
            "fields": [
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Qty", "value": str(quantity), "inline": True},
                {"name": "Price", "value": f"{price_cents}c", "inline": True},
                {"name": "Cost", "value": f"${cost_dollars:.2f}", "inline": True},
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Mode", "value": mode, "inline": True},
            ],
        }
        if cost_dollars >= self._config.large_trade_threshold:
            embed["title"] = f"LARGE TRADE: {embed['title']}"
            embed["color"] = 0xFF0000
        await self._send(embeds=[embed])

    async def send_signal_alert(self, signal_type: str, data: dict[str, Any]) -> None:
        if not self._config.alert_on_signals:
            return
        description = "\n".join(f"**{k}**: {v}" for k, v in data.items())
        embed = {
            "title": f"Signal: {signal_type}",
            "description": description[:2000],
            "color": 0x3498DB,
        }
        await self._send(embeds=[embed])

    async def send_error_alert(self, error_type: str, details: str) -> None:
        if not self._config.alert_on_errors:
            return
        embed = {
            "title": f"Error: {error_type}",
            "description": details[:2000],
            "color": 0xFF0000,
        }
        await self._send(embeds=[embed])

    async def send_system_alert(self, title: str, message: str) -> None:
        embed = {
            "title": title,
            "description": message[:2000],
            "color": 0x9B59B6,
        }
        await self._send(embeds=[embed])

    # ── Internal ──────────────────────────────────────────────────────

    async def _send(self, content: str = "", embeds: list[dict[str, Any]] | None = None) -> None:
        if not self._client or not self.enabled:
            return
        payload: dict[str, Any] = {}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds
        try:
            resp = await self._client.post(self._config.webhook_url, json=payload)
            if resp.status_code >= 400:
                logger.warning("discord_send_failed", status=resp.status_code)
        except Exception:
            logger.exception("discord_send_error")
