"""Integration tests for the Kalshi API client.

These tests run against the real Kalshi Demo API.
They require:
- Network access to demo-api.kalshi.co
- Valid demo API credentials set in environment variables:
  AUTOTRADER__KALSHI__API_KEY_ID
  AUTOTRADER__KALSHI__PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY_PEM)

Run with: pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import os

import pytest

from autotrader.api.client import (
    KalshiAPIClient,
    MarketInfo,
)
from autotrader.config.models import Environment, KalshiConfig

# Skip all tests in this module if no API credentials are available
pytestmark = pytest.mark.integration

HAS_CREDENTIALS = bool(
    os.environ.get("AUTOTRADER__KALSHI__API_KEY_ID")
    and (os.environ.get("AUTOTRADER__KALSHI__PRIVATE_KEY_PATH") or os.environ.get("KALSHI_PRIVATE_KEY_PEM"))
)
skip_no_creds = pytest.mark.skipif(not HAS_CREDENTIALS, reason="No Kalshi API credentials available")


def _make_client() -> KalshiAPIClient:
    """Create a client connected to the Kalshi Demo API."""
    config = KalshiConfig(
        environment=Environment.DEMO,
        api_key_id=os.environ.get("AUTOTRADER__KALSHI__API_KEY_ID", ""),
        private_key_path=os.environ.get("AUTOTRADER__KALSHI__PRIVATE_KEY_PATH", ""),
    )
    client = KalshiAPIClient(config)

    pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
    client.connect(private_key_pem=pem)
    return client


@skip_no_creds
class TestDemoMarketData:
    """Test market data endpoints against the demo API."""

    def test_get_exchange_status(self) -> None:
        client = _make_client()
        status = client.get_exchange_status()
        assert isinstance(status, dict)

    def test_get_markets(self) -> None:
        client = _make_client()
        markets, cursor = client.get_markets(limit=5)
        assert isinstance(markets, list)
        assert len(markets) <= 5
        if markets:
            assert isinstance(markets[0], MarketInfo)
            assert markets[0].ticker

    def test_get_single_market(self) -> None:
        client = _make_client()
        # First, get any market to know a valid ticker
        markets, _ = client.get_markets(limit=1)
        if markets:
            market = client.get_market(markets[0].ticker)
            assert market.ticker == markets[0].ticker

    def test_get_orderbook(self) -> None:
        client = _make_client()
        markets, _ = client.get_markets(limit=1, status="active")
        if markets:
            ob = client.get_orderbook(markets[0].ticker)
            assert ob.ticker == markets[0].ticker

    def test_get_trades(self) -> None:
        client = _make_client()
        trades, cursor = client.get_trades(limit=5)
        assert isinstance(trades, list)


@skip_no_creds
class TestDemoPortfolio:
    """Test authenticated portfolio endpoints against the demo API."""

    def test_get_balance(self) -> None:
        client = _make_client()
        balance = client.get_balance()
        assert balance.balance >= 0

    def test_get_positions(self) -> None:
        client = _make_client()
        positions, cursor = client.get_positions()
        assert isinstance(positions, list)

    def test_get_fills(self) -> None:
        client = _make_client()
        fills, cursor = client.get_fills()
        assert isinstance(fills, list)

    def test_get_orders(self) -> None:
        client = _make_client()
        orders, cursor = client.get_orders()
        assert isinstance(orders, list)


@skip_no_creds
class TestDemoEventDiscovery:
    """Test event/market discovery against the demo API."""

    def test_discover_active_events(self) -> None:
        client = _make_client()
        # Use a series that likely exists on demo
        events = client.discover_active_events("KXTOPMODEL")
        assert isinstance(events, list)
        # May or may not have events on demo, just verify no errors

    def test_discover_markets_for_event(self) -> None:
        client = _make_client()
        events = client.discover_active_events("KXTOPMODEL")
        if events:
            markets = client.discover_markets_for_event(events[0]["ticker"])
            assert isinstance(markets, list)
