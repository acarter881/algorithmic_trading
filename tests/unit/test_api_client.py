"""Unit tests for the Kalshi API client wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from kalshi_python import ApiException

from autotrader.api.client import (
    AmendRequest,
    BalanceInfo,
    FillInfo,
    KalshiAPIClient,
    KalshiAPIError,
    MarketInfo,
    Orderbook,
    OrderRequest,
    PositionInfo,
)
from autotrader.config.models import Environment, KalshiConfig


@pytest.fixture
def demo_config() -> KalshiConfig:
    return KalshiConfig(
        environment=Environment.DEMO,
        api_key_id="test-key-id",
        private_key_path="",
    )


@pytest.fixture
def client(demo_config: KalshiConfig) -> KalshiAPIClient:
    """Create a client with a mocked SDK underneath."""
    api_client = KalshiAPIClient(demo_config)
    api_client._client = MagicMock()
    return api_client


class TestClientInitialization:
    def test_default_is_demo(self, demo_config: KalshiConfig) -> None:
        client = KalshiAPIClient(demo_config)
        assert client.is_demo

    def test_production_config(self) -> None:
        config = KalshiConfig(environment=Environment.PRODUCTION)
        client = KalshiAPIClient(config)
        assert not client.is_demo

    def test_base_url_demo(self, demo_config: KalshiConfig) -> None:
        assert "demo-api.kalshi.co" in demo_config.base_url

    def test_base_url_production(self) -> None:
        config = KalshiConfig(environment=Environment.PRODUCTION)
        assert "api.elections.kalshi.com" in config.base_url

    def test_client_not_connected_raises(self, demo_config: KalshiConfig) -> None:
        api_client = KalshiAPIClient(demo_config)
        with pytest.raises(KalshiAPIError, match="not connected"):
            _ = api_client.client


class TestRetryLogic:
    def test_retries_on_429(self, client: KalshiAPIClient) -> None:
        """Should retry on rate limit (429) responses."""
        mock_fn = MagicMock()
        error_429 = ApiException(status=429, reason="Too Many Requests")
        error_429.headers = {}
        mock_fn.side_effect = [error_429, {"status": "ok"}]

        with patch("time.sleep"):
            result = client._call_with_retry("test", mock_fn)

        assert result == {"status": "ok"}
        assert mock_fn.call_count == 2

    def test_retries_on_500(self, client: KalshiAPIClient) -> None:
        """Should retry on server errors."""
        mock_fn = MagicMock()
        error_500 = ApiException(status=500, reason="Internal Server Error")
        error_500.headers = {}
        mock_fn.side_effect = [error_500, {"ok": True}]

        with patch("time.sleep"):
            result = client._call_with_retry("test", mock_fn)

        assert result == {"ok": True}
        assert mock_fn.call_count == 2

    def test_no_retry_on_400(self, client: KalshiAPIClient) -> None:
        """Should NOT retry on client errors like 400."""
        mock_fn = MagicMock()
        error_400 = ApiException(status=400, reason="Bad Request")
        error_400.body = "invalid params"
        error_400.headers = {}
        mock_fn.side_effect = error_400

        with pytest.raises(KalshiAPIError) as exc_info:
            client._call_with_retry("test", mock_fn)

        assert exc_info.value.status_code == 400
        assert mock_fn.call_count == 1

    def test_no_retry_on_401(self, client: KalshiAPIClient) -> None:
        """Should NOT retry on authentication errors."""
        mock_fn = MagicMock()
        error_401 = ApiException(status=401, reason="Unauthorized")
        error_401.body = ""
        error_401.headers = {}
        mock_fn.side_effect = error_401

        with pytest.raises(KalshiAPIError) as exc_info:
            client._call_with_retry("test", mock_fn)

        assert exc_info.value.status_code == 401

    def test_max_retries_exhausted(self, client: KalshiAPIClient) -> None:
        """Should fail after max retries."""
        mock_fn = MagicMock()
        error = ApiException(status=503, reason="Service Unavailable")
        error.body = ""
        error.headers = {}
        mock_fn.side_effect = error

        with patch("time.sleep"), pytest.raises(KalshiAPIError):
            client._call_with_retry("test", mock_fn)

        # Default max_retries=3, so 4 attempts total (initial + 3 retries)
        assert mock_fn.call_count == 4

    def test_retry_after_header_respected(self, client: KalshiAPIClient) -> None:
        """Should use Retry-After header value when available."""
        mock_fn = MagicMock()
        error_429 = ApiException(status=429, reason="Rate Limited")
        error_429.headers = {"Retry-After": "5"}
        mock_fn.side_effect = [error_429, {"ok": True}]

        with patch("time.sleep") as mock_sleep:
            client._call_with_retry("test", mock_fn)

        mock_sleep.assert_called_once_with(5)


class TestMarketData:
    def test_get_market(self, client: KalshiAPIClient) -> None:
        client.client.get_market.return_value = MagicMock(
            to_dict=lambda: {
                "market": {
                    "ticker": "KXTOPMODEL-26FEB28-CLAUDE",
                    "event_ticker": "KXTOPMODEL-26FEB28",
                    "series_ticker": "KXTOPMODEL",
                    "title": "Claude Opus 4.6",
                    "subtitle": "",
                    "status": "active",
                    "yes_bid": 65,
                    "yes_ask": 67,
                    "no_bid": 33,
                    "no_ask": 35,
                    "last_price": 66,
                    "volume": 15000,
                    "volume_24h": 3000,
                    "open_time": "2026-02-21T00:00:00Z",
                    "close_time": "2026-02-28T15:00:00Z",
                    "expiration_time": "2026-02-28T15:00:00Z",
                }
            }
        )

        market = client.get_market("KXTOPMODEL-26FEB28-CLAUDE")
        assert isinstance(market, MarketInfo)
        assert market.ticker == "KXTOPMODEL-26FEB28-CLAUDE"
        assert market.yes_bid == 65
        assert market.yes_ask == 67
        assert market.volume == 15000

    def test_get_markets_with_pagination(self, client: KalshiAPIClient) -> None:
        client.client.get_markets.return_value = MagicMock(
            to_dict=lambda: {
                "markets": [
                    {
                        "ticker": "T1",
                        "event_ticker": "E1",
                        "series_ticker": "S1",
                        "title": "Market 1",
                        "subtitle": "",
                        "status": "active",
                        "yes_bid": 50,
                        "yes_ask": 52,
                        "no_bid": 48,
                        "no_ask": 50,
                        "last_price": 51,
                        "volume": 100,
                        "volume_24h": 50,
                        "open_time": "",
                        "close_time": "",
                        "expiration_time": "",
                    }
                ],
                "cursor": "next-page-token",
            }
        )

        markets, cursor = client.get_markets(event_ticker="E1")
        assert len(markets) == 1
        assert markets[0].ticker == "T1"
        assert cursor == "next-page-token"

    def test_get_orderbook(self, client: KalshiAPIClient) -> None:
        client.client.get_market_orderbook.return_value = MagicMock(
            to_dict=lambda: {
                "orderbook": {
                    "yes": [[65, 100], [64, 200], [63, 50]],
                    "no": [[35, 150], [34, 100]],
                }
            }
        )

        ob = client.get_orderbook("KXTOPMODEL-26FEB28-CLAUDE")
        assert isinstance(ob, Orderbook)
        assert len(ob.yes_bids) == 3
        assert ob.yes_bids[0].price == 65
        assert ob.yes_bids[0].quantity == 100
        assert len(ob.no_bids) == 2

    def test_get_trades(self, client: KalshiAPIClient) -> None:
        client.client.get_trades.return_value = MagicMock(
            to_dict=lambda: {
                "trades": [{"trade_id": "t1", "ticker": "T1", "yes_price": 65, "count": 10}],
                "cursor": None,
            }
        )

        trades, cursor = client.get_trades(ticker="T1")
        assert len(trades) == 1
        assert trades[0]["yes_price"] == 65
        assert cursor is None


class TestTrading:
    def test_place_order(self, client: KalshiAPIClient) -> None:
        client.client.create_order.return_value = MagicMock(
            to_dict=lambda: {
                "order": {
                    "order_id": "oid-123",
                    "ticker": "KXTOPMODEL-26FEB28-CLAUDE",
                    "side": "yes",
                    "status": "resting",
                }
            }
        )

        order = OrderRequest(
            ticker="KXTOPMODEL-26FEB28-CLAUDE",
            side="yes",
            yes_price=65,
            count=10,
            client_order_id="my-order-001",
        )
        result = client.place_order(order)
        assert result["order"]["order_id"] == "oid-123"

        # Verify the SDK was called with correct params
        call_kwargs = client.client.create_order.call_args
        assert call_kwargs.kwargs["ticker"] == "KXTOPMODEL-26FEB28-CLAUDE"
        assert call_kwargs.kwargs["yes_price"] == 65
        assert call_kwargs.kwargs["count"] == 10

    def test_cancel_order(self, client: KalshiAPIClient) -> None:
        client.client.cancel_order.return_value = MagicMock(
            to_dict=lambda: {"order": {"order_id": "oid-123", "status": "cancelled"}}
        )

        result = client.cancel_order("oid-123")
        assert result["order"]["status"] == "cancelled"

    def test_amend_order(self, client: KalshiAPIClient) -> None:
        client.client.amend_order.return_value = MagicMock(
            to_dict=lambda: {"order": {"order_id": "oid-123", "yes_price": 70}}
        )

        request = AmendRequest(order_id="oid-123", yes_price=70)
        result = client.amend_order(request)
        assert result["order"]["yes_price"] == 70

    def test_batch_cancel(self, client: KalshiAPIClient) -> None:
        client.client.batch_cancel_orders.return_value = MagicMock(to_dict=lambda: {"cancelled": 3})

        result = client.batch_cancel_orders(["o1", "o2", "o3"])
        assert result["cancelled"] == 3


class TestPortfolio:
    def test_get_positions(self, client: KalshiAPIClient) -> None:
        client.client.get_positions.return_value = MagicMock(
            to_dict=lambda: {
                "market_positions": [
                    {
                        "ticker": "KXTOPMODEL-26FEB28-CLAUDE",
                        "event_ticker": "KXTOPMODEL-26FEB28",
                        "market_result": None,
                        "position": 10,
                        "total_cost": 650,
                        "realized_pnl": 0,
                        "fees_paid": 20,
                        "resting_order_count": 1,
                    }
                ],
                "cursor": None,
            }
        )

        positions, cursor = client.get_positions()
        assert len(positions) == 1
        assert isinstance(positions[0], PositionInfo)
        assert positions[0].position == 10
        assert positions[0].total_cost == 650

    def test_get_fills(self, client: KalshiAPIClient) -> None:
        client.client.get_fills.return_value = MagicMock(
            to_dict=lambda: {
                "fills": [
                    {
                        "trade_id": "fill-001",
                        "ticker": "T1",
                        "side": "yes",
                        "action": "buy",
                        "count": 5,
                        "yes_price": 65,
                        "no_price": 35,
                        "is_taker": True,
                        "created_time": "2026-02-26T12:00:00Z",
                        "order_id": "oid-123",
                    }
                ],
                "cursor": None,
            }
        )

        fills, cursor = client.get_fills()
        assert len(fills) == 1
        assert isinstance(fills[0], FillInfo)
        assert fills[0].count == 5
        assert fills[0].is_taker is True

    def test_get_balance(self, client: KalshiAPIClient) -> None:
        client.client.get_balance.return_value = MagicMock(
            to_dict=lambda: {
                "balance": 100000,
                "portfolio_value": 105000,
            }
        )

        balance = client.get_balance()
        assert isinstance(balance, BalanceInfo)
        assert balance.balance == 100000
        assert balance.portfolio_value == 105000


class TestEventDiscovery:
    def test_discover_active_events(self, client: KalshiAPIClient) -> None:
        """Should paginate through all active events."""
        client.client.get_events.side_effect = [
            MagicMock(
                to_dict=lambda: {
                    "events": [{"ticker": "KXTOPMODEL-26FEB28"}],
                    "cursor": "page2",
                }
            ),
            MagicMock(
                to_dict=lambda: {
                    "events": [{"ticker": "KXTOPMODEL-26MAR07"}],
                    "cursor": None,
                }
            ),
        ]

        events = client.discover_active_events("KXTOPMODEL")
        assert len(events) == 2
        assert events[0]["ticker"] == "KXTOPMODEL-26FEB28"
        assert events[1]["ticker"] == "KXTOPMODEL-26MAR07"

    def test_discover_markets_for_event(self, client: KalshiAPIClient) -> None:
        client.client.get_markets.return_value = MagicMock(
            to_dict=lambda: {
                "markets": [
                    {
                        "ticker": f"KXTOPMODEL-26FEB28-M{i}",
                        "event_ticker": "KXTOPMODEL-26FEB28",
                        "series_ticker": "KXTOPMODEL",
                        "title": f"Model {i}",
                        "subtitle": "",
                        "status": "active",
                        "yes_bid": 50,
                        "yes_ask": 52,
                        "no_bid": 48,
                        "no_ask": 50,
                        "last_price": 51,
                        "volume": 100,
                        "volume_24h": 50,
                        "open_time": "",
                        "close_time": "",
                        "expiration_time": "",
                    }
                    for i in range(3)
                ],
                "cursor": None,
            }
        )

        markets = client.discover_markets_for_event("KXTOPMODEL-26FEB28")
        assert len(markets) == 3
        assert all(isinstance(m, MarketInfo) for m in markets)


class TestHelpers:
    def test_to_dict_with_dict(self) -> None:
        assert KalshiAPIClient._to_dict({"a": 1}) == {"a": 1}

    def test_to_dict_with_object(self) -> None:
        obj = MagicMock()
        obj.to_dict.return_value = {"b": 2}
        assert KalshiAPIClient._to_dict(obj) == {"b": 2}

    def test_parse_market(self) -> None:
        data = {
            "ticker": "T1",
            "event_ticker": "E1",
            "series_ticker": "S1",
            "title": "Test",
            "subtitle": "Sub",
            "status": "active",
            "yes_bid": 60,
            "yes_ask": 62,
            "no_bid": 38,
            "no_ask": 40,
            "last_price": 61,
            "volume": 1000,
            "volume_24h": 200,
            "open_time": "2026-01-01",
            "close_time": "2026-02-01",
            "expiration_time": "2026-02-01",
            "result": "yes",
        }
        market = KalshiAPIClient._parse_market(data)
        assert market.ticker == "T1"
        assert market.result == "yes"

    def test_parse_market_missing_fields(self) -> None:
        """Should handle missing fields gracefully with defaults."""
        market = KalshiAPIClient._parse_market({})
        assert market.ticker == ""
        assert market.yes_bid == 0
        assert market.result is None
