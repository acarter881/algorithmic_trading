"""Kalshi API client wrapper with retries, rate limiting, and error handling."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from kalshi_python import ApiException, Configuration, KalshiClient  # type: ignore[attr-defined]

from autotrader.config.models import Environment, KalshiConfig

logger = structlog.get_logger("autotrader.api.client")


@dataclass
class OrderRequest:
    """Parameters for placing an order."""

    ticker: str
    side: str  # "yes" or "no"
    action: str = "buy"  # "buy" or "sell"
    type: str = "limit"
    count: int = 1
    yes_price: int | None = None
    no_price: int | None = None
    client_order_id: str | None = None
    expiration_ts: int | None = None
    self_trade_prevention_type: str | None = None
    time_in_force: str | None = None  # "gtc" or "ioc"


@dataclass
class AmendRequest:
    """Parameters for amending an order."""

    order_id: str
    count: int | None = None
    yes_price: int | None = None
    no_price: int | None = None


@dataclass
class MarketInfo:
    """Parsed market data."""

    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: str
    status: str
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    last_price: int
    volume: int
    volume_24h: int
    open_time: str
    close_time: str
    expiration_time: str
    result: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderbookLevel:
    """A single price level in the orderbook."""

    price: int  # cents
    quantity: int


@dataclass
class Orderbook:
    """Parsed orderbook for a market."""

    ticker: str
    yes_bids: list[OrderbookLevel]
    yes_asks: list[OrderbookLevel]
    no_bids: list[OrderbookLevel]
    no_asks: list[OrderbookLevel]


@dataclass
class PositionInfo:
    """Current position in a market."""

    ticker: str
    event_ticker: str
    market_result: str | None
    position: int
    total_cost: int
    realized_pnl: int
    fees_paid: int
    resting_order_count: int


@dataclass
class FillInfo:
    """A single fill (execution)."""

    trade_id: str
    ticker: str
    side: str
    action: str
    count: int
    yes_price: int
    no_price: int
    is_taker: bool
    created_time: str
    order_id: str


@dataclass
class BalanceInfo:
    """Account balance information."""

    balance: int  # cents
    portfolio_value: int  # cents


class KalshiAPIError(Exception):
    """Wrapper for Kalshi API errors with status and detail."""

    def __init__(self, message: str, status_code: int = 0, detail: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class KalshiAPIClient:
    """High-level wrapper around the Kalshi Python SDK.

    Provides:
    - Automatic retry with exponential backoff on transient errors
    - Rate limit awareness (backs off on 429)
    - Environment toggle (demo/production)
    - Structured logging of all API interactions
    - Typed return values for market data and trading operations
    """

    # Status codes that trigger retry
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, config: KalshiConfig) -> None:
        self._config = config
        self._client: KalshiClient | None = None
        self._last_request_time: float = 0.0

    def connect(self, private_key_pem: str | None = None) -> None:
        """Initialize the SDK client and authenticate.

        Args:
            private_key_pem: PEM-encoded private key string. If None, reads from
                             the path specified in config.
        """
        sdk_config = Configuration()
        sdk_config.host = self._config.base_url

        if private_key_pem:
            pem = private_key_pem
        elif self._config.private_key_path:
            with open(self._config.private_key_path) as f:
                pem = f.read()
        else:
            pem = ""

        if pem and self._config.api_key_id:
            sdk_config.api_key_id = self._config.api_key_id
            sdk_config.private_key_pem = pem

        self._client = KalshiClient(configuration=sdk_config)  # type: ignore[no-untyped-call]
        logger.info(
            "kalshi_client_connected",
            environment=self._config.environment.value,
            base_url=self._config.base_url,
            authenticated=bool(pem and self._config.api_key_id),
        )

    @property
    def client(self) -> KalshiClient:
        """Get the underlying SDK client, ensuring it's connected."""
        if self._client is None:
            raise KalshiAPIError("Client not connected. Call connect() first.")
        return self._client

    @property
    def is_demo(self) -> bool:
        return self._config.environment == Environment.DEMO

    # ── Retry Logic ──────────────────────────────────────────────────────

    def _call_with_retry(self, method_name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Call an API function with retry on transient errors.

        Uses exponential backoff: 1s, 2s, 4s, etc.
        """
        max_retries = self._config.max_retries
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except ApiException as e:
                if e.status in self.RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait = 2**attempt
                    if e.status == 429:
                        # Rate limited — extract Retry-After if available
                        retry_after = None
                        if e.headers:
                            retry_after_str = e.headers.get("Retry-After")
                            if retry_after_str:
                                with contextlib.suppress(ValueError):
                                    retry_after = int(retry_after_str)
                        wait = retry_after if retry_after else max(wait, 2)
                    logger.warning(
                        "api_call_retrying",
                        method=method_name,
                        status=e.status,
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "api_call_failed",
                        method=method_name,
                        status=e.status,
                        reason=e.reason,
                        body=str(e.body)[:500] if e.body else "",
                    )
                    raise KalshiAPIError(
                        f"API call {method_name} failed: {e.reason}",
                        status_code=e.status,
                        detail=str(e.body) if e.body else "",
                    ) from e
        # Should not reach here, but just in case
        raise KalshiAPIError(f"API call {method_name} failed after {max_retries} retries")

    # ── Market Data ──────────────────────────────────────────────────────

    def get_exchange_status(self) -> dict[str, Any]:
        """Get exchange operating status."""
        result = self._call_with_retry(
            "get_exchange_status",
            self.client.get_exchange_status,
        )
        return self._to_dict(result)

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        """Get series metadata."""
        result = self._call_with_retry(
            "get_series",
            self.client.get_series,
            series_ticker,
        )
        return self._to_dict(result)

    def get_event(self, event_ticker: str) -> dict[str, Any]:
        """Get event details including all markets/contracts."""
        result = self._call_with_retry(
            "get_event",
            self.client.get_event,
            event_ticker,
        )
        return self._to_dict(result)

    def get_events(
        self,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List events with optional filters."""
        kwargs: dict[str, Any] = {"limit": limit}
        if series_ticker:
            kwargs["series_ticker"] = series_ticker
        if status:
            kwargs["status"] = status
        if cursor:
            kwargs["cursor"] = cursor
        result = self._call_with_retry(
            "get_events",
            self.client.get_events,
            **kwargs,
        )
        return self._to_dict(result)

    def get_market(self, ticker: str) -> MarketInfo:
        """Get a single market's details."""
        result = self._call_with_retry(
            "get_market",
            self.client.get_market,
            ticker,
        )
        data = self._to_dict(result)
        market = data.get("market", data)
        return self._parse_market(market)

    def get_markets(
        self,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> tuple[list[MarketInfo], str | None]:
        """List markets with optional filters. Returns (markets, next_cursor)."""
        kwargs: dict[str, Any] = {"limit": limit}
        if event_ticker:
            kwargs["event_ticker"] = event_ticker
        if series_ticker:
            kwargs["series_ticker"] = series_ticker
        if status:
            kwargs["status"] = status
        if cursor:
            kwargs["cursor"] = cursor
        result = self._call_with_retry(
            "get_markets",
            self.client.get_markets,
            **kwargs,
        )
        data = self._to_dict(result)
        markets = [self._parse_market(m) for m in data.get("markets", [])]
        next_cursor = data.get("cursor")
        return markets, next_cursor

    def get_orderbook(self, ticker: str, depth: int | None = None) -> Orderbook:
        """Get the orderbook for a market."""
        kwargs: dict[str, Any] = {}
        if depth is not None:
            kwargs["depth"] = depth
        result = self._call_with_retry(
            "get_market_orderbook",
            self.client.get_market_orderbook,
            ticker,
            **kwargs,
        )
        data = self._to_dict(result)
        ob = data.get("orderbook", data)
        return Orderbook(
            ticker=ticker,
            yes_bids=[OrderbookLevel(price=lvl[0], quantity=lvl[1]) for lvl in (ob.get("yes") or [])],
            yes_asks=[],  # Derived from no_bids: yes_ask = 100 - no_bid
            no_bids=[OrderbookLevel(price=lvl[0], quantity=lvl[1]) for lvl in (ob.get("no") or [])],
            no_asks=[],  # Derived from yes_bids: no_ask = 100 - yes_bid
        )

    def get_trades(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Get recent trades. Returns (trades, next_cursor)."""
        kwargs: dict[str, Any] = {"limit": limit}
        if ticker:
            kwargs["ticker"] = ticker
        if cursor:
            kwargs["cursor"] = cursor
        result = self._call_with_retry(
            "get_trades",
            self.client.get_trades,
            **kwargs,
        )
        data = self._to_dict(result)
        return data.get("trades", []), data.get("cursor")

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        period_interval: int = 60,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get OHLC candlestick data."""
        result = self._call_with_retry(
            "get_market_candlesticks",
            self.client.get_market_candlesticks,
            series_ticker=series_ticker,
            ticker=ticker,
            period_interval=period_interval,
        )
        data = self._to_dict(result)
        candlesticks: list[dict[str, Any]] = data.get("candlesticks", [])
        return candlesticks

    # ── Trading ──────────────────────────────────────────────────────────

    def place_order(self, order: OrderRequest) -> dict[str, Any]:
        """Place a new order. Returns order data from the API."""
        kwargs: dict[str, Any] = {
            "ticker": order.ticker,
            "side": order.side,
            "action": order.action,
            "type": order.type,
            "count": order.count,
        }
        if order.yes_price is not None:
            kwargs["yes_price"] = order.yes_price
        if order.no_price is not None:
            kwargs["no_price"] = order.no_price
        if order.client_order_id:
            kwargs["client_order_id"] = order.client_order_id
        if order.expiration_ts is not None:
            kwargs["expiration_ts"] = order.expiration_ts
        if order.self_trade_prevention_type:
            kwargs["self_trade_prevention_type"] = order.self_trade_prevention_type
        if order.time_in_force:
            kwargs["time_in_force"] = order.time_in_force

        logger.info("placing_order", **{k: v for k, v in kwargs.items() if k != "self_trade_prevention_type"})
        result = self._call_with_retry(
            "create_order",
            self.client.create_order,
            **kwargs,
        )
        data = self._to_dict(result)
        logger.info("order_placed", order_id=data.get("order", {}).get("order_id"))
        return data

    def amend_order(self, request: AmendRequest) -> dict[str, Any]:
        """Amend a resting order."""
        kwargs: dict[str, Any] = {"order_id": request.order_id}
        if request.count is not None:
            kwargs["count"] = request.count
        if request.yes_price is not None:
            kwargs["yes_price"] = request.yes_price
        if request.no_price is not None:
            kwargs["no_price"] = request.no_price

        logger.info("amending_order", **kwargs)
        result = self._call_with_retry(
            "amend_order",
            self.client.amend_order,
            **kwargs,
        )
        return self._to_dict(result)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a resting order."""
        logger.info("cancelling_order", order_id=order_id)
        result = self._call_with_retry(
            "cancel_order",
            self.client.cancel_order,
            order_id,
        )
        return self._to_dict(result)

    def batch_cancel_orders(self, order_ids: list[str]) -> dict[str, Any]:
        """Cancel multiple orders at once."""
        logger.info("batch_cancelling_orders", count=len(order_ids))
        result = self._call_with_retry(
            "batch_cancel_orders",
            self.client.batch_cancel_orders,
            ids=order_ids,
        )
        return self._to_dict(result)

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get a single order's details."""
        result = self._call_with_retry(
            "get_order",
            self.client.get_order,
            order_id,
        )
        return self._to_dict(result)

    def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """List orders with optional filters."""
        kwargs: dict[str, Any] = {"limit": limit}
        if ticker:
            kwargs["ticker"] = ticker
        if status:
            kwargs["status"] = status
        if cursor:
            kwargs["cursor"] = cursor
        result = self._call_with_retry(
            "get_orders",
            self.client.get_orders,
            **kwargs,
        )
        data = self._to_dict(result)
        return data.get("orders", []), data.get("cursor")

    # ── Portfolio ────────────────────────────────────────────────────────

    def get_positions(
        self,
        ticker: str | None = None,
        event_ticker: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> tuple[list[PositionInfo], str | None]:
        """Get current positions."""
        kwargs: dict[str, Any] = {"limit": limit}
        if ticker:
            kwargs["ticker"] = ticker
        if event_ticker:
            kwargs["event_ticker"] = event_ticker
        if cursor:
            kwargs["cursor"] = cursor
        result = self._call_with_retry(
            "get_positions",
            self.client.get_positions,
            **kwargs,
        )
        data = self._to_dict(result)
        positions = []
        for p in data.get("market_positions", data.get("positions", [])):
            positions.append(
                PositionInfo(
                    ticker=p.get("ticker", ""),
                    event_ticker=p.get("event_ticker", ""),
                    market_result=p.get("market_result"),
                    position=p.get("position", 0),
                    total_cost=p.get("total_cost", 0),
                    realized_pnl=p.get("realized_pnl", 0),
                    fees_paid=p.get("fees_paid", 0),
                    resting_order_count=p.get("resting_order_count", 0),
                )
            )
        return positions, data.get("cursor")

    def get_fills(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[FillInfo], str | None]:
        """Get fill (execution) history."""
        kwargs: dict[str, Any] = {"limit": limit}
        if ticker:
            kwargs["ticker"] = ticker
        if cursor:
            kwargs["cursor"] = cursor
        result = self._call_with_retry(
            "get_fills",
            self.client.get_fills,
            **kwargs,
        )
        data = self._to_dict(result)
        fills = []
        for f in data.get("fills", []):
            fills.append(
                FillInfo(
                    trade_id=f.get("trade_id", ""),
                    ticker=f.get("ticker", ""),
                    side=f.get("side", ""),
                    action=f.get("action", ""),
                    count=f.get("count", 0),
                    yes_price=f.get("yes_price", 0),
                    no_price=f.get("no_price", 0),
                    is_taker=f.get("is_taker", True),
                    created_time=f.get("created_time", ""),
                    order_id=f.get("order_id", ""),
                )
            )
        return fills, data.get("cursor")

    def get_balance(self) -> BalanceInfo:
        """Get account balance and portfolio value."""
        result = self._call_with_retry(
            "get_balance",
            self.client.get_balance,
        )
        data = self._to_dict(result)
        return BalanceInfo(
            balance=data.get("balance", 0),
            portfolio_value=data.get("portfolio_value", 0),
        )

    # ── Event/Market Discovery ───────────────────────────────────────────

    def discover_active_events(self, series_ticker: str) -> list[dict[str, Any]]:
        """Find all active (open) events for a series.

        Paginates through all results to return the complete list.
        """
        all_events: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            data = self.get_events(series_ticker=series_ticker, status="open", cursor=cursor)
            events = data.get("events", [])
            all_events.extend(events)
            cursor = data.get("cursor")
            if not cursor or not events:
                break
        logger.info("discovered_active_events", series=series_ticker, count=len(all_events))
        return all_events

    def discover_markets_for_event(self, event_ticker: str) -> list[MarketInfo]:
        """Get all markets (contracts) for an event."""
        all_markets: list[MarketInfo] = []
        cursor: str | None = None
        while True:
            markets, cursor = self.get_markets(event_ticker=event_ticker, cursor=cursor)
            all_markets.extend(markets)
            if not cursor or not markets:
                break
        logger.info("discovered_markets", event_ticker=event_ticker, count=len(all_markets))
        return all_markets

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        """Convert SDK response object to a dict."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()  # type: ignore[no-any-return]
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return {"raw": obj}

    @staticmethod
    def _parse_market(data: dict[str, Any]) -> MarketInfo:
        """Parse a market dict into a MarketInfo."""
        return MarketInfo(
            ticker=data.get("ticker", ""),
            event_ticker=data.get("event_ticker", ""),
            series_ticker=data.get("series_ticker", ""),
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            status=data.get("status", ""),
            yes_bid=data.get("yes_bid", 0),
            yes_ask=data.get("yes_ask", 0),
            no_bid=data.get("no_bid", 0),
            no_ask=data.get("no_ask", 0),
            last_price=data.get("last_price", 0),
            volume=data.get("volume", 0),
            volume_24h=data.get("volume_24h", 0),
            open_time=data.get("open_time", ""),
            close_time=data.get("close_time", ""),
            expiration_time=data.get("expiration_time", ""),
            result=data.get("result"),
        )
