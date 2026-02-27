"""Unit tests for the execution engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autotrader.api.client import KalshiAPIClient, KalshiAPIError
from autotrader.execution.engine import ExecutionEngine, ExecutionMode, FillEvent, OrderStatus
from autotrader.strategies.base import OrderUrgency, ProposedOrder

# ── Helpers ──────────────────────────────────────────────────────────────


def _order(
    strategy: str = "leaderboard_alpha",
    ticker: str = "KXTOPMODEL-GPT5",
    side: str = "yes",
    price_cents: int = 50,
    quantity: int = 5,
    urgency: OrderUrgency = OrderUrgency.PASSIVE,
) -> ProposedOrder:
    return ProposedOrder(
        strategy=strategy,
        ticker=ticker,
        side=side,
        price_cents=price_cents,
        quantity=quantity,
        urgency=urgency,
        rationale="test order",
    )


def _paper_engine() -> ExecutionEngine:
    return ExecutionEngine(mode=ExecutionMode.PAPER)


def _mock_api() -> MagicMock:
    api = MagicMock(spec=KalshiAPIClient)
    api.place_order.return_value = {"order": {"order_id": "kalshi-123"}}
    return api


def _live_engine(api: MagicMock | None = None) -> ExecutionEngine:
    return ExecutionEngine(mode=ExecutionMode.LIVE, api_client=api or _mock_api())


# ── Construction ─────────────────────────────────────────────────────────


class TestConstruction:
    def test_paper_mode_no_api_required(self) -> None:
        engine = _paper_engine()
        assert engine.mode == ExecutionMode.PAPER

    def test_live_mode_requires_api(self) -> None:
        with pytest.raises(ValueError, match="api_client"):
            ExecutionEngine(mode=ExecutionMode.LIVE)

    def test_live_mode_with_api(self) -> None:
        engine = _live_engine()
        assert engine.mode == ExecutionMode.LIVE


# ── Paper Execution ──────────────────────────────────────────────────────


class TestPaperExecution:
    async def test_paper_order_fills_instantly(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order())
        assert result.success
        assert result.order.status == OrderStatus.FILLED
        assert result.order.filled_quantity == 5
        assert result.order.is_paper is True

    async def test_paper_order_has_client_id(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order())
        assert result.order.client_order_id.startswith("leaderboard_alpha-KXTOPMODEL-GPT5-")

    async def test_paper_order_tracked(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order())
        assert result.order.client_order_id in engine.orders

    async def test_paper_fills_at_proposed_price(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order(price_cents=42))
        assert result.order.price_cents == 42

    async def test_paper_order_has_filled_at(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order())
        assert result.order.filled_at is not None

    async def test_paper_yes_side(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order(side="yes"))
        assert result.order.side == "yes"

    async def test_paper_no_side(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order(side="no", price_cents=40))
        assert result.order.side == "no"
        assert result.success

    async def test_paper_tracks_strategy(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order(strategy="my_strat"))
        assert result.order.strategy == "my_strat"

    async def test_paper_tracks_urgency(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order(urgency=OrderUrgency.AGGRESSIVE))
        assert result.order.urgency == "aggressive"


# ── Fill Callbacks ───────────────────────────────────────────────────────


class TestFillCallbacks:
    async def test_callback_invoked_on_paper_fill(self) -> None:
        engine = _paper_engine()
        fills: list[dict] = []
        engine.on_fill(fills.append)

        await engine.submit(_order())
        assert len(fills) == 1
        assert fills[0]["ticker"] == "KXTOPMODEL-GPT5"
        assert fills[0]["side"] == "yes"
        assert fills[0]["count"] == 5
        assert fills[0]["is_paper"] is True

    async def test_multiple_callbacks(self) -> None:
        engine = _paper_engine()
        fills_a: list[dict] = []
        fills_b: list[dict] = []
        engine.on_fill(fills_a.append)
        engine.on_fill(fills_b.append)

        await engine.submit(_order())
        assert len(fills_a) == 1
        assert len(fills_b) == 1

    async def test_callback_receives_fee_info(self) -> None:
        engine = _paper_engine()
        fills: list[dict] = []
        engine.on_fill(fills.append)

        await engine.submit(_order(price_cents=50, quantity=10))
        assert "fee_cents" in fills[0]
        assert fills[0]["fee_cents"] > 0

    async def test_callback_error_isolated(self) -> None:
        engine = _paper_engine()

        def bad_callback(fill: dict) -> None:
            raise RuntimeError("callback error")

        good_fills: list[dict] = []
        engine.on_fill(bad_callback)
        engine.on_fill(good_fills.append)

        result = await engine.submit(_order())
        assert result.success
        assert len(good_fills) == 1  # Second callback still runs

    async def test_callback_receives_client_order_id(self) -> None:
        engine = _paper_engine()
        fills: list[dict] = []
        engine.on_fill(fills.append)

        result = await engine.submit(_order())
        assert fills[0]["client_order_id"] == result.order.client_order_id


# ── Live Execution ───────────────────────────────────────────────────────


class TestLiveExecution:
    async def test_live_order_submitted(self) -> None:
        api = _mock_api()
        engine = _live_engine(api)
        result = await engine.submit(_order())

        assert result.success
        assert result.order.status == OrderStatus.SUBMITTED
        assert result.order.kalshi_order_id == "kalshi-123"
        assert result.order.is_paper is False
        api.place_order.assert_called_once()

    async def test_live_order_request_yes(self) -> None:
        api = _mock_api()
        engine = _live_engine(api)
        await engine.submit(_order(side="yes", price_cents=55, quantity=3))

        call_args = api.place_order.call_args[0][0]
        assert call_args.side == "yes"
        assert call_args.yes_price == 55
        assert call_args.no_price is None
        assert call_args.count == 3

    async def test_live_order_request_no(self) -> None:
        api = _mock_api()
        engine = _live_engine(api)
        await engine.submit(_order(side="no", price_cents=40, quantity=2))

        call_args = api.place_order.call_args[0][0]
        assert call_args.side == "no"
        assert call_args.no_price == 40
        assert call_args.yes_price is None
        assert call_args.count == 2

    async def test_live_api_error_marks_rejected(self) -> None:
        api = _mock_api()
        api.place_order.side_effect = KalshiAPIError("Insufficient funds", status_code=400)
        engine = _live_engine(api)

        result = await engine.submit(_order())
        assert not result.success
        assert result.order.status == OrderStatus.REJECTED
        assert "Insufficient funds" in result.error

    async def test_live_order_tracked(self) -> None:
        engine = _live_engine()
        result = await engine.submit(_order())
        assert result.order.client_order_id in engine.orders

    async def test_live_no_fill_callback_on_submit(self) -> None:
        engine = _live_engine()
        fills: list[dict] = []
        engine.on_fill(fills.append)

        await engine.submit(_order())
        # Live orders don't fill immediately — fills come via WebSocket/polling
        assert len(fills) == 0


# ── Cancellation ─────────────────────────────────────────────────────────


class TestCancellation:
    async def test_cancel_paper_resting_order(self) -> None:
        # Paper orders fill instantly, so we need a different approach.
        # Let's test that cancel works on submitted live orders.
        api = _mock_api()
        engine = _live_engine(api)
        result = await engine.submit(_order())
        # Live orders start as SUBMITTED
        assert result.order.status == OrderStatus.SUBMITTED

        success = await engine.cancel(result.order.client_order_id)
        assert success
        assert engine.orders[result.order.client_order_id].status == OrderStatus.CANCELLED

    async def test_cancel_unknown_order(self) -> None:
        engine = _paper_engine()
        success = await engine.cancel("nonexistent-id")
        assert not success

    async def test_cancel_already_filled(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order())
        # Paper orders fill instantly
        assert result.order.status == OrderStatus.FILLED
        success = await engine.cancel(result.order.client_order_id)
        assert not success

    async def test_cancel_live_api_failure(self) -> None:
        api = _mock_api()
        api.cancel_order.side_effect = KalshiAPIError("Not found", status_code=404)
        engine = _live_engine(api)
        result = await engine.submit(_order())

        success = await engine.cancel(result.order.client_order_id)
        assert not success
        # Status should remain SUBMITTED (cancel failed)
        assert engine.orders[result.order.client_order_id].status == OrderStatus.SUBMITTED

    async def test_cancel_sets_timestamp(self) -> None:
        api = _mock_api()
        engine = _live_engine(api)
        result = await engine.submit(_order())
        await engine.cancel(result.order.client_order_id)
        assert engine.orders[result.order.client_order_id].cancelled_at is not None


# ── Batch Submit ─────────────────────────────────────────────────────────


class TestBatchSubmit:
    async def test_batch_submit_all_succeed(self) -> None:
        engine = _paper_engine()
        orders = [_order(ticker=f"T{i}") for i in range(3)]
        results = await engine.submit_batch(orders)
        assert len(results) == 3
        assert all(r.success for r in results)

    async def test_batch_submit_preserves_order(self) -> None:
        engine = _paper_engine()
        orders = [_order(ticker="T1"), _order(ticker="T2"), _order(ticker="T3")]
        results = await engine.submit_batch(orders)
        assert results[0].order.ticker == "T1"
        assert results[1].order.ticker == "T2"
        assert results[2].order.ticker == "T3"


# ── Order Queries ────────────────────────────────────────────────────────


class TestOrderQueries:
    async def test_get_open_orders_paper(self) -> None:
        engine = _paper_engine()
        await engine.submit(_order())
        # Paper fills instantly → no open orders
        assert len(engine.get_open_orders()) == 0

    async def test_get_open_orders_live(self) -> None:
        engine = _live_engine()
        await engine.submit(_order())
        # Live orders start as SUBMITTED → open
        open_orders = engine.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].status == OrderStatus.SUBMITTED

    async def test_get_orders_for_strategy(self) -> None:
        engine = _paper_engine()
        await engine.submit(_order(strategy="strat_a"))
        await engine.submit(_order(strategy="strat_b"))
        await engine.submit(_order(strategy="strat_a"))

        strat_a_orders = engine.get_orders_for_strategy("strat_a")
        assert len(strat_a_orders) == 2
        strat_b_orders = engine.get_orders_for_strategy("strat_b")
        assert len(strat_b_orders) == 1


# ── Client Order ID ─────────────────────────────────────────────────────


class TestClientOrderId:
    async def test_ids_are_unique(self) -> None:
        engine = _paper_engine()
        r1 = await engine.submit(_order())
        r2 = await engine.submit(_order())
        assert r1.order.client_order_id != r2.order.client_order_id

    async def test_id_contains_strategy_and_ticker(self) -> None:
        engine = _paper_engine()
        result = await engine.submit(_order(strategy="alpha", ticker="KXTOPMODEL-X"))
        assert "alpha" in result.order.client_order_id
        assert "KXTOPMODEL-X" in result.order.client_order_id


# ── FillEvent ────────────────────────────────────────────────────────────


class TestFillEvent:
    def test_fill_event_defaults(self) -> None:
        fill = FillEvent(
            ticker="T1",
            side="yes",
            action="buy",
            count=5,
            price_cents=50,
            fee_cents=2,
            is_taker=True,
            is_paper=True,
            client_order_id="test-123",
        )
        assert fill.kalshi_fill_id is None
        assert fill.filled_at is not None
