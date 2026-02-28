"""Unit tests for the TradingRepository."""

from __future__ import annotations

import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autotrader.execution.engine import OrderStatus, TrackedOrder
from autotrader.state.models import (
    Base,
    DailyPnl,
    Fill,
    LeaderboardSnapshot,
    Order,
    RiskEvent,
    Signal,
    SystemEvent,
)
from autotrader.state.repository import TradingRepository


def _repo() -> TradingRepository:
    """Create a repository backed by an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine)
    return TradingRepository(sf)


def _tracked_order(**overrides: object) -> TrackedOrder:
    now = datetime.datetime(2026, 2, 27, 12, 0, 0)
    defaults: dict[str, object] = {
        "client_order_id": "leaderboard_alpha-KXTOPMODEL-GPT5-abc123",
        "kalshi_order_id": None,
        "ticker": "KXTOPMODEL-GPT5",
        "side": "yes",
        "price_cents": 50,
        "quantity": 5,
        "filled_quantity": 5,
        "status": OrderStatus.FILLED,
        "strategy": "leaderboard_alpha",
        "urgency": "aggressive",
        "rationale": "rank jump detected",
        "is_paper": True,
        "created_at": now,
        "updated_at": now,
        "filled_at": now,
        "cancelled_at": None,
    }
    defaults.update(overrides)
    return TrackedOrder(**defaults)  # type: ignore[arg-type]


def _fill_data(**overrides: object) -> dict:
    defaults: dict[str, object] = {
        "ticker": "KXTOPMODEL-GPT5",
        "side": "yes",
        "action": "buy",
        "count": 5,
        "price_cents": 50,
        "fee_cents": 4,
        "is_taker": True,
        "is_paper": True,
        "client_order_id": "leaderboard_alpha-KXTOPMODEL-GPT5-abc123",
        "kalshi_fill_id": None,
        "filled_at": "2026-02-27T12:00:00",
    }
    defaults.update(overrides)
    return defaults  # type: ignore[return-value]


# ── Order persistence ────────────────────────────────────────────────────


class TestRecordOrder:
    def test_persist_order(self) -> None:
        repo = _repo()
        order = _tracked_order()
        repo.record_order(order)

        with repo._session_factory() as session:
            rows = session.query(Order).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.client_order_id == "leaderboard_alpha-KXTOPMODEL-GPT5-abc123"
            assert row.ticker == "KXTOPMODEL-GPT5"
            assert row.event_ticker == "KXTOPMODEL"
            assert row.side == "yes"
            assert row.price_cents == 50
            assert row.quantity == 5
            assert row.filled_quantity == 5
            assert row.remaining_quantity == 0
            assert row.status == "filled"
            assert row.strategy == "leaderboard_alpha"
            assert row.is_paper is True

    def test_persist_order_no_hyphen_ticker(self) -> None:
        repo = _repo()
        order = _tracked_order(ticker="SIMPLETCH")
        repo.record_order(order)

        with repo._session_factory() as session:
            row = session.query(Order).first()
            assert row is not None
            assert row.event_ticker == "SIMPLETCH"

    def test_duplicate_order_does_not_crash(self) -> None:
        repo = _repo()
        order = _tracked_order()
        repo.record_order(order)
        # Second insert with same client_order_id — should catch and log
        repo.record_order(order)


# ── Fill persistence ─────────────────────────────────────────────────────


class TestRecordFill:
    def test_persist_fill(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(), strategy="leaderboard_alpha")

        with repo._session_factory() as session:
            rows = session.query(Fill).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.ticker == "KXTOPMODEL-GPT5"
            assert row.price_cents == 50
            assert row.quantity == 5
            assert row.fee_cents == 4
            assert row.strategy == "leaderboard_alpha"

    def test_fill_creates_daily_pnl(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(), strategy="leaderboard_alpha")

        with repo._session_factory() as session:
            pnl = session.query(DailyPnl).first()
            assert pnl is not None
            assert pnl.strategy == "leaderboard_alpha"
            assert pnl.trade_count == 1
            assert pnl.total_fees_cents == 4

    def test_multiple_fills_accumulate_pnl(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(fee_cents=4), strategy="leaderboard_alpha")
        repo.record_fill(_fill_data(fee_cents=6, kalshi_fill_id="fill-2"), strategy="leaderboard_alpha")

        with repo._session_factory() as session:
            pnl = session.query(DailyPnl).first()
            assert pnl is not None
            assert pnl.trade_count == 2
            assert pnl.total_fees_cents == 10

    def test_realized_pnl_updates_after_winning_round_trip(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(side="yes", price_cents=40, count=10, fee_cents=0), strategy="leaderboard_alpha")
        repo.record_fill(
            _fill_data(side="no", price_cents=30, count=10, fee_cents=0, kalshi_fill_id="fill-win-close"),
            strategy="leaderboard_alpha",
        )

        pnl = repo.get_daily_pnl("leaderboard_alpha")
        assert pnl is not None
        assert pnl.realized_pnl_cents == 300
        assert pnl.unrealized_pnl_cents == 0

    def test_realized_pnl_updates_after_losing_round_trip(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(side="yes", price_cents=70, count=10, fee_cents=0), strategy="leaderboard_alpha")
        repo.record_fill(
            _fill_data(side="no", price_cents=50, count=10, fee_cents=0, kalshi_fill_id="fill-loss-close"),
            strategy="leaderboard_alpha",
        )

        pnl = repo.get_daily_pnl("leaderboard_alpha")
        assert pnl is not None
        assert pnl.realized_pnl_cents == -200
        assert pnl.unrealized_pnl_cents == 0


# ── Signal persistence ───────────────────────────────────────────────────


class TestRecordSignal:
    def test_persist_signal(self) -> None:
        repo = _repo()
        repo.record_signal(
            signal_source="arena_monitor",
            event_type="ranking_change",
            data={"model_name": "GPT-5", "old_rank_ub": 3, "new_rank_ub": 1},
            relevant_series=["KXTOPMODEL"],
            urgency="high",
        )

        with repo._session_factory() as session:
            rows = session.query(Signal).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.source == "arena_monitor"
            assert row.event_type == "ranking_change"
            assert row.urgency == "high"
            assert row.data["model_name"] == "GPT-5"

    def test_signal_with_action(self) -> None:
        repo = _repo()
        repo.record_signal(
            signal_source="arena_monitor",
            event_type="new_leader",
            data={},
            relevant_series=[],
            urgency="high",
            action="proposed BUY KXTOPMODEL-GPT5",
        )

        with repo._session_factory() as session:
            row = session.query(Signal).first()
            assert row is not None
            assert row.action_taken == "proposed BUY KXTOPMODEL-GPT5"


# ── Risk event persistence ──────────────────────────────────────────────


class TestRecordRiskEvent:
    def test_persist_risk_event(self) -> None:
        repo = _repo()
        order_data = {"strategy": "leaderboard_alpha", "ticker": "T1", "side": "yes", "quantity": 10}
        repo.record_risk_event("position_per_contract", order_data, "Would exceed per-contract limit")

        with repo._session_factory() as session:
            rows = session.query(RiskEvent).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.check_name == "position_per_contract"
            assert row.reason == "Would exceed per-contract limit"
            assert row.proposed_order["ticker"] == "T1"


# ── Leaderboard snapshot persistence ─────────────────────────────────────


class TestRecordLeaderboardSnapshot:
    def test_persist_snapshot(self) -> None:
        repo = _repo()
        repo.record_leaderboard_snapshot(
            snapshot_data={"models": [{"name": "GPT-5", "rank": 1}]},
            source_url="https://example.com/csv",
            model_count=5,
            top_model="GPT-5",
            top_org="OpenAI",
        )

        with repo._session_factory() as session:
            rows = session.query(LeaderboardSnapshot).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.model_count == 5
            assert row.top_model == "GPT-5"
            assert row.top_org == "OpenAI"


# ── System event persistence ─────────────────────────────────────────────


class TestRecordSystemEvent:
    def test_persist_system_event(self) -> None:
        repo = _repo()
        repo.record_system_event("startup", {"mode": "paper"}, severity="info")

        with repo._session_factory() as session:
            rows = session.query(SystemEvent).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.event_type == "startup"
            assert row.severity == "info"
            assert row.details["mode"] == "paper"

    def test_system_event_default_severity(self) -> None:
        repo = _repo()
        repo.record_system_event("heartbeat")

        with repo._session_factory() as session:
            row = session.query(SystemEvent).first()
            assert row is not None
            assert row.severity == "info"


# ── Daily P&L queries ───────────────────────────────────────────────────


class TestDailyPnlQueries:
    def test_get_daily_pnl_none_when_empty(self) -> None:
        repo = _repo()
        assert repo.get_daily_pnl("leaderboard_alpha") is None

    def test_get_daily_pnl_after_fill(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(), strategy="leaderboard_alpha")
        pnl = repo.get_daily_pnl("leaderboard_alpha")
        assert pnl is not None
        assert pnl.trade_count == 1

    def test_get_today_trade_count(self) -> None:
        repo = _repo()
        assert repo.get_today_trade_count("leaderboard_alpha") == 0
        repo.record_fill(_fill_data(), strategy="leaderboard_alpha")
        assert repo.get_today_trade_count("leaderboard_alpha") == 1

    def test_get_daily_pnl_wrong_date(self) -> None:
        repo = _repo()
        repo.record_fill(_fill_data(), strategy="leaderboard_alpha")
        # Query for yesterday — should not find today's data
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        assert repo.get_daily_pnl("leaderboard_alpha", date=yesterday) is None


# ── Error isolation ──────────────────────────────────────────────────────


class TestErrorIsolation:
    def test_bad_fill_data_does_not_crash(self) -> None:
        repo = _repo()
        # Missing required keys — should catch exception
        repo.record_fill({"bad": "data"}, strategy="test")

    def test_bad_signal_data_does_not_crash(self) -> None:
        repo = _repo()
        # This should work fine even with empty data
        repo.record_signal("src", "evt", {}, [], "low")
