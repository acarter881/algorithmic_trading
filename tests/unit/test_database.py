"""Unit tests for database schema and operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from autotrader.state.database import init_db
from autotrader.state.models import (
    Fill,
    LeaderboardSnapshot,
    Order,
    Position,
    RiskEvent,
    Signal,
    SystemEvent,
)


@pytest.fixture
def db_session() -> Session:
    """Create an in-memory SQLite database and return a session."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


class TestOrderModel:
    def test_create_order(self, db_session: Session) -> None:
        order = Order(
            client_order_id="test-001",
            ticker="KXTOPMODEL-26FEB28-CLAUDE",
            side="yes",
            price_cents=65,
            quantity=10,
            remaining_quantity=10,
            status="pending",
            strategy="leaderboard_alpha",
            urgency="aggressive",
            rationale="Fair value 70, market 65",
        )
        db_session.add(order)
        db_session.commit()

        fetched = db_session.query(Order).filter_by(client_order_id="test-001").one()
        assert fetched.ticker == "KXTOPMODEL-26FEB28-CLAUDE"
        assert fetched.price_cents == 65
        assert fetched.quantity == 10
        assert fetched.status == "pending"

    def test_unique_client_order_id(self, db_session: Session) -> None:
        order1 = Order(
            client_order_id="dup-001",
            ticker="T1",
            side="yes",
            price_cents=50,
            quantity=1,
            remaining_quantity=1,
            status="pending",
            strategy="test",
            urgency="passive",
        )
        order2 = Order(
            client_order_id="dup-001",
            ticker="T2",
            side="no",
            price_cents=50,
            quantity=1,
            remaining_quantity=1,
            status="pending",
            strategy="test",
            urgency="passive",
        )
        db_session.add(order1)
        db_session.commit()
        db_session.add(order2)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestPositionModel:
    def test_create_position(self, db_session: Session) -> None:
        pos = Position(
            ticker="KXTOPMODEL-26FEB28-CLAUDE",
            side="yes",
            quantity=10,
            avg_cost_cents=65.0,
            total_cost_cents=650.0,
            strategy="leaderboard_alpha",
        )
        db_session.add(pos)
        db_session.commit()

        fetched = db_session.query(Position).filter_by(ticker="KXTOPMODEL-26FEB28-CLAUDE").one()
        assert fetched.quantity == 10
        assert fetched.avg_cost_cents == 65.0


class TestFillModel:
    def test_create_fill(self, db_session: Session) -> None:
        fill = Fill(
            order_client_id="test-001",
            ticker="KXTOPMODEL-26FEB28-CLAUDE",
            side="yes",
            price_cents=65,
            quantity=5,
            fee_cents=2,
            is_taker=True,
            strategy="leaderboard_alpha",
        )
        db_session.add(fill)
        db_session.commit()

        fetched = db_session.query(Fill).filter_by(order_client_id="test-001").one()
        assert fetched.fee_cents == 2
        assert fetched.is_taker is True


class TestSignalModel:
    def test_create_signal(self, db_session: Session) -> None:
        signal = Signal(
            source="arena_monitor",
            event_type="new_number_one",
            urgency="high",
            data={"model": "claude-opus-4-6", "new_rank_ub": 1},
            relevant_series=["KXTOPMODEL", "KXLLM1"],
        )
        db_session.add(signal)
        db_session.commit()

        fetched = db_session.query(Signal).filter_by(event_type="new_number_one").one()
        assert fetched.data["model"] == "claude-opus-4-6"
        assert fetched.urgency == "high"


class TestLeaderboardSnapshot:
    def test_create_snapshot(self, db_session: Session) -> None:
        snapshot = LeaderboardSnapshot(
            snapshot_data={
                "models": [
                    {"name": "claude-opus-4-6", "rank_ub": 1, "score": 1502},
                ]
            },
            source_url="https://arena.ai/leaderboard/text/overall-no-style-control",
            model_count=50,
            top_model="claude-opus-4-6",
            top_org="Anthropic",
        )
        db_session.add(snapshot)
        db_session.commit()

        fetched = db_session.query(LeaderboardSnapshot).first()
        assert fetched is not None
        assert fetched.top_model == "claude-opus-4-6"
        assert fetched.model_count == 50


class TestRiskEvent:
    def test_create_risk_event(self, db_session: Session) -> None:
        event = RiskEvent(
            check_name="max_position_per_contract",
            proposed_order={"ticker": "T1", "quantity": 200},
            reason="Would exceed $100 limit",
        )
        db_session.add(event)
        db_session.commit()

        fetched = db_session.query(RiskEvent).first()
        assert fetched is not None
        assert fetched.check_name == "max_position_per_contract"


class TestSystemEvent:
    def test_create_system_event(self, db_session: Session) -> None:
        event = SystemEvent(
            event_type="startup",
            details={"version": "0.1.0", "environment": "demo"},
            severity="info",
        )
        db_session.add(event)
        db_session.commit()

        fetched = db_session.query(SystemEvent).filter_by(event_type="startup").one()
        assert fetched.details["version"] == "0.1.0"
