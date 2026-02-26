"""Database engine and session management."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine  # noqa: TCH002
from sqlalchemy.orm import Session, sessionmaker

from autotrader.state.models import Base


def create_db_engine(url: str = "sqlite:///autotrader.db", echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine."""
    return create_engine(url, echo=echo)


def init_db(engine: Engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Get a session factory bound to the given engine."""
    return sessionmaker(bind=engine)
