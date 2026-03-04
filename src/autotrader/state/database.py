"""Database engine and session management."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine  # noqa: TCH002
from sqlalchemy.orm import Session, sessionmaker

from autotrader.state.models import Base

logger = logging.getLogger(__name__)


def create_db_engine(url: str = "sqlite:///autotrader.db", echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine."""
    return create_engine(url, echo=echo)


def _migrate_missing_columns(engine: Engine) -> None:
    """Add any columns present in ORM models but missing from the database.

    SQLAlchemy's ``create_all`` creates missing tables but never alters
    existing ones.  This lightweight migration inspects each table and
    adds columns that the ORM defines but that are absent from the
    actual schema.  Only supports simple ``ALTER TABLE ADD COLUMN``
    (sufficient for SQLite).
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue

        existing_cols = {col["name"] for col in inspector.get_columns(table_name)}
        for column in table.columns:
            if column.name in existing_cols:
                continue

            # Build a portable default clause for the ADD COLUMN
            col_type = column.type.compile(engine.dialect)
            nullable = "NULL" if column.nullable else "NOT NULL"
            default = ""
            sd = column.server_default
            if sd is not None and hasattr(sd, "arg"):
                default = f" DEFAULT {sd.arg}"
            elif not column.nullable:
                # SQLite requires a default for NOT NULL on existing rows
                default = " DEFAULT ''"

            ddl = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type} {nullable}{default}"
            logger.info("migrating missing column: %s.%s", table_name, column.name)
            with engine.begin() as conn:
                conn.execute(text(ddl))


def init_db(engine: Engine) -> None:
    """Create all tables if they don't exist, then add any missing columns."""
    Base.metadata.create_all(engine)
    _migrate_missing_columns(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Get a session factory bound to the given engine."""
    return sessionmaker(bind=engine)
