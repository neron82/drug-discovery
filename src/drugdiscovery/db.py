"""Database engine / session (SQLite today, dialect-agnostic for Postgres+pgvector later)."""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _connect(db_path: str):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _connect(settings.storage.db_path)
        if _engine.dialect.name == "sqlite":

            @event.listens_for(_engine, "connect")
            def _fk(dbapi_conn, rec):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                # WAL alone doesn't serialize writers: without busy_timeout any
                # concurrent write (e.g. drain counter vs API insert) raises
                # "database is locked" instantly. Wait 5s instead.
                cur.execute("PRAGMA busy_timeout=5000")
                cur.close()

    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_db(create_all: bool = True):
    """Create schema. For now uses metadata.create_all (idempotent).
    Production path swaps to Alembic migrations before changing tables."""
    from . import models  # noqa: F401  (ensure models registered)
    if create_all:
        Base.metadata.create_all(get_engine())
    # ensuredirs
    os.makedirs(settings.storage.artifact_dir, exist_ok=True)
    os.makedirs(settings.storage.log_dir, exist_ok=True)


@contextmanager
def session_scope():
    Session = get_session_factory()
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session():
    return get_session_factory()()
