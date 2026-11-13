import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# backend_api.app.database builds its engine at import time, so the URL has to
# be in place before any backend module is imported.
os.environ.setdefault("DATABASE_URL", "sqlite://")
# Overwrite, never setdefault: a real key in the developer's shell would
# otherwise let a test reach the live Gemini API.
os.environ["GEMINI_API_KEY"] = "test-key-not-used"

from sqlalchemy.dialects.postgresql import JSONB, UUID  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402


@compiles(JSONB, "sqlite")
def _compile_jsonb_on_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid_on_sqlite(type_, compiler, **kw):
    return "CHAR(36)"


import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import close_all_sessions, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

# Set to a disposable Postgres database to run the database tests against Postgres (CI does both).
# Its public schema is dropped and recreated before every test that asks for a database.
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def make_blank_engine():
    """An empty database: Postgres when TEST_DATABASE_URL is set, else in-memory SQLite."""
    if TEST_DATABASE_URL:
        eng = create_engine(TEST_DATABASE_URL)
        with eng.begin() as conn:
            # A session a previous test left open would hold its locks and block the reset forever.
            conn.execute(text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            ))
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        return eng
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


@pytest.fixture()
def blank_engine():
    eng = make_blank_engine()
    yield eng
    close_all_sessions()
    eng.dispose()


@pytest.fixture()
def engine(blank_engine):
    """A fresh database with the model schema, shared by one test's sessions."""
    from backend_api.app.database import Base

    Base.metadata.create_all(blank_engine)
    return blank_engine


@pytest.fixture()
def db(engine):
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _no_overload_wait(monkeypatch):
    """generate() waits before a second round when every model is overloaded; tests need not."""
    from backend_api.app import llm

    monkeypatch.setattr(llm, "OVERLOAD_RETRY_SECONDS", 0)
