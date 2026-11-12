import pytest

from backend_api.app.database import upsert_user


@pytest.fixture()
def session(db):
    """The shared database fixture: SQLite, or Postgres when TEST_DATABASE_URL is set."""
    return db


def test_display_name_survives_a_later_call_without_one(session):
    """Every authenticated request upserts with None and used to wipe the name."""
    upsert_user(session, "uid-1", "person@example.com", "Asha")
    refreshed = upsert_user(session, "uid-1", "person@example.com", None)
    assert refreshed.display_name == "Asha"


def test_email_survives_a_later_call_without_one(session):
    upsert_user(session, "uid-1", "person@example.com", "Asha")
    refreshed = upsert_user(session, "uid-1", None, None)
    assert refreshed.email == "person@example.com"


def test_a_supplied_display_name_still_updates(session):
    upsert_user(session, "uid-1", "person@example.com", "Asha")
    refreshed = upsert_user(session, "uid-1", "person@example.com", "Asha Verma")
    assert refreshed.display_name == "Asha Verma"
