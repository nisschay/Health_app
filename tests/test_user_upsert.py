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


def _self_profile_name(session, uid):
    from backend_api.app.database import Profile, User

    owner = session.query(User).filter(User.firebase_uid == uid).one()
    return session.query(Profile).filter(Profile.account_owner_id == owner.id, Profile.relationship == "self").one().full_name


def test_a_blank_display_name_does_not_erase_a_saved_one(session):
    upsert_user(session, "uid-1", "person@example.com", "Asha")
    assert upsert_user(session, "uid-1", "person@example.com", "  ").display_name == "Asha"


def test_the_sign_up_race_leaves_the_typed_name_on_the_profile(session):
    """Firebase reports a new user before the name is saved, so the first sync has no name."""
    upsert_user(session, "uid-1", "e2e-test-1@example.com", "")
    assert _self_profile_name(session, "uid-1") == "E2E-Test-1"
    upsert_user(session, "uid-1", "e2e-test-1@example.com", "Asha Verma")
    assert _self_profile_name(session, "uid-1") == "Asha Verma"


def test_a_renamed_profile_is_never_overwritten(session):
    from backend_api.app.database import Profile

    upsert_user(session, "uid-1", "asha@example.com", None)
    session.query(Profile).update({"full_name": "Asha (me)"})
    session.commit()
    upsert_user(session, "uid-1", "asha@example.com", "Asha Verma")
    assert _self_profile_name(session, "uid-1") == "Asha (me)"
