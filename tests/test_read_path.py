from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend_api.app import database, main
from backend_api.app.auth import RequestUser
from backend_api.app.database import Base, Profile, Report, ReportAnalysis, Study, User


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def db(engine):
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed(db, studies=2, reports_per_study=2):
    user = User(firebase_uid="uid-1", email="a@example.com", display_name="A", last_login=datetime.utcnow())
    db.add(user)
    db.flush()
    profile = Profile(account_owner_id=user.id, full_name="A", relationship="self")
    db.add(profile)
    db.flush()
    for s in range(studies):
        study = Study(profile_id=profile.id, name=f"study-{s}")
        db.add(study)
        db.flush()
        for r in range(reports_per_study):
            db.add(
                Report(
                    study_id=study.id,
                    file_name=f"r{s}{r}.pdf",
                    file_url="uploaded://x",
                    report_date=date(2024, 1 + s, 1 + r),
                    lab_name="Lab A" if s == 0 else f"Lab {r}",
                    analysis_data={"health_summary": {"concerns": [{}] * (r + 1)}},
                )
            )
    db.commit()
    return user


def _count_statements(engine):
    counter = {"n": 0}

    def _tick(*args):
        counter["n"] += 1

    event.listen(engine, "before_cursor_execute", _tick)
    return counter, lambda: event.remove(engine, "before_cursor_execute", _tick)


def test_dashboard_query_count_does_not_grow_with_studies(engine, db):
    """Each study used to cost its own report query; the dashboard should be O(1) statements."""
    _seed(db, studies=2)
    request_user = RequestUser(user_id="uid-1", email="a@example.com")

    counter, stop = _count_statements(engine)
    main.studies_dashboard_summary(user=request_user, db=db)
    small = counter["n"]
    stop()

    for s in range(2, 7):
        study = Study(profile_id=db.query(Profile).first().id, name=f"study-{s}")
        db.add(study)
        db.flush()
        db.add(Report(study_id=study.id, file_name="x.pdf", file_url="u", report_date=date(2024, 6, 1),
                      analysis_data={"health_summary": {"concerns": []}}))
    db.commit()

    counter, stop = _count_statements(engine)
    summary = main.studies_dashboard_summary(user=request_user, db=db)
    large = counter["n"]
    stop()

    assert large == small, f"{small} statements for 2 studies, {large} for 7"
    assert summary.total_reports == 2 * 2 + 5
    assert summary.total_alerts == (1 + 2) + (1 + 2) + 5 * 0


def test_dashboard_values(db):
    _seed(db, studies=2, reports_per_study=2)
    summary = main.studies_dashboard_summary(user=RequestUser(user_id="uid-1", email="a@example.com"), db=db)
    first, second = sorted(summary.profiles[0].studies, key=lambda s: s.name)
    assert (first.report_count, first.range_start, first.range_end) == (2, "2024-01-01", "2024-01-02")
    assert first.consistent_lab_name == "Lab A"
    assert second.consistent_lab_name is None
    assert first.alerts_count == 3 and first.has_alerts


def test_study_report_stats_uses_min_and_max(db):
    _seed(db, studies=1, reports_per_study=3)
    study = db.query(Study).first()
    count, first, last = database.study_report_stats(db, [study.id])[study.id]
    assert (count, first, last) == (3, date(2024, 1, 1), date(2024, 1, 3))
    assert database.study_report_stats(db, []) == {}


def test_upsert_skips_the_write_when_recently_seen(engine, db):
    user = _seed(db)
    recent = datetime.utcnow() - timedelta(minutes=1)
    user.last_login = recent
    db.commit()

    counter, stop = _count_statements(engine)
    database.upsert_user(db, "uid-1", "a@example.com", None)
    stop()
    db.refresh(user)
    assert user.last_login == recent, "a request within the interval must not touch last_login"
    assert counter["n"] <= 2, "at most the user and self-profile lookups"


def test_upsert_touches_last_login_after_the_interval(db):
    user = _seed(db)
    stale = datetime.utcnow() - timedelta(minutes=11)
    user.last_login = stale
    db.commit()
    database.upsert_user(db, "uid-1", "a@example.com", None)
    db.refresh(user)
    assert user.last_login > stale


def test_history_is_paginated_and_skips_the_payload(db):
    for i in range(5):
        db.add(ReportAnalysis(firebase_uid="uid-1", total_records=i, analysis_json='{"records": []}',
                              created_at=datetime(2024, 1, 1 + i)))
    db.commit()

    page = database.get_user_analyses(db, "uid-1", limit=2, offset=1)
    assert [row.total_records for row in page] == [3, 2]
    assert "analysis_json" not in page[0].__dict__, "the list must not load the full analysis text"


def test_saved_insights_are_returned_without_recomputing(db, monkeypatch):
    _seed(db, studies=0)
    stored = {
        "user": {"user_id": "uid-1", "email": None},
        "patient_info": {"name": "A", "age": "40", "gender": "F", "patient_id": "P", "date": "N/A", "lab_name": "L"},
        "total_records": 0,
        "records": [],
        "health_summary": {"overall_score": 77, "category_scores": {}, "concerns": []},
        "body_systems": [],
        "raw_texts": [],
    }
    import json

    row = ReportAnalysis(firebase_uid="uid-1", total_records=0, analysis_json=json.dumps(stored))
    db.add(row)
    db.commit()

    def must_not_run(*args, **kwargs):
        raise AssertionError("insights recomputed on read")

    monkeypatch.setattr(main.service, "get_health_insights", must_not_run)
    response = main.get_report_by_id(analysis_id=row.id, user=RequestUser(user_id="uid-1", email=None), db=db)
    assert response.health_summary["overall_score"] == 77
