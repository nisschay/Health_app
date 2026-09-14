import uuid
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend_api.app import jobs, main
from backend_api.app.auth import RequestUser
from backend_api.app.database import (
    JOB_DONE,
    JOB_FAILED,
    JOB_INTERRUPTED,
    JOB_RUNNING,
    Base,
    Profile,
    Report,
    ReportFinding,
    ReportJob,
    Study,
    list_trend_points,
    upsert_user,
)
from backend_api.app.findings import findings_from_records, record_from_finding
from backend_api.app.migrations import run_migrations
from backend_api.app.normalization import NORMALIZATION_VERSION


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


@pytest.fixture()
def engine():
    eng = _engine()
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def db(engine):
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _count_statements(engine):
    counter = {"n": 0}

    def _tick(*args):
        counter["n"] += 1

    event.listen(engine, "before_cursor_execute", _tick)
    return counter, lambda: event.remove(engine, "before_cursor_execute", _tick)


# ── Migrations ─────────────────────────────────────────────────────────────────

def test_migrations_build_exactly_the_model_schema():
    eng = _engine()
    run_migrations(eng)
    insp = inspect(eng)
    assert set(insp.get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
    for name, table in Base.metadata.tables.items():
        assert {c["name"] for c in insp.get_columns(name)} == {c.name for c in table.columns}, name
    run_migrations(eng)  # idempotent


def test_a_database_created_before_alembic_is_stamped_then_upgraded():
    """Production predates Alembic: it must be adopted, not recreated, and still gain the new tables."""
    eng = _engine()
    pre_alembic = [t for n, t in Base.metadata.tables.items() if n not in {"report_findings", "report_jobs"}]
    Base.metadata.create_all(eng, tables=pre_alembic)
    with eng.begin() as conn:
        conn.execute(text("DROP INDEX uq_profiles_owner_self"))
    run_migrations(eng)
    with eng.connect() as conn:
        assert conn.execute(text("select version_num from alembic_version")).scalar() == "0002"
    assert {"report_findings", "report_jobs"} <= set(inspect(eng).get_table_names())


# ── Users ──────────────────────────────────────────────────────────────────────

def test_upsert_is_two_statements_and_one_self_profile(engine, db):
    counter, stop = _count_statements(engine)
    user = upsert_user(db, "uid-1", "a@example.com", "Asha")
    stop()
    assert counter["n"] == 2, "insert-or-update the user, insert-or-ignore the self profile"
    upsert_user(db, "uid-1", None, None)
    upsert_user(db, "uid-1", "a@example.com", "Asha")
    assert db.query(Profile).filter(Profile.account_owner_id == user.id).count() == 1


# ── Findings ───────────────────────────────────────────────────────────────────

def test_findings_keep_the_comparator_and_reference_bounds():
    rows = [
        {"Test_Name": "TSH", "Result": "< 0.5", "Unit": "mIU/L", "Reference_Range": "0.4 - 4.0", "Status": "Low",
         "Test_Date": "05-03-2024", "Test_Category": "Thyroid", "Aliases": ["tsh"]},
        {"Test_Name": "", "Result": "1"},
    ]
    findings = findings_from_records(uuid.uuid4(), rows, "r.pdf")
    assert len(findings) == 1, "rows without a test name are not findings"
    f = findings[0]
    assert (f.value_numeric, f.comparator, f.ref_low, f.ref_high) == (0.5, "<", 0.4, 4.0)
    assert f.test_date == date(2024, 3, 5) and f.source_filename == "r.pdf"
    record = record_from_finding(f)
    assert record["Test_Name"] == "TSH" and record["Result"] == "< 0.5" and record["Test_Date"] == "05-03-2024"
    assert record["Result_Numeric"] == 0.5 and record["Status"] == "Low"


def _add_findings(db, report, profile_id, rows):
    findings = findings_from_records(profile_id, rows, report.file_name)
    for finding in findings:
        finding.report_id = report.id
    db.add_all(findings)


def _seed_study(db, reports=2, with_findings=True):
    user = upsert_user(db, "uid-1", "a@example.com", "A")
    profile = db.query(Profile).filter(Profile.account_owner_id == user.id).one()
    study = Study(profile_id=profile.id, name="s")
    db.add(study)
    db.flush()
    for i in range(reports):
        rows = [{"Test_Name": "Haemoglobin (Hb)", "Result": str(12 + i), "Unit": "g/dL", "Reference_Range": "12 - 15",
                 "Status": "Normal", "Test_Date": f"0{i + 1}-01-2024", "Test_Category": "Haematology", "Source_Filename": f"r{i}.pdf"}]
        report = Report(study_id=study.id, file_name=f"r{i}.pdf", file_url="u", report_date=date(2024, 1, i + 1),
                        analysis_data={"records": rows, "patient_info": {"name": "A"}},
                        normalized_records=rows if with_findings else None,
                        is_normalized=with_findings, normalization_version=NORMALIZATION_VERSION if with_findings else 1)
        db.add(report)
        db.flush()
        if with_findings:
            _add_findings(db, report, profile.id, rows)
    db.commit()
    return RequestUser(user_id="uid-1", email="a@example.com"), profile, study


def test_combined_report_query_count_is_independent_of_report_count(engine, db, monkeypatch):
    monkeypatch.setattr(main.service, "get_health_insights", lambda records: {"health_summary": {}, "body_systems": []})
    requester, _, study = _seed_study(db, reports=2)
    study_id, profile_id = study.id, study.profile_id
    db.close()
    fresh = sessionmaker(bind=engine)
    counter, stop = _count_statements(engine)
    small = main.get_combined_study_report(study_id=study_id, user=requester, db=fresh())
    stop()
    n_small = counter["n"]

    db2 = fresh()
    for i in range(2, 9):
        rows = [{"Test_Name": "Haemoglobin (Hb)", "Result": "13", "Status": "Normal", "Test_Date": f"0{i}-02-2024"}]
        report = Report(study_id=study_id, file_name=f"r{i}.pdf", file_url="u", report_date=date(2024, 2, i),
                        analysis_data={}, normalized_records=rows, is_normalized=True, normalization_version=NORMALIZATION_VERSION)
        db2.add(report)
        db2.flush()
        _add_findings(db2, report, profile_id, rows)
    db2.commit()
    db2.close()
    counter, stop = _count_statements(engine)
    large = main.get_combined_study_report(study_id=study_id, user=requester, db=fresh())
    stop()
    assert counter["n"] == n_small, f"{n_small} statements for 2 reports, {counter['n']} for 9"
    assert small.total_records == 2 and large.total_records == 9


def test_legacy_reports_get_findings_on_first_read(db, monkeypatch):
    """Rows saved before report_findings existed are materialised once, then read like the rest."""
    monkeypatch.setattr(main.service, "get_health_insights", lambda records: {"health_summary": {}, "body_systems": []})
    requester, _, study = _seed_study(db, reports=2, with_findings=False)
    assert db.query(ReportFinding).count() == 0
    result = main.get_combined_study_report(study_id=study.id, user=requester, db=db)
    assert result.total_records == 2 and result.records[0].Test_Name == "Haemoglobin (Hb)"
    assert db.query(ReportFinding).count() == 2
    assert {r.normalization_version for r in db.query(Report)} == {NORMALIZATION_VERSION}


def test_trend_is_one_indexed_select(engine, db):
    requester, profile, _ = _seed_study(db, reports=3)
    profile_id = profile.id
    counter, stop = _count_statements(engine)
    points = list_trend_points(db, profile_id, "Haemoglobin (Hb)")
    stop()
    assert counter["n"] == 1
    assert [p.value_numeric for p in points] == [12.0, 13.0, 14.0]
    assert [p.value_numeric for p in main.profile_trend(profile_id=profile_id, test="Haemoglobin (Hb)", user=requester, db=db)] == [12.0, 13.0, 14.0]


# ── Jobs ───────────────────────────────────────────────────────────────────────

class _InlinePool:
    def submit(self, fn, *args):
        fn(*args)


class _Service:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def analyze_reports(self, pdf_files, existing_data_file, include_raw_texts, user, progress_callback):
        for name, _ in pdf_files:
            progress_callback({"type": "file", "file": name, "step": "done", "percent": 100, "processed": 1, "total": 1})
        if self.error:
            raise self.error
        return dict(self.result, user={"user_id": user.user_id, "email": user.email})

    def get_health_insights(self, records):
        return {"health_summary": {"overall_score": 90, "category_scores": {}, "concerns": []}, "body_systems": []}


RESULT = {
    "patient_info": {"name": "A", "age": "40", "gender": "F", "patient_id": "P", "date": "05-03-2024", "lab_name": "L"},
    "records": [{"Test_Name": "HbA1c", "Result": "7.1", "Unit": "%", "Reference_Range": "4 - 5.6", "Status": "High",
                 "Test_Date": "05-03-2024", "Test_Category": "Diabetes", "Source_Filename": "r.pdf"}],
    "total_records": 1,
    "raw_texts": [],
}


@pytest.fixture()
def inline_jobs(engine, monkeypatch):
    monkeypatch.setattr(jobs, "_pool", _InlinePool())
    monkeypatch.setattr(jobs, "SessionLocal", sessionmaker(bind=engine))


def test_history_job_runs_to_done_and_records_the_saved_analysis(db, inline_jobs):
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    job = jobs.submit(db, owner, _Service(RESULT), [("r.pdf", b"%PDF")], None, False, None)
    db.refresh(job)
    assert job.status == JOB_DONE and job.analysis_id is not None
    assert job.progress["stage"] == "done" and job.progress["files"]["r.pdf"]["step"] == "done"
    assert main.get_report_by_id(analysis_id=job.analysis_id, user=RequestUser(user_id="uid-1", email=None), db=db).total_records == 1


def test_study_job_writes_reports_and_findings_server_side(db, inline_jobs):
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    profile = db.query(Profile).one()
    study = Study(profile_id=profile.id, name="s")
    db.add(study)
    db.commit()
    job = jobs.submit(db, owner, _Service(RESULT), [("r.pdf", b"%PDF")], None, False, study.id)
    db.refresh(job)
    assert job.status == JOB_DONE and job.analysis_id is None
    report = db.query(Report).one()
    finding = db.query(ReportFinding).one()
    assert (report.study_id, finding.report_id, finding.canonical_test, finding.value_numeric) == (study.id, report.id, "Glycated Haemoglobin (HbA1C)", 7.1)


def test_a_failed_job_keeps_a_readable_reason(db, inline_jobs):
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    job = jobs.submit(db, owner, _Service(error=ValueError("No medical data could be extracted")), [("r.pdf", b"")], None, False, None)
    db.refresh(job)
    assert job.status == JOB_FAILED and "No medical data" in job.error
    job = jobs.submit(db, owner, _Service(error=RuntimeError("RATE_LIMIT_EXCEEDED: quota")), [("r.pdf", b"")], None, False, None)
    db.refresh(job)
    assert job.status == JOB_FAILED and "rate limited" in job.error


def test_jobs_in_flight_at_startup_are_marked_interrupted(db, engine, monkeypatch):
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    db.add_all([
        ReportJob(owner_id=owner.id, status=JOB_RUNNING, source_filenames=["a.pdf"], progress={}, created_at=datetime.utcnow()),
        ReportJob(owner_id=owner.id, status=JOB_DONE, source_filenames=["b.pdf"], progress={}, created_at=datetime.utcnow()),
    ])
    db.commit()
    monkeypatch.setattr(jobs, "SessionLocal", sessionmaker(bind=engine))
    jobs.mark_interrupted()
    db.expire_all()
    statuses = {j.source_filenames[0]: (j.status, j.error) for j in db.query(ReportJob)}
    assert statuses["a.pdf"] == (JOB_INTERRUPTED, jobs.INTERRUPTED_MESSAGE)
    assert statuses["b.pdf"] == (JOB_DONE, None)
