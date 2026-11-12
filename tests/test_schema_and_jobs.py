import uuid
from datetime import date, datetime, timedelta

import pytest
from fastapi import HTTPException
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
from backend_api.app.saving import backfill_findings
from backend_api.app.schemas import CreateProfileRequest


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

def _create_pre_alembic_schema(eng):
    """Production's schema before Alembic: today's models minus everything a migration added."""
    pre_alembic = [t for n, t in Base.metadata.tables.items() if n not in {"report_findings", "report_jobs"}]
    Base.metadata.create_all(eng, tables=pre_alembic)
    with eng.begin() as conn:
        conn.execute(text("DROP INDEX uq_profiles_owner_self"))  # 0002
        conn.execute(text("DROP INDEX ix_extraction_cache_owner_uid"))  # 0003
        conn.execute(text("ALTER TABLE extraction_cache DROP COLUMN owner_uid"))  # 0003


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
    _create_pre_alembic_schema(eng)
    run_migrations(eng)
    with eng.connect() as conn:
        assert conn.execute(text("select version_num from alembic_version")).scalar() == "0003"
    assert {"report_findings", "report_jobs"} <= set(inspect(eng).get_table_names())


def test_migration_keeps_one_self_profile_when_an_account_has_two():
    """Before the unique index, concurrent sign-ins could create two self profiles; the upgrade must not crash on them."""
    eng = _engine()
    _create_pre_alembic_schema(eng)
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO users (id, firebase_uid) VALUES (1, 'uid-1')"))
        for pid, rel, created in (("a" * 32, "self", "2024-01-01"), ("b" * 32, "Self", "2024-02-01"), ("c" * 32, "mother", "2024-03-01")):
            conn.execute(
                text("INSERT INTO profiles (id, account_owner_id, full_name, relationship, created_at, updated_at) "
                     "VALUES (:id, 1, 'X', :rel, :c, :c)"),
                {"id": pid, "rel": rel, "c": created},
            )
    run_migrations(eng)
    with eng.connect() as conn:
        rows = dict(conn.execute(text("SELECT id, relationship FROM profiles")).all())
    assert rows == {"a" * 32: "self", "b" * 32: "other", "c" * 32: "mother"}, "oldest stays self, nothing is deleted"


def test_a_second_self_profile_is_a_conflict_not_a_server_error(db):
    upsert_user(db, "uid-1", "a@example.com", "A")
    payload = CreateProfileRequest(full_name="Me again", relationship="Self")
    with pytest.raises(HTTPException) as err:
        main.create_profile_endpoint(payload=payload, user=RequestUser(user_id="uid-1", email=None), db=db)
    assert err.value.status_code == 409
    assert db.query(Profile).count() == 1


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


def test_legacy_reports_are_readable_before_backfill_and_reads_never_write(db, monkeypatch):
    """Rows saved before report_findings existed are built in memory on read; only the backfill writes them."""
    monkeypatch.setattr(main.service, "get_health_insights", lambda records: {"health_summary": {}, "body_systems": []})
    requester, _, study = _seed_study(db, reports=2, with_findings=False)
    result = main.get_combined_study_report(study_id=study.id, user=requester, db=db)
    assert result.total_records == 2 and result.records[0].Test_Name == "Haemoglobin (Hb)"
    assert result.reports_with_data == 2
    db.rollback()
    assert db.query(ReportFinding).count() == 0, "a GET must not insert findings"


def test_backfill_fills_only_reports_without_findings_and_is_idempotent(db):
    _seed_study(db, reports=2, with_findings=False)
    assert backfill_findings(db) == (2, 2)
    assert backfill_findings(db) == (0, 0), "a second run finds nothing to do"
    assert db.query(ReportFinding).count() == 2
    assert backfill_findings(db, rebuild_all=True) == (2, 2)
    assert db.query(ReportFinding).count() == 2, "rebuilding replaces rows, never duplicates them"


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


TWO_FILES = {
    "patient_info": {"name": "A", "date": "05-03-2024"},
    "records": [{"Test_Name": "HbA1c", "Result": "7.1", "Status": "High", "Test_Date": "05-03-2024",
                 "Test_Category": "Diabetes", "Source_Filename": "good.pdf"}],
    "total_records": 1,
    "raw_texts": [],
    "failed_files": {"bad.pdf": "No extractable text found."},
}


def _study(db):
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    study = Study(profile_id=db.query(Profile).one().id, name="s")
    db.add(study)
    db.commit()
    return owner, study


def test_one_unreadable_file_does_not_lose_the_rest_of_the_batch(db, inline_jobs):
    owner, study = _study(db)
    job = jobs.submit(db, owner, _Service(TWO_FILES), [("good.pdf", b"%PDF"), ("bad.pdf", b"")], None, False, study.id)
    db.refresh(job)
    assert job.status == JOB_DONE
    assert [r.file_name for r in db.query(Report)] == ["good.pdf"], "no empty report for the failed file"
    assert job.progress["warnings"] == ["bad.pdf: No extractable text found."]


def test_a_file_without_a_date_is_skipped_not_fatal(db, inline_jobs):
    owner, study = _study(db)
    undated = dict(TWO_FILES, patient_info={"name": "A"}, failed_files={},
                   records=TWO_FILES["records"] + [{"Test_Name": "TSH", "Result": "2", "Status": "Normal",
                                                    "Test_Date": "", "Source_Filename": "undated.pdf"}])
    job = jobs.submit(db, owner, _Service(undated), [("good.pdf", b"%PDF"), ("undated.pdf", b"%PDF")], None, False, study.id)
    db.refresh(job)
    assert job.status == JOB_DONE and db.query(Report).count() == 1
    assert "undated.pdf" in job.progress["warnings"][0]


def test_when_no_file_can_be_saved_the_job_fails_with_each_reason(db, inline_jobs):
    owner, study = _study(db)
    nothing = dict(TWO_FILES, records=[], failed_files={"good.pdf": "Extraction failed", "bad.pdf": "No extractable text found."})
    job = jobs.submit(db, owner, _Service(nothing), [("good.pdf", b""), ("bad.pdf", b"")], None, False, study.id)
    db.refresh(job)
    assert job.status == JOB_FAILED and "bad.pdf" in job.error and db.query(Report).count() == 0


def test_a_failed_save_leaves_no_reports_and_a_failed_job(db, inline_jobs, monkeypatch):
    """Reports and the done status commit together, so a failure cannot leave saved reports behind a failed job."""
    owner, study = _study(db)

    def broken_commit_after_reports(*args, **kwargs):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(jobs, "normalize_analysis_payload", lambda result, service: result)
    real = jobs.save_to_study

    def save_then_break(session, *args):
        warnings = real(session, *args)
        session.flush()
        broken_commit_after_reports()
        return warnings

    monkeypatch.setattr(jobs, "save_to_study", save_then_break)
    job = jobs.submit(db, owner, _Service(TWO_FILES), [("good.pdf", b"%PDF")], None, False, study.id)
    db.refresh(job)
    assert job.status == JOB_FAILED and db.query(Report).count() == 0


def test_a_job_that_stopped_making_progress_is_reported_failed(db):
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    old = datetime.utcnow() - timedelta(hours=2)
    job = ReportJob(owner_id=owner.id, status=JOB_RUNNING, source_filenames=["a.pdf"],
                    progress={"stage": "processing", "files": {}, "processed": 0, "total": 1}, created_at=old, started_at=old)
    db.add(job)
    db.commit()
    response = main.read_job(job_id=job.id, user=RequestUser(user_id="uid-1", email=None), db=db)
    assert response.status == JOB_FAILED and response.error == jobs.STALE_MESSAGE


def test_a_finished_result_does_not_overwrite_an_interrupted_job(db, inline_jobs, monkeypatch):
    owner, study = _study(db)

    class InterruptedMidway(_Service):
        def analyze_reports(self, **kwargs):
            with jobs.SessionLocal() as other:
                other.query(ReportJob).update({"status": JOB_INTERRUPTED})
                other.commit()
            return super().analyze_reports(**kwargs)

    job = jobs.submit(db, owner, InterruptedMidway(TWO_FILES), [("good.pdf", b"%PDF")], None, False, study.id)
    db.refresh(job)
    assert job.status == JOB_INTERRUPTED and db.query(Report).count() == 0


class _NeverRuns:
    def submit(self, fn, *args):
        pass  # the job stays queued, like one waiting behind others


def test_a_user_cannot_queue_more_than_the_active_job_limit(db, engine, monkeypatch):
    monkeypatch.setattr(jobs, "_pool", _NeverRuns())
    monkeypatch.setattr(jobs, "_pending", 0)
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    for _ in range(jobs.settings.max_active_jobs_per_user):
        jobs.submit(db, owner, _Service(RESULT), [("r.pdf", b"%PDF")], None, False, None)
    with pytest.raises(jobs.TooManyJobs):
        jobs.submit(db, owner, _Service(RESULT), [("r.pdf", b"%PDF")], None, False, None)


def test_the_server_refuses_new_jobs_when_its_queue_is_full(db, monkeypatch):
    monkeypatch.setattr(jobs, "_pool", _NeverRuns())
    monkeypatch.setattr(jobs, "_pending", jobs.settings.max_pending_jobs)
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    with pytest.raises(jobs.TooManyJobs):
        jobs.submit(db, owner, _Service(RESULT), [("r.pdf", b"%PDF")], None, False, None)
    assert db.query(ReportJob).count() == 0


def test_a_finished_job_frees_its_queue_slot(db, inline_jobs, monkeypatch):
    monkeypatch.setattr(jobs, "_pending", 0)
    owner = upsert_user(db, "uid-1", "a@example.com", "A")
    jobs.submit(db, owner, _Service(RESULT), [("r.pdf", b"%PDF")], None, False, None)
    jobs.submit(db, owner, _Service(error=ValueError("bad")), [("r.pdf", b"")], None, False, None)
    assert jobs._pending == 0


def test_a_recently_seen_user_costs_one_select_per_request(engine, db):
    upsert_user(db, "uid-1", "a@example.com", "A")
    counter, stop = _count_statements(engine)
    main._current_account_owner(RequestUser(user_id="uid-1", email="a@example.com"), db)
    stop()
    assert counter["n"] == 1, "polling must not write"


def test_another_users_study_looks_missing(db):
    _, _, study = _seed_study(db, reports=1)
    upsert_user(db, "uid-2", "b@example.com", "B")
    with pytest.raises(HTTPException) as err:
        main.get_combined_study_report(study_id=study.id, user=RequestUser(user_id="uid-2", email=None), db=db)
    assert err.value.status_code == 404
