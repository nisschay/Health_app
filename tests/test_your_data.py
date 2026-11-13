"""A user can download everything stored about them, and delete it."""
import json
from datetime import datetime

import pytest
from fastapi import HTTPException
from test_schema_and_jobs import _seed_study

from backend_api.app import main
from backend_api.app.auth import RequestUser
from backend_api.app.database import (
    ExtractionCache,
    Profile,
    Report,
    ReportAnalysis,
    ReportFinding,
    ReportJob,
    Study,
    User,
    save_analysis,
    upsert_user,
)
from backend_api.app.schemas import CreateProfileRequest, RenameStudyRequest


def _everything(db, uid):
    """Rows of every table that belong to one user."""
    user = db.query(User).filter(User.firebase_uid == uid).one_or_none()
    profiles = db.query(Profile).filter(Profile.account_owner_id == (user.id if user else -1)).count()
    return {
        "user": int(user is not None),
        "profiles": profiles,
        "analyses": db.query(ReportAnalysis).filter(ReportAnalysis.firebase_uid == uid).count(),
        "cache": db.query(ExtractionCache).filter(ExtractionCache.owner_uid == uid).count(),
    }


def _populate(db, uid="uid-1"):
    requester, profile, study = _seed_study(db, reports=2)
    user = db.query(User).filter(User.firebase_uid == uid).one()
    save_analysis(db, uid, {"name": "A"}, {"records": [{"Test_Name": "TSH", "Result": "2"}]}, ["h.pdf"])
    db.add(ExtractionCache(text_hash="h1", payload={"x": 1}, owner_uid=uid))
    db.add(ReportJob(owner_id=user.id, study_id=study.id, status="done", source_filenames=["a.pdf"], progress={}, created_at=datetime.utcnow()))
    db.commit()
    return requester, profile, study


def test_deleting_an_account_removes_every_row_and_spares_other_users(db):
    requester, _, _ = _populate(db)
    upsert_user(db, "uid-2", "b@example.com", "B")
    save_analysis(db, "uid-2", {}, {"records": []}, [])
    db.add(ExtractionCache(text_hash="h2", payload={}, owner_uid="uid-2"))
    db.commit()

    main.delete_my_account(user=requester, db=db)
    db.expire_all()

    assert _everything(db, "uid-1") == {"user": 0, "profiles": 0, "analyses": 0, "cache": 0}
    assert db.query(Study).count() == 0 and db.query(Report).count() == 0 and db.query(ReportFinding).count() == 0
    assert db.query(ReportJob).count() == 0
    assert _everything(db, "uid-2") == {"user": 1, "profiles": 1, "analyses": 1, "cache": 1}


def test_deleting_an_account_also_removes_the_login(db, monkeypatch):
    """The browser's deleteUser needs a sign-in from the last few minutes, so the login used to survive."""
    requester, _, _ = _populate(db)
    removed = []
    monkeypatch.setattr(main, "delete_sign_in", removed.append)
    main.delete_my_account(user=requester, db=db)
    assert removed == [requester.user_id]


def test_deleting_a_study_removes_its_reports_and_findings(db):
    requester, _, study = _populate(db)
    main.delete_study_endpoint(study_id=study.id, user=requester, db=db)
    db.expire_all()
    assert db.query(Report).count() == 0 and db.query(ReportFinding).count() == 0
    assert db.query(ReportJob).one().study_id is None, "the job history stays, pointing nowhere"


def test_someone_elses_study_cannot_be_deleted(db):
    _, _, study = _populate(db)
    upsert_user(db, "uid-2", "b@example.com", "B")
    with pytest.raises(HTTPException) as err:
        main.delete_study_endpoint(study_id=study.id, user=RequestUser(user_id="uid-2", email=None), db=db)
    assert err.value.status_code == 404
    assert db.query(Study).count() == 1


def test_a_study_can_be_renamed_by_its_owner_only(db):
    requester, _, study = _populate(db)
    renamed = main.rename_study_endpoint(study_id=study.id, payload=RenameStudyRequest(name="  Yearly checkup "), user=requester, db=db)
    assert renamed.name == "Yearly checkup" and renamed.report_count == 2
    upsert_user(db, "uid-2", "b@example.com", "B")
    with pytest.raises(HTTPException) as err:
        main.rename_study_endpoint(study_id=study.id, payload=RenameStudyRequest(name="Mine"), user=RequestUser(user_id="uid-2", email=None), db=db)
    assert err.value.status_code == 404


def test_a_family_member_can_be_removed_but_not_yourself(db):
    requester, self_profile, _ = _populate(db)
    mother = main.create_profile_endpoint(CreateProfileRequest(full_name="Mum", relationship="Mother"), user=requester, db=db)
    main.delete_profile_endpoint(profile_id=mother.id, user=requester, db=db)
    with pytest.raises(HTTPException) as err:
        main.delete_profile_endpoint(profile_id=self_profile.id, user=requester, db=db)
    assert err.value.status_code == 409
    assert db.query(Profile).count() == 1


def test_a_saved_analysis_can_be_deleted(db):
    requester, _, _ = _populate(db)
    analysis_id = db.query(ReportAnalysis).one().id
    main.delete_history_item(analysis_id=analysis_id, user=requester, db=db)
    assert db.query(ReportAnalysis).count() == 0


def test_the_export_holds_every_profile_report_finding_and_saved_analysis(db):
    requester, _, _ = _populate(db)
    data = json.loads(main.export_my_data(user=requester, db=db).body)
    study = data["profiles"][0]["studies"][0]
    assert data["account"]["email"] == "a@example.com"
    assert len(study["reports"]) == 2
    assert study["reports"][0]["findings"][0]["test"] == "Haemoglobin (Hb)"
    assert data["saved_analyses"][0]["analysis"]["records"][0]["Test_Name"] == "TSH"
