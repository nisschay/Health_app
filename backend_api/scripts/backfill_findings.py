#!/usr/bin/env python3
"""Re-normalise every stored report and rebuild its report_findings rows. Safe to re-run."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend_api.app.database import Report, SessionLocal, Study  # noqa: E402
from backend_api.app.saving import rebuild_report  # noqa: E402


def main() -> None:
    reports = findings = 0
    with SessionLocal() as db:
        for report, profile_id in db.query(Report, Study.profile_id).join(Study, Study.id == Report.study_id):
            findings += len(rebuild_report(db, report, profile_id))
            reports += 1
            db.commit()
    print(f"Rebuilt {findings} findings across {reports} reports")


if __name__ == "__main__":
    main()
