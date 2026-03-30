#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend_api.app.database import Report, ReportAnalysis, SessionLocal
from backend_api.app.normalization import normalize_records
from backend_api.app.services import MedicalAnalysisService


def _empty_insights() -> dict[str, Any]:
    return {
        "health_summary": {"overall_score": 0, "category_scores": {}, "concerns": []},
        "body_systems": [],
    }


def _normalize_payload(payload: dict[str, Any], service: MedicalAnalysisService) -> dict[str, Any]:
    normalized = dict(payload)
    raw_records = normalized.get("records", [])
    rows = raw_records if isinstance(raw_records, list) else []
    normalized_records = normalize_records([row for row in rows if isinstance(row, dict)])

    normalized["records"] = normalized_records
    normalized["total_records"] = len(normalized_records)

    insights = service.get_health_insights(normalized_records) if normalized_records else _empty_insights()
    normalized["health_summary"] = insights.get("health_summary", _empty_insights()["health_summary"])
    normalized["body_systems"] = insights.get("body_systems", [])

    return normalized


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def main() -> None:
    db = SessionLocal()
    service = MedicalAnalysisService(api_key="")

    report_rows = db.query(Report).all()
    history_rows = db.query(ReportAnalysis).all()

    updated_reports = 0
    skipped_reports = 0
    updated_histories = 0
    skipped_histories = 0

    try:
        for row in report_rows:
            payload = row.analysis_data if isinstance(row.analysis_data, dict) else {}
            before = _stable_json(payload)
            after_payload = _normalize_payload(payload, service)
            after = _stable_json(after_payload)

            if before != after:
                row.analysis_data = after_payload
                updated_reports += 1
            else:
                skipped_reports += 1

        for row in history_rows:
            try:
                payload = json.loads(row.analysis_json)
            except Exception:
                skipped_histories += 1
                continue

            before = _stable_json(payload)
            after_payload = _normalize_payload(payload, service)
            after = _stable_json(after_payload)

            if before != after:
                row.analysis_json = json.dumps(after_payload)
                row.total_records = int(after_payload.get("total_records", 0) or 0)

                patient_info = after_payload.get("patient_info", {})
                if isinstance(patient_info, dict):
                    row.patient_name = patient_info.get("name")
                    row.patient_age = patient_info.get("age")
                    row.patient_gender = patient_info.get("gender")
                    row.patient_id = patient_info.get("patient_id")
                    row.lab_name = patient_info.get("lab_name")
                    row.report_date = patient_info.get("date")

                updated_histories += 1
            else:
                skipped_histories += 1

        db.commit()

        print("Normalization migration complete")
        print(f"- Study report payloads updated: {updated_reports}")
        print(f"- Study report payloads unchanged: {skipped_reports}")
        print(f"- History payloads updated: {updated_histories}")
        print(f"- History payloads unchanged/skipped: {skipped_histories}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
