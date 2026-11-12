#!/usr/bin/env python3
"""Re-normalise every stored report and rebuild its report_findings rows. Safe to re-run.

Run this after a normaliser change. Reports that merely lack findings are filled in
automatically at server start; this script rebuilds all of them, in one transaction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend_api.app.database import SessionLocal  # noqa: E402
from backend_api.app.saving import backfill_findings  # noqa: E402


def main() -> None:
    with SessionLocal() as db:
        reports, findings = backfill_findings(db, rebuild_all=True, wait=True)
    print(f"Rebuilt {findings} findings across {reports} reports")


if __name__ == "__main__":
    main()
