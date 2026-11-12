"""Bring the database to the current schema at startup."""
from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from .database import engine as default_engine

BASELINE_REVISION = "0001"
logger = logging.getLogger(__name__)


def run_migrations(engine: Engine | None = None) -> None:
    """A database created before Alembic is stamped at the baseline, then upgraded like any other."""
    engine = engine or default_engine
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(inspect(connection).get_table_names())
        if "alembic_version" not in tables and "users" in tables:
            logger.info("Existing schema without Alembic history; stamping %s", BASELINE_REVISION)
            command.stamp(config, BASELINE_REVISION)
        command.upgrade(config, "head")
