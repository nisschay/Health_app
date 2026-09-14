import sys
from pathlib import Path

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend_api.app import database  # noqa: E402

target_metadata = database.Base.metadata


def run() -> None:
    connection = context.config.attributes.get("connection")
    if connection is None:
        with database.engine.connect() as connection:
            _run(connection)
            connection.commit()
    else:
        _run(connection)


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


run()
