import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# backend_api.app.database builds its engine at import time, so the URL has to
# be in place before any backend module is imported.
os.environ.setdefault("DATABASE_URL", "sqlite://")
# Overwrite, never setdefault: a real key in the developer's shell would
# otherwise let a test reach the live Gemini API.
os.environ["GEMINI_API_KEY"] = "test-key-not-used"

from sqlalchemy.dialects.postgresql import JSONB, UUID  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402


@compiles(JSONB, "sqlite")
def _compile_jsonb_on_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid_on_sqlite(type_, compiler, **kw):
    return "CHAR(36)"
