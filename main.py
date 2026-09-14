# Uvicorn entrypoint for the container: `uvicorn main:app`.
from backend_api.app.main import app  # noqa: F401
