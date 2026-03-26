import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv


# Load central project .env first, with fallback to backend_api/.env.
_APP_DIR = Path(__file__).resolve().parent
_ROOT_ENV_PATH = _APP_DIR.parent.parent / ".env"
_LOCAL_ENV_PATH = _APP_DIR.parent / ".env"
load_dotenv(dotenv_path=_ROOT_ENV_PATH)
load_dotenv(dotenv_path=_LOCAL_ENV_PATH)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_prefix: str = "/api/v1"
    app_name: str = "Medical Project API"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    require_auth: bool = _as_bool(os.getenv("API_REQUIRE_AUTH"), default=False)
    firebase_credentials_path: str | None = os.getenv("FIREBASE_CREDENTIALS_PATH")
    firebase_project_id: str | None = os.getenv("FIREBASE_PROJECT_ID")
    cors_origins: list[str] = field(default_factory=lambda: _as_list(os.getenv("API_CORS_ORIGINS")))


settings = Settings()
