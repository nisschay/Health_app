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


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    api_prefix: str = "/api/v1"
    app_name: str = "Medical Project API"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    require_auth: bool = _as_bool(os.getenv("API_REQUIRE_AUTH"), default=False)
    enable_batch_ingestion_queue: bool = _as_bool(
        os.getenv("ENABLE_BATCH_INGESTION_QUEUE"),
        default=True,
    )
    batch_queue_min_files: int = max(
        2,
        _as_int(os.getenv("BATCH_QUEUE_MIN_FILES"), default=10),
    )
    batch_ingestion_workers: int = max(
        1,
        _as_int(os.getenv("BATCH_INGESTION_WORKERS"), default=4),
    )
    ocr_fallback_enabled: bool = _as_bool(
        os.getenv("PDF_OCR_FALLBACK_ENABLED"),
        default=True,
    )
    ocr_min_text_chars: int = max(
        0,
        _as_int(os.getenv("PDF_OCR_MIN_TEXT_CHARS"), default=120),
    )
    normalization_alias_whitelist_only: bool = _as_bool(
        os.getenv("NORMALIZATION_ALIAS_WHITELIST_ONLY"),
        default=True,
    )
    firebase_credentials_path: str | None = os.getenv("FIREBASE_CREDENTIALS_PATH")
    firebase_project_id: str | None = os.getenv("FIREBASE_PROJECT_ID")
    firebase_clock_skew_seconds: int = min(
        60,
        max(0, _as_int(os.getenv("FIREBASE_CLOCK_SKEW_SECONDS"), default=60)),
    )
    cors_origins: list[str] = field(default_factory=lambda: _as_list(os.getenv("API_CORS_ORIGINS")))


settings = Settings()
