import json
import os
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path

from fastapi import Header, HTTPException, status

from .config import settings

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    from firebase_admin import credentials
except ImportError:  # pragma: no cover - optional dependency in local setup
    firebase_admin = None
    firebase_auth = None
    credentials = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestUser:
    user_id: str
    email: str | None = None
    authenticated: bool = False


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


@lru_cache(maxsize=1)
def _firebase_enabled() -> bool:
    if not firebase_admin or not credentials or not firebase_auth:
        return False

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH") or settings.firebase_credentials_path
    if not service_account_json and not cred_path:
        return False

    if cred_path and not service_account_json:
        creds_path = Path(cred_path)
        if not creds_path.exists():
            # In local/dev, allow API to continue without Firebase if auth is optional.
            print(f"[WARN] Firebase credentials file not found: {creds_path}")
            return False

    if not firebase_admin._apps:
        # In local/dev, allow API to continue without Firebase if auth is optional.
        try:
            if service_account_json:
                # Production (Hugging Face): load from env secret.
                cred_dict = json.loads(service_account_json)
                cred = credentials.Certificate(cred_dict)
            elif cred_path:
                # Local dev: load from file path.
                cred = credentials.Certificate(cred_path)
            else:
                raise ValueError(
                    "No Firebase credentials found. Set FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_CREDENTIALS_PATH"
                )

            firebase_admin.initialize_app(cred)
        except ValueError as exc:
            # Concurrent requests can race on first init; if default app exists, continue.
            if "already exists" not in str(exc):
                print(f"[WARN] Firebase initialization failed: {exc}")
                return False
        except Exception as exc:
            # Avoid taking down the API when Firebase is misconfigured in optional-auth mode.
            print(f"[WARN] Firebase initialization failed: {exc}")
            return False

    return True


def get_request_user(authorization: str | None = Header(default=None)) -> RequestUser:
    token = _extract_bearer_token(authorization)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    if not _firebase_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase authentication is not configured on the API.",
        )

    try:
        decoded_token = firebase_auth.verify_id_token(
            token,
            clock_skew_seconds=settings.firebase_clock_skew_seconds,
        )
    except firebase_auth.ExpiredIdTokenError as exc:  # pragma: no cover - depends on firebase runtime
        logger.warning("Firebase token expired: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase token expired. Please sign in again.",
        ) from exc
    except firebase_auth.RevokedIdTokenError as exc:  # pragma: no cover - depends on firebase runtime
        logger.warning("Firebase token revoked: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase token revoked. Please sign in again.",
        ) from exc
    except firebase_auth.InvalidIdTokenError as exc:  # pragma: no cover - depends on firebase runtime
        logger.warning("Firebase token invalid: %s", exc)
        raw_message = str(exc)
        if "Token used too early" in raw_message or "clock is set correctly" in raw_message:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Firebase token rejected due to system clock mismatch. Sync your system time, then sign in again.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token. Please refresh your session.",
        ) from exc
    except firebase_auth.CertificateFetchError as exc:  # pragma: no cover - depends on firebase runtime
        logger.warning("Firebase certificate fetch failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to validate Firebase token right now. Please retry.",
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on firebase runtime
        logger.warning("Firebase token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token. Please refresh your session.",
        ) from exc

    expected_project = (settings.firebase_project_id or "").strip()
    token_project = str(decoded_token.get("aud") or "").strip()
    if expected_project and token_project and token_project != expected_project:
        logger.warning(
            "Firebase token project mismatch: expected=%s token_aud=%s",
            expected_project,
            token_project,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase token project mismatch. Please sign out and sign in again.",
        )

    return RequestUser(
        user_id=decoded_token["uid"],
        email=decoded_token.get("email"),
        authenticated=True,
    )
