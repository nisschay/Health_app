import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
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


EMAIL_NOT_VERIFIED = "Verify your email address to continue. Check your inbox for the link."


def needs_email_verification(decoded_token: dict) -> bool:
    """Google and other providers vouch for the address; an email-and-password sign-up has to confirm it."""
    provider = (decoded_token.get("firebase") or {}).get("sign_in_provider")
    return provider == "password" and not decoded_token.get("email_verified")


def delete_sign_in(firebase_uid: str) -> bool:
    """Remove the Firebase login itself. The browser cannot do this without a recent sign-in; the admin API can.

    Returns False, after logging, when the login could not be removed; a login that is already gone counts as removed.
    """
    if not _firebase_enabled():
        return False
    try:
        firebase_auth.delete_user(firebase_uid)
    except firebase_auth.UserNotFoundError:
        pass
    except Exception:  # the health data is already gone; report the leftover login instead of failing the request
        logger.exception("Could not delete the Firebase login for a deleted account")
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

    if settings.require_verified_email and needs_email_verification(decoded_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=EMAIL_NOT_VERIFIED)

    return RequestUser(
        user_id=decoded_token["uid"],
        email=decoded_token.get("email"),
    )


def get_signed_in_user(authorization: str | None = Header(default=None)) -> RequestUser:
    """Any valid sign-in, confirmed address or not. Only for deleting the account, which must always work."""
    try:
        return get_request_user(authorization)
    except HTTPException as exc:
        if exc.detail != EMAIL_NOT_VERIFIED:
            raise
    decoded_token = firebase_auth.verify_id_token(
        _extract_bearer_token(authorization), clock_skew_seconds=settings.firebase_clock_skew_seconds
    )
    return RequestUser(user_id=decoded_token["uid"], email=decoded_token.get("email"))
