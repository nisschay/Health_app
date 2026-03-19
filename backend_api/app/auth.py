from dataclasses import dataclass
from functools import lru_cache

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
    if not settings.firebase_credentials_path:
        return False

    if not firebase_admin._apps:
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(settings.firebase_credentials_path)
            )
        except ValueError as exc:
            # Concurrent requests can race on first init; if default app exists, continue.
            if "already exists" not in str(exc):
                raise

    return True


def get_request_user(authorization: str | None = Header(default=None)) -> RequestUser:
    token = _extract_bearer_token(authorization)

    if not token and not settings.require_auth:
        return RequestUser(user_id="anonymous", authenticated=False)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    if not _firebase_enabled():
        if settings.require_auth:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Firebase authentication is not configured on the API.",
            )
        return RequestUser(user_id="anonymous", authenticated=False)

    try:
        decoded_token = firebase_auth.verify_id_token(token)
    except Exception as exc:  # pragma: no cover - depends on firebase runtime
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token.",
        ) from exc

    return RequestUser(
        user_id=decoded_token["uid"],
        email=decoded_token.get("email"),
        authenticated=True,
    )
