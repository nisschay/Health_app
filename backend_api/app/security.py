"""Upload validation, per-user rate limiting, and safe error responses."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import HTTPException, Request, UploadFile, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import settings

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
EXISTING_DATA_SUFFIXES = (".csv", ".xlsx")
_MB = 1024 * 1024
_CHUNK_BYTES = 64 * 1024


def _too_large(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=detail)


async def _read_within_limit(upload: UploadFile, limit: int, detail: str) -> bytes:
    """Read an upload in chunks, stopping as soon as it passes the limit.

    Reading first and measuring after still buffers an oversized file in memory.
    """
    if upload.size is not None and upload.size > limit:
        raise _too_large(detail)

    chunks: list[bytes] = []
    read = 0
    while True:
        chunk = await upload.read(_CHUNK_BYTES)
        if not chunk:
            break
        read += len(chunk)
        if read > limit:
            raise _too_large(detail)
        chunks.append(chunk)
    return b"".join(chunks)


async def read_pdf_uploads(uploads: list[UploadFile] | None) -> list[tuple[str, bytes]]:
    """Read PDF uploads into memory, rejecting anything outside the configured limits."""
    files = uploads or []
    if len(files) > settings.max_upload_files:
        raise _too_large(f"Too many files. Upload at most {settings.max_upload_files} at a time.")

    per_file_limit = settings.max_upload_file_mb * _MB
    total_limit = settings.max_upload_total_mb * _MB
    payloads: list[tuple[str, bytes]] = []
    total = 0

    for upload in files:
        name = upload.filename or "uploaded.pdf"
        remaining = total_limit - total
        allowance = min(per_file_limit, remaining)
        detail = (
            f"'{name}' is larger than {settings.max_upload_file_mb} MB."
            if allowance == per_file_limit
            else f"Upload exceeds {settings.max_upload_total_mb} MB in total."
        )
        payload = await _read_within_limit(upload, allowance, detail)
        total += len(payload)
        if not payload.startswith(PDF_MAGIC):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"'{name}' is not a PDF.",
            )
        payloads.append((name, payload))

    return payloads


async def read_existing_data_upload(upload: UploadFile | None) -> tuple[str, bytes] | None:
    """Read the optional spreadsheet merged with the extracted reports."""
    if upload is None:
        return None

    name = upload.filename or "medical-data.xlsx"
    if not name.lower().endswith(EXISTING_DATA_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Existing data must be a .csv or .xlsx file.",
        )

    payload = await _read_within_limit(
        upload,
        settings.max_upload_file_mb * _MB,
        f"'{name}' is larger than {settings.max_upload_file_mb} MB.",
    )
    return name, payload


class RateLimiter:
    """Fixed 60-second sliding window per user. One process, one container."""

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = max(1, int(60 - (now - hits[0])))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please wait a moment and try again.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)


rate_limiter = RateLimiter(settings.rate_limit_per_minute)


def internal_error(exc: Exception, context: str) -> HTTPException:
    """Log the real cause against a correlation id and return a safe message."""
    correlation_id = uuid.uuid4().hex[:12]
    logger.exception("%s failed [%s]", context, correlation_id, exc_info=exc)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{context} failed. Quote reference {correlation_id} if you report this.",
    )


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies on every route, not just the upload endpoints."""

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        limit = settings.max_upload_total_mb * _MB
        if declared and declared.isdigit() and int(declared) > limit:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"detail": f"Request body exceeds {settings.max_upload_total_mb} MB."},
            )
        return await call_next(request)
