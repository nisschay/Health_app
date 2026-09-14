import pytest
from fastapi import HTTPException

from backend_api.app.security import RateLimiter, internal_error


def test_requests_under_the_limit_pass():
    limiter = RateLimiter(limit_per_minute=3)
    for _ in range(3):
        limiter.check("user-1")


def test_the_next_request_over_the_limit_is_refused():
    limiter = RateLimiter(limit_per_minute=2)
    limiter.check("user-1")
    limiter.check("user-1")
    with pytest.raises(HTTPException) as excinfo:
        limiter.check("user-1")
    assert excinfo.value.status_code == 429
    assert "Retry-After" in excinfo.value.headers


def test_one_user_cannot_exhaust_another_users_budget():
    limiter = RateLimiter(limit_per_minute=1)
    limiter.check("user-1")
    limiter.check("user-2")


def test_internal_error_hides_the_cause_but_keeps_a_reference():
    """Exception text used to be returned verbatim, leaking SDK and quota detail."""
    exc = RuntimeError("quota exceeded for project secret-project-42 key AIzaSyABC")
    http_exc = internal_error(exc, "Analysis")
    assert http_exc.status_code == 500
    assert "secret-project-42" not in http_exc.detail
    assert "AIzaSy" not in http_exc.detail
    assert "Analysis failed" in http_exc.detail
    reference = http_exc.detail.split("reference ")[1].split()[0]
    assert len(reference) == 12
