import pytest
from fastapi import HTTPException

from backend_api.app.config import settings
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


@pytest.mark.parametrize(
    "endpoint",
    ["analyze_reports", "analyze_reports_stream", "chat_about_report"],
)
def test_every_model_spending_endpoint_is_rate_limited(endpoint):
    """Chat was left unlimited, and it is the endpoint that spends the API key."""
    import inspect

    from backend_api.app import main

    source = inspect.getsource(getattr(main, endpoint))
    assert "rate_limiter.check(user.user_id)" in source


def test_the_limiter_uses_the_configured_limit():
    assert settings.rate_limit_per_minute >= 1
    limiter = RateLimiter(settings.rate_limit_per_minute)
    for _ in range(settings.rate_limit_per_minute):
        limiter.check("user-1")
    with pytest.raises(HTTPException) as excinfo:
        limiter.check("user-1")
    assert excinfo.value.status_code == 429
