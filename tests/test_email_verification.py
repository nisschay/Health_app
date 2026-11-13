"""An email-and-password account must confirm its address before it can use the API."""
import pytest
from fastapi import HTTPException

from backend_api.app import auth


def _token(provider, verified):
    return {"uid": "u1", "email": "a@example.com", "email_verified": verified, "aud": "", "firebase": {"sign_in_provider": provider}}


@pytest.fixture
def decoded(monkeypatch):
    holder = {}
    monkeypatch.setattr(auth, "_firebase_enabled", lambda: True)
    monkeypatch.setattr(auth.firebase_auth, "verify_id_token", lambda token, **kwargs: holder["token"])
    return holder


def test_an_unconfirmed_password_account_is_refused(decoded):
    decoded["token"] = _token("password", False)
    with pytest.raises(HTTPException) as err:
        auth.get_request_user("Bearer t")
    assert err.value.status_code == 403 and err.value.detail == auth.EMAIL_NOT_VERIFIED


@pytest.mark.parametrize(("provider", "verified"), [("password", True), ("google.com", False), ("google.com", True)])
def test_confirmed_or_google_accounts_pass(decoded, provider, verified):
    decoded["token"] = _token(provider, verified)
    assert auth.get_request_user("Bearer t").user_id == "u1"


def test_the_rule_can_be_switched_off(decoded, monkeypatch):
    import dataclasses

    monkeypatch.setattr(auth, "settings", dataclasses.replace(auth.settings, require_verified_email=False))
    decoded["token"] = _token("password", False)
    assert auth.get_request_user("Bearer t").user_id == "u1"


def test_an_unconfirmed_account_can_still_delete_itself(decoded):
    decoded["token"] = _token("password", False)
    assert auth.get_signed_in_user("Bearer t").user_id == "u1"
