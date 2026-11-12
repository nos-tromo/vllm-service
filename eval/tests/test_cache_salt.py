"""Tests for the router's prefix-cache salt callback (docker/router/cache_salt.py).

The callback runs inside the pinned LiteLLM image. LiteLLM is not installed
here, so the names the module (and ``router_audit``, which it reuses) imports
from it are stubbed in ``sys.modules`` before the import. All data is
synthetic.
"""

import asyncio
import base64
import enum
import hashlib
import hmac
import importlib.util
import json
import os
import string
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

if importlib.util.find_spec("litellm") is not None:
    pytest.skip("cache_salt tests stub LiteLLM; run them where it is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
NOW = 1_800_000_000.0
MASTER_KEY = "synthetic-master-key"
SECRET = "synthetic-shared-secret"
SCOPE_A = "a" * 64
SCOPE_B = "0123456789abcdef" * 4


class _CallTypes(enum.Enum):
    """The ``litellm.types.utils.CallTypes`` values the modules rely on."""

    completion = "completion"
    acompletion = "acompletion"
    text_completion = "text_completion"
    atext_completion = "atext_completion"
    embedding = "embedding"
    aembedding = "aembedding"
    rerank = "rerank"
    arerank = "arerank"
    transcription = "transcription"
    atranscription = "atranscription"
    responses = "responses"
    aresponses = "aresponses"
    pass_through = "pass_through_endpoint"


class _CustomLogger:
    """Stand-in for ``litellm.integrations.custom_logger.CustomLogger``."""

    def __init__(self, **kwargs: object) -> None:
        """Accept and ignore LiteLLM's logger options."""


def _install_litellm_stubs() -> list[str]:
    """Register the stub LiteLLM modules the callbacks import.

    Returns:
        The module names inserted, to remove after the import.
    """
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = _CustomLogger  # type: ignore[attr-defined]
    utils = types.ModuleType("litellm.types.utils")
    utils.CallTypes = _CallTypes  # type: ignore[attr-defined]
    stubs = {
        "litellm": types.ModuleType("litellm"),
        "litellm.integrations": types.ModuleType("litellm.integrations"),
        "litellm.integrations.custom_logger": custom_logger,
        "litellm.types": types.ModuleType("litellm.types"),
        "litellm.types.utils": utils,
    }
    return [name for name, module in stubs.items() if sys.modules.setdefault(name, module) is module]


_inserted_stubs = _install_litellm_stubs()

# docker/router/ is mounted onto PYTHONPATH in the router container; do the same here.
sys.path.insert(0, str(ROOT / "docker" / "router"))

try:
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": MASTER_KEY}):
        import cache_salt
finally:
    for _name in _inserted_stubs:
        del sys.modules[_name]


def _b64(raw: bytes) -> str:
    """Encode bytes as unpadded base64url.

    Args:
        raw: The bytes.

    Returns:
        The encoded text.
    """
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _token(secret: str = SECRET, **overrides: object) -> str:
    """Mint an Open WebUI user token the way Open WebUI does.

    Args:
        secret: The signing secret.
        **overrides: Claims to replace.

    Returns:
        The token.
    """
    claims: dict[str, object] = {
        "sub": "0b7e6a2c-0000-4000-8000-0000000000aa",
        "email": "alice@example.org",
        "name": "Alice Example",
        "role": "user",
        "iss": "open-webui",
        "iat": int(NOW),
        "exp": int(NOW) + 60,
        **overrides,
    }
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64(signature)}"


def _salter(**env: str) -> Any:
    """Build a callback with a fixed clock.

    Args:
        **env: Settings replacing the defaults (master key, JWT secret).

    Returns:
        The callback.
    """
    return cache_salt.CacheSalt(
        environ={"OPENAI_API_KEY": MASTER_KEY, "AUDIT_OPENWEBUI_JWT_SECRET": SECRET, **env},
        clock=lambda: NOW,
    )


def _data(headers: Mapping[str, str] | None = None, **body: object) -> dict[str, Any]:
    """Return proxy request data: the body plus LiteLLM's header copies.

    Args:
        headers: Extra request headers (lowercase names).
        **body: Further body fields.

    Returns:
        The data LiteLLM hands its hooks.
    """
    request_headers = {"content-type": "application/json", **(headers or {})}
    return {
        "model": "synthetic/chat-model",
        "messages": [{"role": "user", "content": "Synthetic question."}],
        **body,
        "proxy_server_request": {
            "url": "http://router/v1/chat/completions",
            "method": "POST",
            "headers": request_headers,
        },
    }


def _salt_of(
    headers: Mapping[str, str] | None = None,
    *,
    salter: Any = None,
    call_type: str = "acompletion",
) -> str:
    """Run the hook on a chat call and return the salt it set.

    Args:
        headers: The request headers.
        salter: The callback; a default one when omitted.
        call_type: The proxy's call type.

    Returns:
        The request's ``cache_salt``.
    """
    data = _data(headers)
    result = asyncio.run((salter or _salter()).async_pre_call_hook(None, None, data, call_type))
    salt = result["cache_salt"]
    assert isinstance(salt, str)
    return salt


SERVICE = _salt_of()
OPEN_WEBUI = _salt_of({"x-openwebui-user-id": "0b7e6a2c-0000-4000-8000-0000000000aa"})


def test_verified_user_gets_a_salt_of_their_own() -> None:
    """Each verified Open WebUI user has a stable salt shared with no one else."""
    alice = _salt_of({"x-openwebui-user-jwt": _token()})
    bob = _salt_of({"x-openwebui-user-jwt": _token(email="bob@example.org", sub="bob")})
    assert alice == _salt_of({"x-openwebui-user-jwt": _token()})
    assert alice != bob
    assert {alice, bob}.isdisjoint({SERVICE, OPEN_WEBUI})


def test_scope_header_gets_a_salt_of_its_own() -> None:
    """Each app scope has a stable salt shared with no other partition."""
    a = _salt_of({"x-cache-scope": SCOPE_A})
    b = _salt_of({"x-cache-scope": SCOPE_B})
    assert a == _salt_of({"x-cache-scope": SCOPE_A})
    assert a != b
    assert {a, b}.isdisjoint({SERVICE, OPEN_WEBUI})


def test_verified_user_wins_over_scope_header() -> None:
    """A verified user token decides the partition before a scope header does."""
    alice = _salt_of({"x-openwebui-user-jwt": _token()})
    assert _salt_of({"x-openwebui-user-jwt": _token(), "x-cache-scope": SCOPE_A}) == alice


@pytest.mark.parametrize(
    "scope",
    ["A" * 64, "a" * 63, "a" * 65, "g" * 64, "", " " + "a" * 63, "a" * 64 + "\n"],
)
def test_malformed_scope_header_is_ignored(scope: str) -> None:
    """A scope that is not 64 lowercase hex characters selects no partition."""
    assert _salt_of({"x-cache-scope": scope}) == SERVICE


@pytest.mark.parametrize(
    "headers",
    [
        {"x-openwebui-user-jwt": _token(secret="another-secret")},
        {"x-openwebui-user-jwt": _token(exp=int(NOW) - 3600)},
        {"x-openwebui-user-jwt": _token(email="")},
        {"x-openwebui-user-jwt": "not-a-token"},
        {"x-openwebui-user-email": "alice@example.org"},
        {"x-openwebui-user-name": "Alice Example"},
    ],
)
def test_unverified_open_webui_calls_share_the_open_webui_salt(headers: dict[str, str]) -> None:
    """Open WebUI calls that name no verified user share one partition, apart from the apps."""
    assert _salt_of(headers) == OPEN_WEBUI


def test_without_a_jwt_secret_open_webui_users_share_one_salt() -> None:
    """Without the shared secret no token verifies, so Open WebUI users share one partition."""
    salter = _salter(AUDIT_OPENWEBUI_JWT_SECRET="")
    assert _salt_of({"x-openwebui-user-jwt": _token()}, salter=salter) == OPEN_WEBUI


def test_unverified_token_does_not_hide_a_scope_header() -> None:
    """A token that fails to verify leaves the scope header in charge."""
    headers = {"x-openwebui-user-jwt": _token(secret="another-secret"), "x-cache-scope": SCOPE_A}
    assert _salt_of(headers) == _salt_of({"x-cache-scope": SCOPE_A})


def test_client_chosen_salt_is_replaced() -> None:
    """A salt the client sent, top-level or in ``extra_body``, never reaches the backend."""
    data = _data(cache_salt="client-salt", extra_body={"cache_salt": "client-salt", "top_k": 20})
    result = asyncio.run(_salter().async_pre_call_hook(None, None, data, "acompletion"))
    assert result["cache_salt"] == SERVICE
    assert result["extra_body"] == {"top_k": 20}


def test_hook_returns_the_request_data() -> None:
    """The hook returns the data, which the proxy then uses as the request."""
    data = _data()
    assert asyncio.run(_salter().async_pre_call_hook(None, None, data, "acompletion")) is data


@pytest.mark.parametrize("call_type", ["acompletion", "completion", "atext_completion", "text_completion"])
def test_completion_call_types_are_salted(call_type: str) -> None:
    """Chat and text completions get a salt, with or without the async prefix."""
    assert _salt_of(call_type=call_type) == SERVICE


def test_call_type_enum_members_are_understood() -> None:
    """A ``CallTypes`` member is read by its value."""
    data = _data()
    result = asyncio.run(_salter().async_pre_call_hook(None, None, data, _CallTypes.acompletion))
    assert result["cache_salt"] == SERVICE


@pytest.mark.parametrize(
    "call_type",
    ["aembedding", "arerank", "atranscription", "aresponses", "pass_through_endpoint", "unknown", ""],
)
def test_other_call_types_are_left_alone(call_type: str) -> None:
    """Calls whose backend caches no prefixes, and pass-through bodies, are not touched."""
    data = _data({"x-cache-scope": SCOPE_A}, cache_salt="client-salt")
    before = json.loads(json.dumps(data))
    asyncio.run(_salter().async_pre_call_hook(None, None, data, call_type))
    assert data == before


def test_missing_headers_fall_back_to_the_service_salt() -> None:
    """Request data without readable headers still gets a salt: the shared one."""
    data: dict[str, Any] = {"model": "synthetic/chat-model", "proxy_server_request": "not-a-mapping"}
    result = asyncio.run(_salter().async_pre_call_hook(None, None, data, "acompletion"))
    assert result["cache_salt"] == SERVICE


def test_salt_passes_vllm_validation() -> None:
    """The salt fits vLLM's check: at most 128 hex characters, so no ``@``, slash, backslash or NUL."""
    for salt in (SERVICE, OPEN_WEBUI, _salt_of({"x-cache-scope": SCOPE_A})):
        assert 0 < len(salt) <= 128
        assert set(salt) <= set(string.hexdigits.lower())


def test_salt_depends_on_the_master_key() -> None:
    """The salt is keyed, so it cannot be computed without the master key."""
    assert _salt_of(salter=_salter(OPENAI_API_KEY="another-master-key")) != SERVICE


@pytest.mark.parametrize("key", ["", "   "])
def test_missing_master_key_is_refused(key: str) -> None:
    """Without a master key the callback refuses to start."""
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        cache_salt.CacheSalt(environ={"OPENAI_API_KEY": key})


def test_hook_is_defined_on_the_registered_class() -> None:
    """LiteLLM calls a callback's pre-call hook only when its own class defines it."""
    assert "async_pre_call_hook" in vars(type(cache_salt.handler))


def test_router_config_registers_the_callback() -> None:
    """The router configuration registers the callback."""
    config = yaml.safe_load((ROOT / "docker" / "litellm.config.yaml").read_text())
    assert "cache_salt.handler" in config["litellm_settings"]["callbacks"]
