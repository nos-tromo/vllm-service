"""Per-caller prefix-cache salt for the router's completion calls.

vLLM shares one prefix cache across all requests that carry the same
``cache_salt``, or none. A cache hit shortens the time to the first token, so
without a salt one caller can test whether another caller's prompt prefix is
cached. This callback gives every completion call the salt of its caller's
partition, so cached blocks are shared within a partition only:

* a verified Open WebUI user token: that user;
* an ``X-Cache-Scope`` header of 64 lowercase hex characters, the per-user
  pseudonym docint, Nextext and translator send: that scope;
* any other Open WebUI header: all Open WebUI users together;
* none of these: every other caller together.

The salt is an HMAC of the partition under the master key: it cannot be
guessed, it is the same in every worker, and it replaces any ``cache_salt`` a
client sent. Only completion calls are salted. vLLM caches no prefixes for the
embedding, rerank and Whisper models, and pass-through bodies are forwarded
verbatim.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any, Final

from litellm.integrations.custom_logger import CustomLogger
from router_audit import (
    _OPENWEBUI_PLAIN_HEADERS,
    OPENWEBUI_JWT_HEADER,
    _header,
    _proxy_headers,
    normalize_call_type,
    verify_openwebui_token,
)

#: The header carrying an app's per-user pseudonym.
SCOPE_HEADER: Final = "x-cache-scope"
#: The call types whose backend caches prompt prefixes.
SALTED_CALL_TYPES: Final = frozenset({"completion", "text_completion"})
_SCOPE: Final = re.compile(r"[0-9a-f]{64}")
_SALT_LABEL: Final = b"vllm-cache-salt/v1\0"


class CacheSalt(CustomLogger):
    """LiteLLM callback that sets the ``cache_salt`` of every completion call."""

    def __init__(self, *, environ: Mapping[str, str] | None = None, clock: Callable[[], float] = time.time) -> None:
        """Read the salt key and the Open WebUI token secret.

        Args:
            environ: The environment to read ``OPENAI_API_KEY`` and
                ``AUDIT_OPENWEBUI_JWT_SECRET`` from; the process environment
                by default.
            clock: The Unix time, for token expiry.

        Raises:
            ValueError: If ``OPENAI_API_KEY`` is unset or blank, so the
                router does not start without a salt key.
        """
        super().__init__()
        env = os.environ if environ is None else environ
        key = (env.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY must be set: it keys the prefix-cache salt")
        self._key = key.encode("utf-8")
        secret = (env.get("AUDIT_OPENWEBUI_JWT_SECRET") or "").strip()
        self._jwt_secret = secret.encode("utf-8") if secret else None
        self._clock = clock

    def _partition(self, headers: Mapping[str, Any] | None) -> str:
        """Name the cache partition of a call.

        Args:
            headers: The request headers the proxy recorded.

        Returns:
            ``user:<email>``, ``scope:<pseudonym>``, ``open-webui`` or
            ``service``.
        """
        token = _header(headers, OPENWEBUI_JWT_HEADER)
        if token is not None and self._jwt_secret is not None:
            claims = verify_openwebui_token(token, self._jwt_secret, now=self._clock())
            email = claims.get("email") if claims is not None else None
            if isinstance(email, str) and email.strip():
                return f"user:{email.strip()}"
        scope = _header(headers, SCOPE_HEADER)
        if scope is not None and _SCOPE.fullmatch(scope):
            return f"scope:{scope}"
        if token is not None or any(_header(headers, name) is not None for name in _OPENWEBUI_PLAIN_HEADERS):
            return "open-webui"
        return "service"

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Any,
    ) -> dict[str, Any]:
        """Set the call's ``cache_salt``, replacing any the client sent.

        LiteLLM forwards a body field it does not know to the backend in
        ``extra_body``, which the OpenAI client sends as top-level JSON.

        Args:
            user_api_key_dict: The caller's key (unused).
            cache: LiteLLM's cache (unused).
            data: The request data: the body plus what the proxy adds, among
                it the request headers.
            call_type: The proxy's call type, e.g. ``acompletion``.

        Returns:
            The request data.
        """
        value = getattr(call_type, "value", call_type)
        if not isinstance(value, str) or normalize_call_type(value) not in SALTED_CALL_TYPES:
            return data
        try:
            partition = self._partition(_proxy_headers(data))
        except Exception:  # an unreadable header reads as no header
            partition = self._partition(None)
        extra_body = data.get("extra_body")
        if isinstance(extra_body, dict):
            extra_body.pop("cache_salt", None)
        data["cache_salt"] = hmac.new(self._key, _SALT_LABEL + partition.encode("utf-8"), hashlib.sha256).hexdigest()
        return data


#: The callback ``litellm.config.yaml`` registers (``cache_salt.handler``).
handler: Final = CacheSalt()
