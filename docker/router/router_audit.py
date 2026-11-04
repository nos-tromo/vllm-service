"""Data-protection audit events for every model call through the LiteLLM router.

The router is where data passes to the models, so every call that reaches a
model through it gets one ``ai_call`` event: the requested model, the
deployment that served it and its revision, the call's size and duration, how
it ended, and the calling app's ``X-Request-ID`` for correlation. What was
sent, and for whom, the calling app records. This module never records
prompts, inputs, outputs or keys; it only hashes a call's single text output
(``output_sha256``) so an app's copy of it can be matched.

LiteLLM loads the module by name from the directory compose puts on
``PYTHONPATH``, twice: ``LITELLM_WORKER_STARTUP_HOOKS=router_audit:start``
opens the worker's audit chain at startup, after any fork, and
``litellm_settings.callbacks`` registers :data:`handler`.

Fail closed. Each event is written in the last hook LiteLLM runs before it
returns the result, and a failed write turns the call into an error wherever
LiteLLM lets a hook do so. Where it does not (pass-through routes, and the
part of a stream already sent) a failed write closes a gate instead: every
later audited call is refused before anything reaches a model, until a write
succeeds again and the trail records how many events could not be written.
``docs/audit.md`` lists what each path does.

Facts are read defensively: whatever fails while one is read (a malformed
response, a hostile token), the event is still written, without that fact
and marked ``incomplete``. Request headers are taken only from what the proxy
itself records, never from a request body: a pass-through route's payload is
the client's body.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import contextvars
import hashlib
import hmac
import json
import os
import re
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar
from urllib.parse import urlsplit

from audit_event import Actor, AiCall, AuditEmitter, sha256_text
from audit_runtime import REQUEST_ID_HEADER, clean_request_id, open_audit_stream
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes
from starlette.datastructures import Headers

#: The compose project, which the audit store's verifier binds chains to.
SERVICE: Final = "vllm-service"
#: The lock every audit writer in the router container takes. It lives on a
#: tmpfs of the router's own: the router's ``/tmp`` is the upload spool volume
#: it shares with other services, where the library's default lock would be
#: reachable from their containers.
LOCK_PATH: Final = "/run/audit/audit-stream.lock"
#: LiteLLM's call type for the ``pass_through_endpoints`` routes.
PASS_THROUGH: Final = "pass_through_endpoint"
#: Call types whose LiteLLM handler runs ``async_post_call_success_hook``
#: before it returns a non-streaming result. Every other call type is recorded
#: from the deployment hook, the last one its handler runs.
PROXY_SUCCESS_HOOK_TYPES: Final = frozenset(
    {"completion", "text_completion", "embedding", "responses", "anthropic_messages"}
)
#: Call types whose output tokens count generated text.
_TEXT_OUTPUT_TYPES: Final = frozenset({"completion", "text_completion", "responses", "anthropic_messages"})

#: Open WebUI's signed user token (``FORWARD_USER_INFO_HEADER_JWT``).
OPENWEBUI_JWT_HEADER: Final = "x-openwebui-user-jwt"
#: Open WebUI's chat id header (``FORWARD_SESSION_INFO_HEADER_CHAT_ID``).
OPENWEBUI_CHAT_HEADER: Final = "x-openwebui-chat-id"
#: The ``iss`` claim Open WebUI puts in its user token.
OPENWEBUI_ISSUER: Final = "open-webui"
#: Open WebUI's unsigned user headers, sent when it has no token secret.
_OPENWEBUI_PLAIN_HEADERS: Final = ("x-openwebui-user-email", "x-openwebui-user-id", "x-openwebui-user-name")
#: Clock skew tolerated on a token's ``exp`` and ``iat``, in seconds.
JWT_LEEWAY_S: Final = 30.0
#: Longest token accepted, in characters.
MAX_JWT_LENGTH: Final = 4096

_B64URL: Final = re.compile(r"[A-Za-z0-9_-]+")
_RELEASE_TAG: Final = re.compile(r"v\d")
_T = TypeVar("_T")
_SYSTEM: Final = Actor(user=None, via="system")
_CALLER: Final = Actor(user=None, via="service")
_CALL_TYPE_VALUES: Final = frozenset(str(member.value) for member in CallTypes)


def normalize_call_type(value: str) -> str:
    """Return LiteLLM's name for a call type without its async prefix.

    The proxy hands its hooks ``acompletion`` for a chat call but ``rerank``
    for a rerank call; both are ``CallTypes`` values. The name without the
    prefix is the stable one: ``acompletion`` becomes ``completion``, while
    ``anthropic_messages``, whose ``a`` belongs to the name, stays.

    Args:
        value: The call type as LiteLLM passed it.

    Returns:
        The call type without a leading ``a`` that only marks the async twin.
    """
    if value.startswith("a") and value[1:] in _CALL_TYPE_VALUES:
        return value[1:]
    return value


#: Every name ``AUDIT_ROUTER_CALL_TYPES`` may list.
KNOWN_CALL_TYPES: Final = frozenset(normalize_call_type(value) for value in _CALL_TYPE_VALUES)


def parse_call_types(value: str | None) -> frozenset[str] | None:
    """Parse ``AUDIT_ROUTER_CALL_TYPES``: the call types to audit.

    Args:
        value: A comma-separated list such as ``"completion, rerank"``;
            ``None`` or blank for every call type.

    Returns:
        The listed call types, or ``None`` for all of them.

    Raises:
        ValueError: If an entry is not a LiteLLM call type, so a misspelt
            name stops the router instead of switching auditing off.
    """
    if value is None or not value.strip():
        return None
    names = {part.strip().lower() for part in value.split(",")} - {""}
    unknown = sorted(names - KNOWN_CALL_TYPES)
    if unknown:
        raise ValueError(f"unknown call type(s) in AUDIT_ROUTER_CALL_TYPES: {', '.join(unknown)}")
    return frozenset(names)


def service_version(value: str | None) -> str:
    """Return the release recorded as ``service_version``.

    Args:
        value: ``AUDIT_SERVICE_VERSION``, which compose fills from the
            stack's image tag: a release tag such as ``v0.7.0`` or a dev
            build's date and commit.

    Returns:
        The version without a release tag's ``v``, or ``unknown``.
    """
    text = (value or "").strip()
    if _RELEASE_TAG.match(text):
        text = text[1:]
    return text or "unknown"


class AuditUnavailable(Exception):
    """An audit event could not be written, so the call must not succeed.

    LiteLLM answers a raised exception with an error response built from its
    ``message`` and ``status_code``.

    Attributes:
        message: The error text the client receives.
        status_code: The HTTP status the client receives.
        refused: Whether the call was stopped before it reached a model.
    """

    def __init__(self, *, refused: bool) -> None:
        """Build the error.

        Args:
            refused: Whether the call was stopped before it reached a model.
        """
        super().__init__("audit log unavailable")
        self.message = "audit log unavailable"
        self.status_code = 503
        self.refused = refused


@dataclass
class _Call:
    """What the router knows about the audited call of one request."""

    call_type: str
    route: str | None
    model_alias: str
    stream: bool
    started: float
    batch: int | None = None
    documents: int | None = None
    request_id: str | None = None
    actor: Actor = _CALLER
    identity: dict[str, object] = field(default_factory=dict)
    headers_seen: bool = False
    attempts: int = 0
    deployment: str | None = None
    revision: str | None = None
    incomplete: bool = False
    written: bool = False
    lost: bool = False


#: The audited call of the current request. Set by the proxy's pre-call hook
#: in the request's task; the router's tasks for that request inherit it.
_CURRENT: Final[contextvars.ContextVar[_Call | None]] = contextvars.ContextVar("router_audit_call", default=None)


def _read(call: _Call, read: Callable[[], _T], fallback: _T) -> _T:
    """Read one fact for a call's event without risking the event.

    Whatever goes wrong while a fact is read, the call keeps its record: the
    fact falls back and the event is marked ``incomplete``. Only facts are
    read this way, never the write.

    Args:
        call: The call the fact belongs to.
        read: Reads the fact.
        fallback: The value to use when reading fails.

    Returns:
        The fact, or ``fallback``.
    """
    try:
        return read()
    except Exception:  # any failure, e.g. a RecursionError from hostile input
        call.incomplete = True
        return fallback


def _get(obj: object, name: str) -> Any:
    """Read an attribute or a mapping key, whichever the object has.

    Args:
        obj: A LiteLLM response, chunk or part of one.
        name: The field to read.

    Returns:
        The value, or ``None`` when the object has no such field.
    """
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: object) -> int | None:
    """Return a value if it is a plain integer.

    Args:
        value: Any value.

    Returns:
        The integer, or ``None`` (also for booleans).
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    """Return a value if it is a plain number.

    Args:
        value: Any value.

    Returns:
        The number as a float, or ``None`` (also for booleans).
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _count(value: object) -> int | None:
    """Count the items a request sent in one field.

    Args:
        value: A request field such as an embedding ``input``.

    Returns:
        The number of items: a list's length, ``1`` for a single string.
    """
    if isinstance(value, list) or isinstance(value, tuple):
        return len(value)
    if isinstance(value, str):
        return 1
    return None


def _tokens(usage: object) -> tuple[int | None, int | None]:
    """Read the token counts a response reported.

    Args:
        usage: A response's ``usage`` object or mapping.

    Returns:
        ``(input_tokens, output_tokens)``, each ``None`` when not reported.
    """
    if usage is None:
        return None, None
    prompt = _int(_get(usage, "prompt_tokens"))
    completion = _int(_get(usage, "completion_tokens"))
    if prompt is None:
        prompt = _int(_get(usage, "input_tokens"))
    if completion is None:
        completion = _int(_get(usage, "output_tokens"))
    return prompt, completion


def _output_text(call_type: str, response: object) -> str | None:
    """Return a call's text output when it is a single, unambiguous text.

    Args:
        call_type: The normalized call type.
        response: The response LiteLLM returned.

    Returns:
        The text of the only choice (a chat or text completion) or of a
        transcription; ``None`` for every other shape.
    """
    if call_type == "transcription":
        text = _get(response, "text")
        return text if isinstance(text, str) and text else None
    if call_type not in ("completion", "text_completion"):
        return None
    choices = _get(response, "choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    if call_type == "completion":
        text = _get(_get(choices[0], "message"), "content")
    else:
        text = _get(choices[0], "text")
    return text if isinstance(text, str) and text else None


def _hidden_model_id(response: object) -> str | None:
    """Return the deployment id LiteLLM attached to a response.

    Args:
        response: The response LiteLLM returned.

    Returns:
        The id, or ``None`` when the response carries none.
    """
    model_id = _get(_get(response, "_hidden_params"), "model_id")
    return model_id if isinstance(model_id, str) and model_id else None


def _route(user_api_key_dict: object) -> str | None:
    """Return the request path, which LiteLLM's key check records.

    Args:
        user_api_key_dict: The caller's key object.

    Returns:
        The path, or ``None``.
    """
    route = getattr(user_api_key_dict, "request_route", None)
    return route if isinstance(route, str) and route else None


def _proxy_headers(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the request headers the proxy recorded for a model-list call.

    LiteLLM builds ``proxy_server_request`` itself for these routes, after
    it has dropped any value the client sent under that name, so its headers
    are the request's own. Never use this on a pass-through payload: that
    starts as the client's body.

    Args:
        data: The request data of a chat, embedding, rerank, ... call.

    Returns:
        The headers, keyed by lowercase name, or ``None`` when absent.
    """
    headers = _get(_get(data, "proxy_server_request"), "headers")
    return headers if isinstance(headers, Mapping) else None


def _passthrough_failure_headers(request_data: Mapping[str, Any]) -> Headers | None:
    """Return the request headers LiteLLM recorded for a failed pass-through call.

    The failure payload of a pass-through route is the client's body with
    LiteLLM's own fields added, so only what a JSON body cannot contain is
    trusted: the Starlette ``Headers`` of the request, which LiteLLM puts
    under ``litellm_params.proxy_server_request`` once it has read the body.

    Args:
        request_data: The failure hook's payload.

    Returns:
        The headers, or ``None`` when LiteLLM did not get that far.
    """
    headers = _get(_get(_get(request_data, "litellm_params"), "proxy_server_request"), "headers")
    return headers if isinstance(headers, Headers) else None


def _header(headers: Mapping[str, Any] | None, name: str) -> str | None:
    """Read one request header.

    Args:
        headers: Headers as LiteLLM hands them over, keyed by lowercase name.
        name: The header name, lowercase.

    Returns:
        The value, or ``None`` when absent.
    """
    if headers is None:
        return None
    value = headers.get(name)
    return value if isinstance(value, str) else None


def _model_info(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``model_info`` of the deployment the router picked.

    Args:
        kwargs: The kwargs of the router's call to a deployment.

    Returns:
        The deployment's ``model_info`` from ``litellm.config.yaml``, or an
        empty mapping.
    """
    for key in ("metadata", "litellm_metadata"):
        info = _get(_get(kwargs, key), "model_info")
        if isinstance(info, Mapping):
            return info
    return {}


def _deployment_of(kwargs: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return the id and revision of the deployment the router picked.

    Args:
        kwargs: The kwargs of the router's call to a deployment.

    Returns:
        ``(id, revision)`` from its ``model_info``, each ``None`` when unset.
    """
    info = _model_info(kwargs)
    deployment, revision = info.get("id"), info.get("revision")
    return (
        deployment if isinstance(deployment, str) and deployment else None,
        revision.strip() if isinstance(revision, str) and revision.strip() else None,
    )


def _b64url(segment: str) -> bytes:
    """Decode one unpadded base64url segment of a token.

    Args:
        segment: The segment.

    Returns:
        The decoded bytes.

    Raises:
        ValueError: If the segment is not base64url.
    """
    if not _B64URL.fullmatch(segment):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def verify_openwebui_token(token: str, secret: bytes, *, now: float) -> dict[str, Any] | None:
    """Check an Open WebUI user token and return its claims.

    Open WebUI signs its token with HS256 and the shared secret
    (``FORWARD_USER_INFO_HEADER_JWT_SECRET``), naming itself as issuer and
    expiring the token after a few minutes. The signature is checked with
    HS256 before anything the token carries is parsed, so unsigned input
    never reaches the JSON decoder; whatever fails while the token is
    decoded, it is simply not verified.

    Args:
        token: The ``X-OpenWebUI-User-Jwt`` header value.
        secret: The shared secret.
        now: The current Unix time.

    Returns:
        The claims, or ``None`` when the token is malformed, signed with
        another key or algorithm, from another issuer, expired, or issued in
        the future.
    """
    try:
        return _verify_openwebui_token(token, secret, now)
    except Exception:  # e.g. a RecursionError from deeply nested JSON
        return None


def _verify_openwebui_token(token: str, secret: bytes, now: float) -> dict[str, Any] | None:
    """Check a token as :func:`verify_openwebui_token` describes; may raise.

    Args:
        token: The token.
        secret: The shared secret.
        now: The current Unix time.

    Returns:
        The claims of a valid token, else ``None``.
    """
    if len(token) > MAX_JWT_LENGTH:
        return None
    parts = token.split(".")
    if len(parts) != 3 or not all(_B64URL.fullmatch(part) for part in parts):
        return None
    head, body, signature = parts
    expected = hmac.new(secret, f"{head}.{body}".encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url(signature)):
        return None
    # Signed with the shared secret: only now is what it signed parsed.
    header = json.loads(_b64url(head))
    claims = json.loads(_b64url(body))
    if not isinstance(header, dict) or header.get("alg") != "HS256" or not isinstance(claims, dict):
        return None
    if claims.get("iss") != OPENWEBUI_ISSUER:
        return None
    expires = _number(claims.get("exp"))
    if expires is None or expires + JWT_LEEWAY_S < now:
        return None
    issued = claims.get("iat")
    if issued is not None:
        issued_at = _number(issued)
        if issued_at is None or issued_at - JWT_LEEWAY_S > now:
            return None
    return claims


def _configured_passthrough_targets() -> dict[str, str]:
    """Map every configured pass-through path to its backend's host.

    Returns:
        ``{"/gliner": "gliner", ...}`` from ``pass_through_endpoints`` in
        ``litellm.config.yaml``, as the running proxy loaded it.
    """
    from litellm.proxy import proxy_server  # loaded long before any call arrives

    targets: dict[str, str] = {}
    for entry in proxy_server.general_settings.get("pass_through_endpoints") or []:
        path, target = _get(entry, "path"), _get(entry, "target")
        if isinstance(path, str) and isinstance(target, str):
            host = urlsplit(target).hostname
            if host:
                targets[path] = host
    return targets


class _StreamTally:
    """What a stream carried, observed chunk by chunk without keeping it."""

    def __init__(self) -> None:
        """Start with an empty tally."""
        self._digest = hashlib.sha256()
        self._text_seen = False
        self._ambiguous = False
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def observe(self, chunk: object) -> bool:
        """Record one chunk.

        Args:
            chunk: A streamed chunk as LiteLLM yields it.

        Returns:
            Whether the chunk carries a finish reason.
        """
        finished = False
        choices = _get(chunk, "choices")
        for choice in choices if isinstance(choices, list) else ():
            if _get(choice, "index") not in (0, None):
                self._ambiguous = True
            text = _get(_get(choice, "delta"), "content")
            if text is None:
                text = _get(choice, "text")
            if isinstance(text, str) and text:
                self._digest.update(text.encode("utf-8"))
                self._text_seen = True
            if _get(choice, "finish_reason"):
                finished = True
        prompt, completion = _tokens(_get(chunk, "usage"))
        if prompt is not None:
            self.input_tokens = prompt
        if completion is not None:
            self.output_tokens = completion
        return finished

    def discard(self) -> None:
        """Give up on the output facts after a chunk could not be read."""
        self._ambiguous = True
        self.input_tokens = None
        self.output_tokens = None

    @property
    def output_sha256(self) -> str | None:
        """The hash of the streamed text, if there was one unambiguous text.

        Returns:
            The SHA-256 of the concatenated text deltas of the only choice,
            equal to :func:`audit_event.sha256_text` of the assembled text.
        """
        return self._digest.hexdigest() if self._text_seen and not self._ambiguous else None


class RouterAudit(CustomLogger):
    """LiteLLM callback that writes one ``ai_call`` event per model call."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        emitter_factory: Callable[[], AuditEmitter] | None = None,
        passthrough_targets: Callable[[], Mapping[str, str]] = _configured_passthrough_targets,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Read the configuration; nothing is written until the first event.

        Args:
            environ: The environment to read the ``AUDIT_*`` settings from;
                the process environment by default.
            emitter_factory: Builds this process's emitter; by default one
                writing to the stream ``AUDIT_OUTPUT`` selects.
            passthrough_targets: Returns the backend host of each
                pass-through path.
            clock: The Unix time, for token expiry.
            monotonic: A monotonic clock, for call durations.

        Raises:
            ValueError: If ``AUDIT_ROUTER_CALL_TYPES`` names an unknown call
                type.
        """
        super().__init__()
        env = os.environ if environ is None else environ
        self._call_types = parse_call_types(env.get("AUDIT_ROUTER_CALL_TYPES"))
        secret = (env.get("AUDIT_OPENWEBUI_JWT_SECRET") or "").strip()
        self._jwt_secret = secret.encode("utf-8") if secret else None
        self._output = env.get("AUDIT_OUTPUT")
        self._version = service_version(env.get("AUDIT_SERVICE_VERSION"))
        self._emitter_factory = emitter_factory or self._open_emitter
        self._passthrough_targets = passthrough_targets
        self._targets: Mapping[str, str] = {}
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._emitter: AuditEmitter | None = None
        self._emitter_pid: int | None = None
        self._unwritten = 0

    # -- the emitter ---------------------------------------------------------

    def _open_emitter(self) -> AuditEmitter:
        """Build the production emitter.

        Returns:
            An emitter writing to the stream ``AUDIT_OUTPUT`` selects.
        """
        stream = open_audit_stream(self._output, lock_path=LOCK_PATH)
        return AuditEmitter(SERVICE, self._version, stream=stream)

    def _mode(self) -> dict[str, object]:
        """Describe the operating mode for the ``config.audit`` event.

        Returns:
            The audited call types and whether Open WebUI's signed user
            token is checked; no secrets.
        """
        call_types: object = "all" if self._call_types is None else sorted(self._call_types)
        return {"call_types": call_types, "openwebui_jwt": self._jwt_secret is not None}

    def open(self) -> AuditEmitter:
        """Return this process's emitter, starting its chain on first use.

        A forked child gets a chain of its own; it never writes to its
        parent's.

        Returns:
            The started emitter.

        Raises:
            AuditWriteError: If the chain's first events cannot be written.
            OSError: If the audit stream cannot be opened.
        """
        with self._lock:
            pid = os.getpid()
            if self._emitter is not None and self._emitter_pid == pid:
                return self._emitter
            emitter = self._emitter_factory()
            emitter.start()
            try:
                emitter.emit("admin", "config.audit", _SYSTEM, criteria=self._mode())
            except BaseException:
                with contextlib.suppress(Exception):
                    emitter.close()
                raise
            self._emitter, self._emitter_pid = emitter, pid
            return emitter

    def forget_inherited(self) -> None:
        """Drop the parent's emitter in a forked child (``os.register_at_fork``).

        The child opens its own chain on first use; the parent's must not get
        the child's shutdown event at exit.
        """
        inherited = self._emitter
        self._lock = threading.RLock()
        self._emitter = None
        self._emitter_pid = None
        if inherited is not None:
            atexit.unregister(inherited.close)

    def _record_gap(self, emitter: AuditEmitter) -> None:
        """Write the pending ``audit.recovered`` event, if events were lost.

        Args:
            emitter: The started emitter. Call with ``self._lock`` held.
        """
        if self._unwritten:
            emitter.emit("admin", "audit.recovered", _SYSTEM, criteria={"unwritten_events": self._unwritten})
            self._unwritten = 0

    def _admit(self) -> None:
        """Let a call through only while the audit log can be written.

        Raises:
            AuditUnavailable: With ``refused=True`` while an earlier event
                could not be written and the gap cannot be recorded yet, or
                the chain cannot be opened.
        """
        try:
            emitter = self.open()
            with self._lock:
                self._record_gap(emitter)
        except Exception as exc:  # whatever stops the write stops the call
            raise AuditUnavailable(refused=True) from exc

    def _write(self, call: _Call, *, outcome: str, criteria: dict[str, object], ai: AiCall, recipient: str) -> None:
        """Write the call's event, after a pending gap record.

        Args:
            call: The call.
            outcome: ``ok`` or ``error``.
            criteria: The call's non-content facts.
            ai: Its model provenance.
            recipient: Where the data went.

        Raises:
            AuditUnavailable: If the event could not be written; the gate
                then stays closed until a later write succeeds.
        """
        try:
            emitter = self.open()
            with self._lock:
                self._record_gap(emitter)
                emitter.emit(
                    "ai_call",
                    f"router.{call.call_type}",
                    call.actor,
                    outcome=outcome,
                    request_id=call.request_id,
                    criteria=criteria,
                    recipient=recipient,
                    ai=ai,
                )
        except Exception as exc:  # never swallowed: counted, then the call fails
            with self._lock:
                self._unwritten += 1
            call.lost = True
            raise AuditUnavailable(refused=False) from exc
        call.written = True

    # -- building the event --------------------------------------------------

    def _audits(self, call_type: str) -> bool:
        """Whether calls of a type are audited (``AUDIT_ROUTER_CALL_TYPES``).

        Args:
            call_type: The normalized call type.

        Returns:
            ``True`` when every type is audited or this one is listed.
        """
        return self._call_types is None or call_type in self._call_types

    def _identity(self, headers: Mapping[str, Any] | None) -> tuple[Actor, dict[str, object]]:
        """Map a request's forwarded user identity to the event's actor.

        Every caller shares the master key, so the router cannot tell them
        apart by themselves. Open WebUI names its user in a token it signs
        with a secret the router shares; only a valid token names the actor.

        Args:
            headers: The request headers.

        Returns:
            The actor, and the criteria saying how the identity was decided.
        """
        token = _header(headers, OPENWEBUI_JWT_HEADER)
        if token is not None:
            claims = None
            if self._jwt_secret is not None:
                claims = verify_openwebui_token(token, self._jwt_secret, now=self._clock())
            email = claims.get("email") if claims is not None else None
            if not isinstance(email, str) or not email.strip():
                return _CALLER, {"forwarded_identity": "invalid"}
            identity: dict[str, object] = {"caller": "open-webui", "forwarded_identity": "signed"}
            chat_id = clean_request_id(_header(headers, OPENWEBUI_CHAT_HEADER))
            if chat_id is not None:
                identity["chat_id"] = chat_id
            return Actor(user=email.strip(), via="service"), identity
        if any(_header(headers, name) is not None for name in _OPENWEBUI_PLAIN_HEADERS):
            return _CALLER, {"forwarded_identity": "unsigned"}
        return _CALLER, {}

    def _see_headers(self, call: _Call, headers: Mapping[str, Any] | None) -> None:
        """Take the request id and the actor from the first headers seen.

        Args:
            call: The call.
            headers: The request headers, if this hook has them.
        """
        if call.headers_seen or headers is None:
            return
        call.headers_seen = True
        call.request_id = _read(call, lambda: clean_request_id(_header(headers, REQUEST_ID_HEADER.lower())), None)
        call.actor, call.identity = _read(call, lambda: self._identity(headers), (_CALLER, {}))

    def _target(self, route: str | None) -> str | None:
        """Return the backend host a pass-through route forwards to.

        Args:
            route: The request path.

        Returns:
            The host, or ``None`` when the path is not configured.
        """
        if route is None:
            return None
        if not self._targets:
            self._targets = self._passthrough_targets()
        return self._targets.get(route)

    def _finish(
        self,
        call: _Call,
        *,
        outcome: str,
        status: int | None,
        response: object = None,
        tally: _StreamTally | None = None,
        error: str | None = None,
    ) -> None:
        """Write the call's one event, unless it has been written.

        Args:
            call: The call.
            outcome: ``ok`` or ``error``.
            status: The HTTP status of the call, when known.
            response: The response, for its token counts and text output.
            tally: What a stream carried, instead of a response.
            error: The error's class name, for ``outcome="error"``.

        Raises:
            AuditUnavailable: If the event could not be written.
        """
        if call.written:
            return
        try:
            criteria, ai, recipient = self._facts(call, status=status, response=response, tally=tally, error=error)
        except Exception:  # backstop: the facts are read defensively already
            call.incomplete = True
            criteria, ai, recipient = self._bare_facts(call, status=status, error=error)
        self._write(call, outcome=outcome, criteria=criteria, ai=ai, recipient=recipient)

    def _facts(
        self,
        call: _Call,
        *,
        status: int | None,
        response: object,
        tally: _StreamTally | None,
        error: str | None,
    ) -> tuple[dict[str, object], AiCall, str]:
        """Collect a call's event fields; a fact that cannot be read is left out.

        Args:
            call: The call.
            status: The HTTP status of the call, when known.
            response: The response, for its token counts and text output.
            tally: What a stream carried, instead of a response.
            error: The error's class name.

        Returns:
            The criteria, the model provenance and the recipient.
        """
        criteria = self._base_criteria(call, status=status, error=error)
        input_tokens: int | None = None
        output_tokens: int | None = None
        output_sha256: str | None = None
        if call.call_type == PASS_THROUGH:
            deployment = _read(call, lambda: self._target(call.route), None)
        else:
            deployment = call.deployment
            if deployment is None:
                deployment = _read(call, lambda: _hidden_model_id(response), None)
            criteria["stream"] = call.stream
            criteria["attempts"] = call.attempts
            if tally is not None:
                input_tokens, output_tokens, output_sha256 = (
                    tally.input_tokens,
                    tally.output_tokens,
                    tally.output_sha256,
                )
            elif response is not None:
                input_tokens, output_tokens = _read(call, lambda: _tokens(_get(response, "usage")), (None, None))
                text = _read(call, lambda: _output_text(call.call_type, response), None)
                output_sha256 = sha256_text(text) if text is not None else None
            if call.call_type not in _TEXT_OUTPUT_TYPES:
                output_tokens = None
        if call.batch is not None:
            criteria["batch"] = call.batch
        if call.documents is not None:
            criteria["documents"] = call.documents
        criteria.update(call.identity)
        if call.incomplete:
            criteria["incomplete"] = True
        ai = AiCall(
            model_alias=call.model_alias,
            model_id=deployment,
            revision=call.revision,
            output_sha256=output_sha256,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return criteria, ai, f"model:{deployment or 'unknown'}"

    def _base_criteria(self, call: _Call, *, status: int | None, error: str | None) -> dict[str, object]:
        """Return the criteria every event of a call carries.

        Args:
            call: The call.
            status: The HTTP status of the call, when known.
            error: The error's class name.

        Returns:
            Duration, route, status and error, as known.
        """
        criteria: dict[str, object] = {"duration_ms": round((self._monotonic() - call.started) * 1000)}
        if call.route is not None:
            criteria["route"] = call.route
        if status is not None:
            criteria["status"] = status
        if error is not None:
            criteria["error"] = error
        return criteria

    def _bare_facts(
        self, call: _Call, *, status: int | None, error: str | None
    ) -> tuple[dict[str, object], AiCall, str]:
        """Return the event fields known without reading anything further.

        Args:
            call: The call.
            status: The HTTP status of the call, when known.
            error: The error's class name.

        Returns:
            The criteria, marked incomplete, the model provenance and the
            recipient.
        """
        criteria = {**self._base_criteria(call, status=status, error=error), "incomplete": True}
        ai = AiCall(model_alias=call.model_alias, model_id=call.deployment, revision=call.revision)
        return criteria, ai, f"model:{call.deployment or 'unknown'}"

    # -- LiteLLM hooks -------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Any,
    ) -> None:
        """Open the call's record, or refuse it while audit writes fail.

        LiteLLM runs this for every authenticated model call before anything
        is sent, and answers with an error when it raises.

        Args:
            user_api_key_dict: The caller's key; carries the request path.
            cache: LiteLLM's cache (unused).
            data: The request data: the body plus what the proxy adds, among
                it the request headers; for a pass-through route, the
                client's body alone, from which nothing but its counts is
                read.
            call_type: The proxy's call type, e.g. ``acompletion``.

        Raises:
            AuditUnavailable: While the audit log cannot be written.
        """
        value = getattr(call_type, "value", call_type)
        kind = normalize_call_type(value) if isinstance(value, str) and value else "unknown"
        if not self._audits(kind):
            return
        self._admit()
        call = _Call(call_type=kind, route=None, model_alias=kind, stream=False, started=self._monotonic())
        _CURRENT.set(call)
        call.route = _read(call, lambda: _route(user_api_key_dict), None)
        if kind == PASS_THROUGH:
            call.model_alias = call.route or kind
            return  # its headers arrive with a later hook; the body never supplies them
        model = _read(call, lambda: data.get("model"), None)
        call.model_alias = model if isinstance(model, str) and model else call.route or kind
        call.stream = _read(call, lambda: data.get("stream") is True, False)
        if kind == "embedding":
            call.batch = _read(call, lambda: _count(data.get("input")), None)
        elif kind == "rerank":
            call.documents = _read(call, lambda: _count(data.get("documents")), None)
        self._see_headers(call, _read(call, lambda: _proxy_headers(data), None))

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: Any) -> None:
        """Count a call to a deployment, retries included; gate each one.

        Args:
            kwargs: The router's kwargs for the deployment call.
            call_type: LiteLLM's call type.

        Raises:
            AuditUnavailable: While the audit log cannot be written, so a
                retry is not sent either.
        """
        call = _CURRENT.get()
        if call is None:
            return  # not a client call, e.g. a health check
        self._admit()
        call.attempts += 1
        deployment, revision = _read(call, lambda: _deployment_of(kwargs), (None, None))
        if deployment is not None:
            call.deployment = deployment
        call.revision = revision

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict[str, Any],
        response: Any,
        call_type: Any,
    ) -> None:
        """Record a call whose handler has no later success hook (rerank, ASR).

        Args:
            request_data: The router's kwargs for the deployment call.
            response: The deployment's response.
            call_type: LiteLLM's call type.

        Raises:
            AuditUnavailable: If the event could not be written; the call
                then fails instead of returning the result.
        """
        call = _CURRENT.get()
        if call is None or call.written or call.call_type in PROXY_SUCCESS_HOOK_TYPES or call.call_type == PASS_THROUGH:
            return
        self._finish(call, outcome="ok", status=200, response=response)

    async def async_post_call_success_hook(self, data: dict[str, Any], user_api_key_dict: Any, response: Any) -> None:
        """Record a non-streaming call before its result is returned.

        Args:
            data: The request data.
            user_api_key_dict: The caller's key.
            response: The result about to be returned.

        Raises:
            AuditUnavailable: If the event could not be written; the client
                then gets an error and no result.
        """
        call = _CURRENT.get()
        if call is None or call.written or call.call_type == PASS_THROUGH:
            return
        self._finish(call, outcome="ok", status=200, response=response)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """Pass a stream through; record it when the model's stream ends.

        Chunks go out as they arrive until one carries a finish reason. That
        chunk and the ones after it (the usage chunk) are held until the
        stream ends and the event is written, so a failed write ends the
        stream with an error instead of its final chunk and ``[DONE]``.

        Args:
            user_api_key_dict: The caller's key.
            response: The stream of chunks.
            request_data: The request data.

        Yields:
            The stream's chunks, unchanged and in order.

        Raises:
            AuditUnavailable: If the event could not be written.
        """
        call = _CURRENT.get()
        if call is None or call.written:
            async for chunk in response:
                yield chunk
            return
        tally = _StreamTally()
        held: list[Any] = []
        try:
            async for chunk in response:
                try:
                    finished = tally.observe(chunk)
                except Exception:  # a chunk that cannot be read still goes out
                    call.incomplete = True
                    tally.discard()
                    finished = False
                if finished or held:
                    held.append(chunk)
                else:
                    yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            with contextlib.suppress(AuditUnavailable):
                self._finish(call, outcome="error", status=None, error="client_disconnected")
            raise
        self._finish(call, outcome="ok", status=200, tally=tally)
        for chunk in held:
            yield chunk

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, Any] | None = None,
    ) -> None:
        """Record a pass-through call once its backend has answered.

        LiteLLM ignores what this hook raises, so a failed write cannot stop
        this response; it closes the gate for the calls after it.

        Args:
            data: The client's request body (not read).
            user_api_key_dict: The caller's key.
            response: The backend's response.
            request_headers: The request's headers, as the proxy received
                them.
            litellm_call_info: LiteLLM's routing details (unused).
        """
        call = _CURRENT.get()
        if call is None or call.written or call.call_type != PASS_THROUGH:
            return
        self._see_headers(call, request_headers)
        status = _read(call, lambda: _int(getattr(response, "status_code", None)), None)
        outcome = "error" if status is not None and status >= 400 else "ok"
        with contextlib.suppress(AuditUnavailable):
            self._finish(call, outcome=outcome, status=status)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:
        """Record a call that failed after it was sent.

        Covers backend errors and timeouts and a backend's error status on a
        pass-through route. A call that failed because the audit log could
        not be written is counted for the ``audit.recovered`` record instead.
        A call that never reached a deployment (an unknown model, or refused
        by the audit gate) sent nothing and gets no event.

        Args:
            request_data: The request data.
            original_exception: Why the call failed.
            user_api_key_dict: The caller's key.
            traceback_str: LiteLLM's traceback text (unused, never recorded).
        """
        call = _CURRENT.get()
        if call is None or call.written:
            return  # e.g. a refused key: such requests are not audited
        if call.call_type != PASS_THROUGH and call.attempts == 0:
            return  # no deployment was called (pass-through routes have no deployment hooks)
        if isinstance(original_exception, AuditUnavailable):
            # The log is not writable: count the lost record once instead of
            # trying again. A failed write has counted it already; a retry the
            # gate refused after data had gone out has not.
            if original_exception.refused and not call.lost:
                with self._lock:
                    self._unwritten += 1
            return
        if call.call_type == PASS_THROUGH:
            headers = _read(call, lambda: _passthrough_failure_headers(request_data), None)
        else:
            headers = _read(call, lambda: _proxy_headers(request_data), None)
        self._see_headers(call, headers)
        status = _read(call, lambda: _int(getattr(original_exception, "status_code", None)), None)
        if status is None or not 400 <= status <= 599:
            status = 500
        with contextlib.suppress(AuditUnavailable):
            self._finish(call, outcome="error", status=status, error=type(original_exception).__name__)


#: The callback ``litellm.config.yaml`` registers (``router_audit.handler``).
handler: Final = RouterAudit()
os.register_at_fork(after_in_child=handler.forget_inherited)


def start() -> None:
    """Open this worker's audit chain (``LITELLM_WORKER_STARTUP_HOOKS``).

    LiteLLM runs it in every worker before the worker serves, after any fork;
    if the chain cannot be opened, the worker does not start.
    """
    handler.open()
