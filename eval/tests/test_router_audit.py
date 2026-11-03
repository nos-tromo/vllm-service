"""Tests for the router's audit callback (docker/router/router_audit.py).

The callback runs inside the pinned LiteLLM image. LiteLLM is not installed
here, so the two names the module imports from it are stubbed in
``sys.modules`` before the import. Each scenario calls the hooks in the order
the pinned LiteLLM calls them for that route, inside one asyncio task so the
request's context carries from hook to hook as it does in the proxy, and ends
with ``verify_chain`` over the captured audit stream. All data is synthetic.
"""

import asyncio
import base64
import enum
import hashlib
import hmac
import importlib.util
import json
import sys
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.datastructures import Headers
from typing_extensions import override

if importlib.util.find_spec("litellm") is not None:
    pytest.skip("router_audit tests stub LiteLLM; run them where it is not installed", allow_module_level=True)


class _CallTypes(enum.Enum):
    """The ``litellm.types.utils.CallTypes`` values the callback relies on."""

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
    anthropic_messages = "anthropic_messages"
    aanthropic_messages = "aanthropic_messages"
    pass_through = "pass_through_endpoint"


class _CustomLogger:
    """Stand-in for ``litellm.integrations.custom_logger.CustomLogger``."""

    def __init__(self, **kwargs: object) -> None:
        """Accept and ignore LiteLLM's logger options."""


def _install_litellm_stubs() -> list[str]:
    """Register the stub LiteLLM modules the callback imports.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker" / "router"))

try:
    import router_audit
    from audit_event import AuditEmitter, sha256_text, verify_chain
finally:
    for _name in _inserted_stubs:
        del sys.modules[_name]

NOW = 1_800_000_000.0
SECRET = "synthetic-shared-secret"
REVISION = "0123456789abcdef0123456789abcdef01234567"
ANSWER = "Synthetic answer for Alice Example."
PROMPT = "Summarize the note about Alice Example."
TARGETS = {"/gliner": "gliner", "/pooling": "embed-sparse"}


class _Stream:
    """An in-memory audit stream whose writes can be made to fail."""

    def __init__(self) -> None:
        """Start empty and working."""
        self.broken = False
        self._parts: list[str] = []

    def write(self, s: str) -> int:
        """Store the text, or fail like a closed pipe.

        Args:
            s: The text.

        Returns:
            The number of characters written.

        Raises:
            OSError: While the stream is broken.
        """
        if self.broken:
            raise OSError("synthetic write failure")
        self._parts.append(s)
        return len(s)

    def flush(self) -> None:
        """Nothing is buffered."""

    def getvalue(self) -> str:
        """Return everything written.

        Returns:
            The text.
        """
        return "".join(self._parts)


class _BackendError(Exception):
    """A backend's error, as LiteLLM raises it."""

    status_code = 500


class _Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current reading.

        Returns:
            Seconds.
        """
        return self.now


def _targets() -> Mapping[str, str]:
    """Return the pass-through targets of the test configuration.

    Returns:
        Path to backend host.
    """
    return TARGETS


def _audit(
    env: Mapping[str, str] | None = None,
    stream: _Stream | None = None,
    clock: _Clock | None = None,
    targets: Callable[[], Mapping[str, str]] = _targets,
) -> tuple[Any, _Stream]:
    """Build a callback writing to an in-memory stream.

    Args:
        env: The ``AUDIT_*`` settings.
        stream: The stream to write to; a new one by default.
        clock: The monotonic clock; a new one by default.
        targets: Returns the pass-through targets.

    Returns:
        The callback and its stream.
    """
    out = stream if stream is not None else _Stream()
    audit = router_audit.RouterAudit(
        environ={"AUDIT_OPENWEBUI_JWT_SECRET": SECRET, **(env or {})},
        emitter_factory=lambda: AuditEmitter("vllm-service", "0.7.0", stream=out, heartbeat_interval_s=0),
        passthrough_targets=targets,
        clock=lambda: NOW,
        monotonic=clock if clock is not None else _Clock(),
    )
    return audit, out


def _events(stream: _Stream) -> list[dict[str, Any]]:
    """Parse every line written so far.

    Args:
        stream: The audit stream.

    Returns:
        The events in order.
    """
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def _ai_calls(stream: _Stream) -> list[dict[str, Any]]:
    """Return the ``ai_call`` events written so far.

    Args:
        stream: The audit stream.

    Returns:
        The model-call events in order.
    """
    return [event for event in _events(stream) if event["action"] == "ai_call"]


def _assert_chain(stream: _Stream) -> None:
    """Assert the stream is one intact chain.

    Args:
        stream: The audit stream.
    """
    report = verify_chain(_events(stream))
    assert report.ok, report.problems


def _assert_no_content(stream: _Stream) -> None:
    """Assert no prompt, answer or credential text reached the stream.

    Args:
        stream: The audit stream.
    """
    text = stream.getvalue()
    for needle in ("Alice Example", "Summarize", "synthetic-shared-secret", "sk-synthetic"):
        assert needle not in text


def _run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Run one request's hooks in one task, as the proxy does.

    Args:
        scenario: The coroutine function calling the hooks.
    """
    asyncio.run(scenario())


def _key(route: str) -> SimpleNamespace:
    """Return the caller's key object, which carries the request path.

    Args:
        route: The request path.

    Returns:
        A stand-in for LiteLLM's ``UserAPIKeyAuth``.
    """
    return SimpleNamespace(request_route=route)


def _data(model: str, headers: Mapping[str, str] | None = None, **body: object) -> dict[str, Any]:
    """Return proxy request data: the body plus LiteLLM's header copies.

    Args:
        model: The requested model.
        headers: Extra request headers (lowercase names).
        **body: Further body fields.

    Returns:
        The data LiteLLM hands its hooks.
    """
    request_headers = {"content-type": "application/json", "x-request-id": "req-0001", **(headers or {})}
    return {
        "model": model,
        **body,
        "proxy_server_request": {"url": "http://router/x", "method": "POST", "headers": request_headers},
        "metadata": {"headers": request_headers},
    }


def _deployment(deployment: str, revision: str | None = REVISION) -> dict[str, Any]:
    """Return the router's kwargs for a call to a deployment.

    Args:
        deployment: The deployment id from ``model_info``.
        revision: Its configured revision.

    Returns:
        The kwargs the deployment hooks receive.
    """
    info: dict[str, object] = {"id": deployment, "db_model": False}
    if revision is not None:
        info["revision"] = revision
    return {"metadata": {"model_info": info, "model_group": "example/chat-model"}}


def _chat_response(text: str = ANSWER) -> SimpleNamespace:
    """Return a chat completion response.

    Args:
        text: The answer.

    Returns:
        A stand-in for LiteLLM's ``ModelResponse``.
    """
    message = SimpleNamespace(role="assistant", content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=6, total_tokens=17),
        _hidden_params={"model_id": "chat"},
    )


def _chunk(content: str | None = None, finish: str | None = None, usage: object = None) -> SimpleNamespace:
    """Return one streamed chat chunk.

    Args:
        content: The text delta.
        finish: The finish reason.
        usage: The usage block of the final chunk.

    Returns:
        A stand-in for LiteLLM's ``ModelResponseStream``.
    """
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish)], usage=usage)


def _stream_chunks() -> list[SimpleNamespace]:
    """Return a complete chat stream: text, finish, usage.

    Returns:
        The chunks in order.
    """
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=6, total_tokens=17)
    return [
        _chunk("Synthetic answer"),
        _chunk(" for Alice"),
        _chunk(" Example."),
        _chunk(finish="stop"),
        SimpleNamespace(choices=[], usage=usage),
    ]


async def _iterate(chunks: list[Any]) -> AsyncIterator[Any]:
    """Yield chunks like a model's stream.

    Args:
        chunks: The chunks.

    Yields:
        Each chunk.
    """
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


def _b64(raw: bytes) -> str:
    """Encode unpadded base64url.

    Args:
        raw: The bytes.

    Returns:
        The encoded text.
    """
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _token(secret: str = SECRET, alg: str = "HS256", **overrides: object) -> str:
    """Mint an Open WebUI user token the way Open WebUI does.

    Args:
        secret: The signing secret.
        alg: The header's algorithm.
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
        "exp": int(NOW) + 300,
        **overrides,
    }
    header = _b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    body = _b64(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64(signature)}"


async def _chat_call(audit: Any, headers: Mapping[str, str] | None = None) -> None:
    """Run a non-streaming chat call through every hook LiteLLM calls.

    Args:
        audit: The callback.
        headers: Extra request headers.
    """
    data = _data("example/chat-model", headers, messages=[{"role": "user", "content": PROMPT}])
    await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
    kwargs = _deployment("chat")
    await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.acompletion)
    response = _chat_response()
    await audit.async_post_call_success_deployment_hook(kwargs, response, _CallTypes.acompletion)
    await audit.async_post_call_success_hook(data, _key("/v1/chat/completions"), response)
    await audit.async_post_call_response_headers_hook(data, _key("/v1/chat/completions"), response, {})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("acompletion", "completion"),
        ("aembedding", "embedding"),
        ("atext_completion", "text_completion"),
        ("rerank", "rerank"),
        ("transcription", "transcription"),
        ("anthropic_messages", "anthropic_messages"),
        ("pass_through_endpoint", "pass_through_endpoint"),
    ],
)
def test_call_type_loses_only_its_async_prefix(value: str, expected: str) -> None:
    """``acompletion`` and ``completion`` are one call type; ``anthropic_…`` keeps its ``a``."""
    assert router_audit.normalize_call_type(value) == expected


def test_call_type_setting_is_parsed_and_checked() -> None:
    """Blank means every type; a misspelt name stops the router."""
    assert router_audit.parse_call_types(None) is None
    assert router_audit.parse_call_types(" ") is None
    assert router_audit.parse_call_types("Completion, rerank,") == frozenset({"completion", "rerank"})
    with pytest.raises(ValueError, match="embeddings"):
        router_audit.parse_call_types("completion,embeddings")
    with pytest.raises(ValueError):
        _audit({"AUDIT_ROUTER_CALL_TYPES": "chat"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("v0.7.0", "0.7.0"), ("2026-11-03-abc1234", "2026-11-03-abc1234"), ("", "unknown"), (None, "unknown")],
)
def test_service_version_comes_from_the_image_tag(value: str | None, expected: str) -> None:
    """A release tag loses its ``v``; a dev build keeps its date and commit."""
    assert router_audit.service_version(value) == expected


def test_opening_writes_start_and_mode_once() -> None:
    """The chain opens with ``start`` and the operating mode, no secret in it."""
    audit, stream = _audit({"AUDIT_ROUTER_CALL_TYPES": "rerank,completion"})
    first = audit.open()
    assert audit.open() is first
    events = _events(stream)
    assert [(e["action"], e["operation"]) for e in events] == [("lifecycle", "start"), ("admin", "config.audit")]
    assert events[1]["criteria"] == {"call_types": ["completion", "rerank"], "openwebui_jwt": True}
    assert events[0]["service"] == "vllm-service"
    _assert_no_content(stream)
    _assert_chain(stream)


def test_chat_completion_is_one_event_with_provenance() -> None:
    """A chat call is recorded once, by reference, before its result returns."""
    audit, stream = _audit()
    _run(lambda: _chat_call(audit))
    (event,) = _ai_calls(stream)
    assert event["operation"] == "router.completion"
    assert event["outcome"] == "ok"
    assert event["actor"] == {"user": None, "via": "service"}
    assert event["request_id"] == "req-0001"
    assert event["recipient"] == "model:chat"
    assert event["ai"] == {
        "model_alias": "example/chat-model",
        "model_id": "chat",
        "revision": REVISION,
        "output_sha256": sha256_text(ANSWER),
        "input_tokens": 11,
        "output_tokens": 6,
    }
    assert event["criteria"] == {
        "attempts": 1,
        "duration_ms": 0,
        "route": "/v1/chat/completions",
        "status": 200,
        "stream": False,
    }
    assert "purpose" not in event
    assert "target" not in event
    _assert_no_content(stream)
    _assert_chain(stream)


def test_stream_is_recorded_before_its_final_chunk() -> None:
    """Text goes out at once; the finish and usage chunks wait for the event."""
    audit, stream = _audit()
    chunks = _stream_chunks()
    seen: list[tuple[SimpleNamespace, int]] = []

    async def scenario() -> None:
        data = _data("example/chat-model", stream=True, messages=[{"role": "user", "content": PROMPT}])
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        hooked = audit.async_post_call_streaming_iterator_hook(_key("/v1/chat/completions"), _iterate(chunks), data)
        async for chunk in hooked:
            seen.append((chunk, len(_ai_calls(stream))))

    _run(scenario)
    assert [chunk for chunk, _ in seen] == chunks
    assert [written for _, written in seen] == [0, 0, 0, 1, 1]
    (event,) = _ai_calls(stream)
    assert event["criteria"]["stream"] is True
    assert event["ai"]["output_sha256"] == sha256_text(ANSWER)
    assert (event["ai"]["input_tokens"], event["ai"]["output_tokens"]) == (11, 6)
    _assert_no_content(stream)
    _assert_chain(stream)


def test_embedding_records_its_batch_size_and_no_output() -> None:
    """Embeddings count their inputs; there is no text output to hash."""
    audit, stream = _audit()

    async def scenario() -> None:
        data = _data("example/embed-model", input=["Alice Example one", "two", "three"])
        await audit.async_pre_call_hook(_key("/v1/embeddings"), None, data, "aembedding")
        kwargs = _deployment("embed", revision=None)
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.aembedding)
        usage = SimpleNamespace(prompt_tokens=9, completion_tokens=0)
        response = SimpleNamespace(data=[], usage=usage, _hidden_params={})
        await audit.async_post_call_success_deployment_hook(kwargs, response, _CallTypes.aembedding)
        await audit.async_post_call_success_hook(data, _key("/v1/embeddings"), response)

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["operation"] == "router.embedding"
    assert event["criteria"]["batch"] == 3
    assert event["ai"] == {"model_alias": "example/embed-model", "model_id": "embed", "input_tokens": 9}
    assert event["recipient"] == "model:embed"
    _assert_no_content(stream)
    _assert_chain(stream)


def test_rerank_is_recorded_from_the_deployment_hook() -> None:
    """Rerank has no later success hook, so the deployment hook records it."""
    audit, stream = _audit()

    async def scenario() -> None:
        data = _data("example/rerank-model", query="Alice Example", documents=["one", "two"])
        await audit.async_pre_call_hook(_key("/v1/rerank"), None, data, "rerank")
        kwargs = _deployment("rerank", revision=None)
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.arerank)
        response = SimpleNamespace(results=[], _hidden_params={})
        await audit.async_post_call_success_deployment_hook(kwargs, response, _CallTypes.arerank)

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["operation"] == "router.rerank"
    assert event["criteria"]["documents"] == 2
    assert event["recipient"] == "model:rerank"
    _assert_no_content(stream)
    _assert_chain(stream)


def test_transcription_hashes_its_text() -> None:
    """A transcript's text is matched by hash, never stored."""
    audit, stream = _audit()
    transcript = "Synthetic transcript of a test tone."

    async def scenario() -> None:
        data = _data("openai/whisper-large-v3")
        await audit.async_pre_call_hook(_key("/v1/audio/transcriptions"), None, data, "transcription")
        kwargs = _deployment("asr", revision=None)
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.atranscription)
        response = SimpleNamespace(text=transcript, usage=None, _hidden_params={})
        await audit.async_post_call_success_deployment_hook(kwargs, response, _CallTypes.atranscription)
        await audit.async_post_call_response_headers_hook(data, _key("/v1/audio/transcriptions"), response, {})

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["operation"] == "router.transcription"
    assert event["ai"]["output_sha256"] == sha256_text(transcript)
    assert transcript not in stream.getvalue()
    _assert_chain(stream)


def test_pass_through_is_recorded_once_its_backend_answered() -> None:
    """A pass-through route names its backend; its request id comes late."""
    audit, stream = _audit()

    async def scenario() -> None:
        body: dict[str, Any] = {"text": "Alice Example", "labels": ["person"]}
        await audit.async_pre_call_hook(_key("/gliner"), None, body, "pass_through_endpoint")
        headers = {"authorization": "Bearer sk-synthetic", "x-request-id": "req-0042"}
        await audit.async_post_call_response_headers_hook(
            body, _key("/gliner"), SimpleNamespace(status_code=200), headers
        )

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["operation"] == "router.pass_through_endpoint"
    assert event["request_id"] == "req-0042"
    assert event["recipient"] == "model:gliner"
    assert event["ai"] == {"model_alias": "/gliner", "model_id": "gliner"}
    assert event["criteria"] == {"duration_ms": 0, "route": "/gliner", "status": 200}
    _assert_no_content(stream)
    _assert_chain(stream)


def test_pass_through_upstream_error_is_one_error_event() -> None:
    """LiteLLM reports a backend's error status first; the headers hook adds nothing."""
    audit, stream = _audit()

    async def scenario() -> None:
        body: dict[str, Any] = {"input": ["Alice Example"]}
        await audit.async_pre_call_hook(_key("/pooling"), None, body, "pass_through_endpoint")
        # LiteLLM adds the request's own Starlette Headers once it has read the body.
        recorded = {"proxy_server_request": {"headers": Headers({"x-request-id": "req-0043"})}}
        failure = {**body, "litellm_params": recorded}
        await audit.async_post_call_failure_hook(failure, _BackendError("synthetic"), _key("/pooling"))
        await audit.async_post_call_response_headers_hook(body, _key("/pooling"), SimpleNamespace(status_code=500), {})

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["outcome"] == "error"
    assert event["recipient"] == "model:embed-sparse"
    assert event["request_id"] == "req-0043"
    assert event["criteria"]["status"] == 500
    assert event["criteria"]["error"] == "_BackendError"
    _assert_chain(stream)


def test_failed_call_counts_every_attempt() -> None:
    """The router's retries each sent the data again; the error event says how often."""
    clock = _Clock()
    audit, stream = _audit(clock=clock)

    async def scenario() -> None:
        data = _data("example/chat-model", messages=[{"role": "user", "content": PROMPT}])
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        for _ in range(3):
            await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        clock.now = 1.5
        await audit.async_post_call_failure_hook(data, _BackendError("synthetic"), _key("/v1/chat/completions"))

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["outcome"] == "error"
    assert event["criteria"]["attempts"] == 3
    assert event["criteria"]["duration_ms"] == 1500
    assert event["criteria"]["status"] == 500
    assert "output_sha256" not in event["ai"]
    _assert_chain(stream)


def test_calls_that_send_nothing_are_not_audited() -> None:
    """A refused key, an unknown model, a health check: no client data reached a model."""
    audit, stream = _audit()

    async def refused_key() -> None:
        await audit.async_post_call_failure_hook({}, _BackendError("bad key"), _key("/v1/chat/completions"))

    async def unknown_model() -> None:
        data = _data("example/no-such-model", messages=[{"role": "user", "content": PROMPT}])
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_post_call_failure_hook(data, _BackendError("no deployment"), _key("/v1/chat/completions"))

    async def health_check() -> None:
        kwargs = _deployment("chat")
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.acompletion)
        await audit.async_post_call_success_deployment_hook(kwargs, _chat_response(), _CallTypes.acompletion)

    _run(refused_key)
    _run(unknown_model)
    _run(health_check)
    assert _ai_calls(stream) == []
    _assert_chain(stream)


def test_failed_write_fails_the_call_and_closes_the_gate() -> None:
    """No result without its event; later calls wait until the gap is recorded."""
    audit, stream = _audit()
    audit.open()
    raised: list[Any] = []

    async def lost_call() -> None:
        data = _data("example/chat-model", messages=[{"role": "user", "content": PROMPT}])
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        stream.broken = True
        try:
            await audit.async_post_call_success_hook(data, _key("/v1/chat/completions"), _chat_response())
        except router_audit.AuditUnavailable as exc:
            raised.append(exc)
            # LiteLLM then runs the failure hook with that exception.
            await audit.async_post_call_failure_hook(data, exc, _key("/v1/chat/completions"))

    async def refused_call() -> None:
        data = _data("example/chat-model")
        with pytest.raises(router_audit.AuditUnavailable) as caught:
            await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        assert caught.value.refused
        await audit.async_post_call_failure_hook(data, caught.value, _key("/v1/chat/completions"))

    _run(lost_call)
    (exc,) = raised
    assert not exc.refused
    assert (exc.status_code, exc.message) == (503, "audit log unavailable")
    _run(refused_call)
    assert _ai_calls(stream) == []
    stream.broken = False
    _run(lambda: _chat_call(audit))
    events = _events(stream)
    gap = [e for e in events if e["operation"] == "audit.recovered"]
    assert [e["criteria"] for e in gap] == [{"unwritten_events": 1}]
    assert [e["operation"] for e in events[-2:]] == ["audit.recovered", "router.completion"]
    _assert_chain(stream)


def test_failed_write_on_the_deployment_hook_stops_the_retry() -> None:
    """Rerank fails closed in the router; a retry is refused before it is sent."""
    audit, stream = _audit()
    audit.open()

    async def scenario() -> None:
        data = _data("example/rerank-model", documents=["one"])
        await audit.async_pre_call_hook(_key("/v1/rerank"), None, data, "rerank")
        kwargs = _deployment("rerank", revision=None)
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.arerank)
        stream.broken = True
        with pytest.raises(router_audit.AuditUnavailable):
            await audit.async_post_call_success_deployment_hook(kwargs, SimpleNamespace(), _CallTypes.arerank)
        with pytest.raises(router_audit.AuditUnavailable) as retry:
            await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.arerank)
        assert retry.value.refused
        await audit.async_post_call_failure_hook(data, retry.value, _key("/v1/rerank"))

    _run(scenario)
    assert _ai_calls(stream) == []
    stream.broken = False
    _run(lambda: _chat_call(audit))
    assert [e["criteria"] for e in _events(stream) if e["operation"] == "audit.recovered"] == [{"unwritten_events": 1}]
    assert [e["operation"] for e in _ai_calls(stream)] == ["router.completion"]
    _assert_chain(stream)


def test_refused_retry_after_the_data_went_out_is_counted() -> None:
    """A retry the gate stops still leaves a sent attempt to account for."""
    audit, stream = _audit()
    audit.open()

    async def other_call_loses_its_event() -> None:
        data = _data("example/chat-model")
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        stream.broken = True
        with pytest.raises(router_audit.AuditUnavailable):
            await audit.async_post_call_success_hook(data, _key("/v1/chat/completions"), _chat_response())

    async def scenario() -> None:
        data = _data("example/rerank-model", documents=["one"])
        await audit.async_pre_call_hook(_key("/v1/rerank"), None, data, "rerank")
        kwargs = _deployment("rerank", revision=None)
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.arerank)
        # The first attempt fails at the backend; meanwhile another request's write fails.
        await asyncio.create_task(other_call_loses_its_event())
        with pytest.raises(router_audit.AuditUnavailable) as retry:
            await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.arerank)
        await audit.async_post_call_failure_hook(data, retry.value, _key("/v1/rerank"))

    _run(scenario)
    stream.broken = False
    _run(lambda: _chat_call(audit))
    assert [e["criteria"] for e in _events(stream) if e["operation"] == "audit.recovered"] == [{"unwritten_events": 2}]
    _assert_chain(stream)


def test_failed_write_withholds_a_streams_final_chunk() -> None:
    """The stream ends with an error instead of its finish and usage chunks."""
    audit, stream = _audit()
    audit.open()
    chunks = _stream_chunks()
    seen: list[SimpleNamespace] = []

    async def scenario() -> None:
        data = _data("example/chat-model", stream=True)
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        stream.broken = True
        hooked = audit.async_post_call_streaming_iterator_hook(_key("/v1/chat/completions"), _iterate(chunks), data)
        with pytest.raises(router_audit.AuditUnavailable):
            async for chunk in hooked:
                seen.append(chunk)

    _run(scenario)
    stream.broken = False
    assert seen == chunks[:3]
    _assert_chain(stream)


def test_failed_pass_through_write_closes_the_gate() -> None:
    """LiteLLM ignores the headers hook's errors, so the next call is refused."""
    audit, stream = _audit()
    audit.open()

    async def lost_call() -> None:
        body: dict[str, Any] = {"text": "Alice Example"}
        await audit.async_pre_call_hook(_key("/gliner"), None, body, "pass_through_endpoint")
        stream.broken = True
        await audit.async_post_call_response_headers_hook(body, _key("/gliner"), SimpleNamespace(status_code=200), {})

    async def next_call() -> None:
        await audit.async_pre_call_hook(_key("/gliner"), None, {}, "pass_through_endpoint")

    _run(lost_call)
    with pytest.raises(router_audit.AuditUnavailable):
        _run(next_call)
    stream.broken = False
    _run(next_call)
    assert [e["criteria"] for e in _events(stream) if e["operation"] == "audit.recovered"] == [{"unwritten_events": 1}]
    assert _ai_calls(stream) == []
    _assert_chain(stream)


def test_client_disconnect_mid_stream_is_an_error_event() -> None:
    """The data was sent; the event says the client went away."""
    audit, stream = _audit()

    async def scenario() -> None:
        data = _data("example/chat-model", stream=True)
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        hooked = audit.async_post_call_streaming_iterator_hook(
            _key("/v1/chat/completions"), _iterate(_stream_chunks()), data
        )
        await hooked.__anext__()
        await hooked.aclose()

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["outcome"] == "error"
    assert event["criteria"]["error"] == "client_disconnected"
    assert "status" not in event["criteria"]
    _assert_chain(stream)


def test_signed_openwebui_token_names_the_user() -> None:
    """Open WebUI's signed token identifies its user, with the caller noted."""
    audit, stream = _audit()
    headers = {"x-openwebui-user-jwt": _token(), "x-openwebui-chat-id": "0b7e6a2c-0000-4000-8000-000000000001"}
    _run(lambda: _chat_call(audit, headers))
    (event,) = _ai_calls(stream)
    assert event["actor"] == {"user": "alice@example.org", "via": "service"}
    assert event["criteria"]["caller"] == "open-webui"
    assert event["criteria"]["forwarded_identity"] == "signed"
    assert event["criteria"]["chat_id"] == "0b7e6a2c-0000-4000-8000-000000000001"
    _assert_chain(stream)


@pytest.mark.parametrize(
    "token",
    [
        _token(secret="another-secret"),
        _token(exp=int(NOW) - 60),
        _token(iat=int(NOW) + 600),
        _token(iss="someone-else"),
        _token(alg="none"),
        _token(email=""),
        "not.a.token",
        "x" * 5000,
    ],
)
def test_invalid_openwebui_token_names_nobody(token: str) -> None:
    """A token that does not verify leaves the actor unnamed."""
    audit, stream = _audit()
    _run(lambda: _chat_call(audit, {"x-openwebui-user-jwt": token}))
    (event,) = _ai_calls(stream)
    assert event["actor"] == {"user": None, "via": "service"}
    assert event["criteria"]["forwarded_identity"] == "invalid"
    assert "caller" not in event["criteria"]
    _assert_chain(stream)


def test_token_without_a_shared_secret_names_nobody() -> None:
    """Without the secret the router cannot check a token, so it trusts none."""
    audit, stream = _audit({"AUDIT_OPENWEBUI_JWT_SECRET": ""})
    _run(lambda: _chat_call(audit, {"x-openwebui-user-jwt": _token()}))
    (event,) = _ai_calls(stream)
    assert event["actor"]["user"] is None
    assert event["criteria"]["forwarded_identity"] == "invalid"
    _assert_chain(stream)


def test_unsigned_openwebui_headers_name_nobody() -> None:
    """Any master-key holder could send these, so they are noted, not trusted."""
    audit, stream = _audit()
    _run(lambda: _chat_call(audit, {"x-openwebui-user-email": "mallory@example.org"}))
    (event,) = _ai_calls(stream)
    assert event["actor"]["user"] is None
    assert event["criteria"]["forwarded_identity"] == "unsigned"
    assert "mallory" not in stream.getvalue()
    _assert_chain(stream)


def test_unlisted_call_types_are_neither_audited_nor_gated() -> None:
    """``AUDIT_ROUTER_CALL_TYPES`` drops a type from the log and from the gate."""
    audit, stream = _audit({"AUDIT_ROUTER_CALL_TYPES": "completion"})
    audit.open()
    stream.broken = True

    async def scenario() -> None:
        data = _data("example/embed-model", input=["one"])
        await audit.async_pre_call_hook(_key("/v1/embeddings"), None, data, "aembedding")
        kwargs = _deployment("embed", revision=None)
        await audit.async_pre_call_deployment_hook(kwargs, _CallTypes.aembedding)
        await audit.async_post_call_success_hook(data, _key("/v1/embeddings"), SimpleNamespace())

    _run(scenario)
    stream.broken = False
    assert _ai_calls(stream) == []
    _assert_chain(stream)


def test_another_process_opens_its_own_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process never writes to a chain another process opened."""
    audit, stream = _audit()
    first = audit.open()
    monkeypatch.setattr(router_audit.os, "getpid", lambda: 4242)
    second = audit.open()
    assert second is not first
    audit.forget_inherited()
    third = audit.open()
    assert len({first.instance, second.instance, third.instance}) == 3
    assert len([e for e in _events(stream) if e["operation"] == "start"]) == 3
    _assert_chain(stream)


_HS256 = '{"alg":"HS256","typ":"JWT"}'
_NESTED = "[" * 1400 + "]" * 1400


def _signed(header: str, body: str, secret: str = SECRET) -> str:
    """Sign arbitrary token parts with HS256.

    Args:
        header: The header JSON.
        body: The claims JSON, or anything else.
        secret: The signing secret.

    Returns:
        The token.
    """
    head, payload = _b64(header.encode()), _b64(body.encode())
    signature = hmac.new(secret.encode(), f"{head}.{payload}".encode(), hashlib.sha256).digest()
    return f"{head}.{payload}.{_b64(signature)}"


class _Unreadable:
    """An object every attribute read of which fails."""

    def __getattr__(self, name: str) -> object:
        """Fail like a broken response object.

        Args:
            name: The attribute.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(f"cannot read {name}")


class _HostileHeaders(Mapping[str, str]):
    """Request headers that fail when read."""

    @override
    def __getitem__(self, key: str) -> str:
        """Fail like a broken header object.

        Args:
            key: The header name.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("cannot read headers")

    @override
    def __iter__(self) -> Iterator[str]:
        """Iterate over no names.

        Returns:
            An empty iterator.
        """
        return iter(())

    @override
    def __len__(self) -> int:
        """Count no headers.

        Returns:
            Zero.
        """
        return 0


def test_pass_through_body_cannot_supply_headers() -> None:
    """A pass-through payload is the client's body: header-shaped fields in it change nothing."""
    audit, stream = _audit()
    planted = {"x-request-id": "req-spoofed", "x-openwebui-user-jwt": _token(), "x-openwebui-user-email": "x@x"}

    async def scenario() -> None:
        body: dict[str, Any] = {
            "text": "Alice Example",
            "proxy_server_request": {"headers": planted},
            "metadata": {"headers": planted},
            "litellm_metadata": {"headers": planted},
            "litellm_params": {"proxy_server_request": {"headers": planted}},
        }
        await audit.async_pre_call_hook(_key("/gliner"), None, body, "pass_through_endpoint")
        real = {"x-request-id": "req-0050"}
        await audit.async_post_call_response_headers_hook(body, _key("/gliner"), SimpleNamespace(status_code=200), real)

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["request_id"] == "req-0050"
    assert event["actor"] == {"user": None, "via": "service"}
    assert event["criteria"] == {"duration_ms": 0, "route": "/gliner", "status": 200}
    assert "spoofed" not in stream.getvalue()
    _assert_chain(stream)


def test_failed_pass_through_takes_headers_only_from_litellm() -> None:
    """A failure before LiteLLM recorded the request's headers leaves them out."""
    audit, stream = _audit()
    planted = {"x-request-id": "req-spoofed", "x-openwebui-user-jwt": _token()}

    async def scenario() -> None:
        body: dict[str, Any] = {"text": "Alice", "litellm_params": {"proxy_server_request": {"headers": planted}}}
        await audit.async_pre_call_hook(_key("/gliner"), None, body, "pass_through_endpoint")
        await audit.async_post_call_failure_hook(body, _BackendError("connection refused"), _key("/gliner"))

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["outcome"] == "error"
    assert "request_id" not in event
    assert event["actor"]["user"] is None
    assert "forwarded_identity" not in event["criteria"]
    _assert_chain(stream)


def test_model_list_headers_come_only_from_the_proxys_record() -> None:
    """Headers under the request's metadata are the client's, not the proxy's."""
    audit, stream = _audit()

    async def scenario() -> None:
        data: dict[str, Any] = {"model": "example/chat-model", "metadata": {"headers": {"x-request-id": "req-spoofed"}}}
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        await audit.async_post_call_success_hook(data, _key("/v1/chat/completions"), _chat_response())

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert "request_id" not in event
    _assert_chain(stream)


def test_a_tokens_content_is_parsed_only_after_its_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsigned input never reaches the JSON decoder."""

    def no_parsing(*args: object, **kwargs: object) -> object:
        raise AssertionError("parsed before the signature was checked")

    monkeypatch.setattr(router_audit.json, "loads", no_parsing)
    for token in (_token(secret="another-secret"), _signed(_HS256, _NESTED, secret="another-secret"), "a.b.c", "a.b."):
        assert router_audit.verify_openwebui_token(token, SECRET.encode(), now=NOW) is None


def test_a_token_that_breaks_the_decoder_is_simply_not_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a correctly signed token cannot make verification raise."""
    assert router_audit.verify_openwebui_token(_signed(_HS256, _NESTED), SECRET.encode(), now=NOW) is None

    def exhausted(*args: object, **kwargs: object) -> object:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(router_audit.json, "loads", exhausted)
    assert router_audit.verify_openwebui_token(_token(), SECRET.encode(), now=NOW) is None


def test_a_hostile_token_still_leaves_a_record() -> None:
    """The call is recorded; the token only fails to name a user."""
    audit, stream = _audit()
    _run(lambda: _chat_call(audit, {"x-openwebui-user-jwt": _signed(_HS256, _NESTED)}))
    (event,) = _ai_calls(stream)
    assert event["actor"]["user"] is None
    assert event["criteria"]["forwarded_identity"] == "invalid"
    _assert_chain(stream)


def test_unreadable_facts_cost_facts_never_the_event() -> None:
    """Headers, deployment details and a response that cannot be read leave the event incomplete."""
    audit, stream = _audit()

    async def scenario() -> None:
        data = _data("example/chat-model")
        data["proxy_server_request"]["headers"] = _HostileHeaders()
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook({"metadata": _HostileHeaders()}, _CallTypes.acompletion)
        await audit.async_post_call_success_hook(data, _key("/v1/chat/completions"), _Unreadable())

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["outcome"] == "ok"
    assert event["criteria"]["incomplete"] is True
    assert event["criteria"]["attempts"] == 1
    assert "request_id" not in event
    assert event["ai"] == {"model_alias": "example/chat-model"}
    assert event["recipient"] == "model:unknown"
    _assert_chain(stream)


def test_unreadable_pass_through_details_still_leave_a_record() -> None:
    """A config or backend response that cannot be read still leaves the call's event."""

    def broken_targets() -> Mapping[str, str]:
        raise RuntimeError("pass-through endpoints not loaded")

    audit, stream = _audit(targets=broken_targets)

    async def scenario() -> None:
        await audit.async_pre_call_hook(_key("/gliner"), None, {}, "pass_through_endpoint")
        headers = {"x-request-id": "req-0051"}
        await audit.async_post_call_response_headers_hook({}, _key("/gliner"), _Unreadable(), headers)

    _run(scenario)
    (event,) = _ai_calls(stream)
    assert event["request_id"] == "req-0051"
    assert event["recipient"] == "model:unknown"
    assert event["criteria"]["incomplete"] is True
    assert "status" not in event["criteria"]
    _assert_chain(stream)


def test_an_unreadable_chunk_still_goes_out() -> None:
    """A chunk the tally cannot read is passed on; the stream is still recorded."""
    audit, stream = _audit()
    chunks: list[Any] = [_chunk("Synthetic"), _Unreadable(), _chunk(finish="stop")]
    seen: list[Any] = []

    async def scenario() -> None:
        data = _data("example/chat-model", stream=True)
        await audit.async_pre_call_hook(_key("/v1/chat/completions"), None, data, "acompletion")
        await audit.async_pre_call_deployment_hook(_deployment("chat"), _CallTypes.acompletion)
        hooked = audit.async_post_call_streaming_iterator_hook(_key("/v1/chat/completions"), _iterate(chunks), data)
        async for chunk in hooked:
            seen.append(chunk)

    _run(scenario)
    assert seen == chunks
    (event,) = _ai_calls(stream)
    assert event["criteria"]["incomplete"] is True
    assert "output_sha256" not in event["ai"]
    _assert_chain(stream)
