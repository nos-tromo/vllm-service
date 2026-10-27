"""Topic-model fit server (BERTopic over caller-supplied vectors).

Wraps ``topics_fit.py`` in a small FastAPI app so a consumer (docint) can fit a topic model to a
sample of embedding vectors it already holds. Nothing is embedded, downloaded or stored: the caller
posts the vectors and the matching texts, and gets one topic label per input and a keyword list
per topic.

Endpoints:

    POST /topics/fit
        Body ``multipart/form-data``, every part sent as a file part:

        - ``meta`` (``meta.json``): ``{"n", "dim", "seed", "min_topic_size", "nr_topics",
          "top_n_words", "min_df", "stop_words"}``, all required. ``nr_topics`` is 0 (no
          reduction) or at least 2.
        - ``texts`` (``texts.json``): a JSON array of ``n`` strings, the c-TF-IDF input.
        - ``vectors`` (``vectors.f32``): ``n x dim`` little-endian float32, row-major.

        Returns ``{"labels", "outliers", "topics", "versions", "runtime", "timings"}``:
        ``labels[i]`` is input ``i``'s topic (``-1`` marks an outlier), and ``topics`` lists the
        ids ``>= 0`` in ascending order with their size and ``[word, score]`` keywords.

        Errors: 400 for a malformed or out-of-range ``meta`` (``dim`` above 8192 and a stop word
        over 64 characters included) or a multipart body without a boundary; 408, closing the
        connection, when an admitted fit's body has not arrived within 300 s; 413 when the body
        is larger than any fit needs (below), ``n`` exceeds ``TOPICS_MAX_POINTS``, ``n * dim``
        exceeds ``TOPICS_MAX_POINTS * 1024``, ``meta`` is over 1 MiB, or ``texts`` is over
        ``min(n * 16384 + 65536, 512 MiB)`` bytes; 422 for a missing part, a ``texts`` part that
        is not a JSON array of strings, a text count other than ``n``, a vector byte count other
        than ``n * dim * 4``, a NaN or infinity, or fewer points than the fit needs; 503 with
        ``Retry-After``, before any of the body is read, while another fit is being received or
        run (one at a time, never queued); 500 when the fit itself fails.

    GET /health
        Liveness probe; returns ``{"status": "ok", "versions": {...}}``. The server only starts
        answering once the startup self-check has passed.

Configuration:

    TOPICS_MAX_POINTS   Largest accepted ``n``. Default 150000. It also caps ``n * dim`` at 1024
                        times the limit, so a wider vector means fewer points, not more memory,
                        and sets the largest body a request may send (1,152,385,024 bytes at the
                        default). Must be a positive integer. It does not bound memory on its
                        own: at 1024 dimensions with texts of about 2,000 characters a fit
                        measured 2.3 GB at 50k points and 3.9 GB at 100k, about 5.4 GB at 150k
                        extrapolated, and more points, wider vectors and longer texts raise it.
                        Compose's ``mem_limit`` (``TOPICS_MEM_LIMIT``, default 7g) contains a
                        request that outgrows it.
    TOPICS_WARMUP       Run ``topics_fit.self_check()`` at import, which compiles the numba code
                        and fits a synthetic set twice. On unless set to false, 0, no or off.

Uploads are bounded before they are parsed, because FastAPI reads a multipart body in full,
spooling each part to a temporary file (on disk beyond 1 MiB), before the handler runs.
``_UploadGuard`` answers first: a declared length over the largest body a fit needs is a 413, and a
fit that arrives while another holds the one fit slot is a 503, both before a byte of the body is
read; what such a client sends anyway is read and dropped, never stored. An admitted fit holds the
slot from its first byte until its response has been sent, and its body is counted as it arrives,
chunked or not, and cut off with a 413 past that size or a 408 past the deadline, so at most one
body is spooled at a time and a stalled client cannot keep the slot. The slot is a lock in this
process, so these bounds hold per uvicorn worker; the image pins uvicorn to one (``--workers 1``),
so a ``WEB_CONCURRENCY`` or ``UVICORN_WORKERS`` from the shared ``.env`` cannot start a second.

The handler then reads a part into memory only once the checks before it have passed, and those
judge sizes alone: ``meta``'s size, then ``n`` and ``n * dim`` against the limits and the size of
``texts``, so that every 413 comes before any 422, then ``n`` against the fit's minimum and the
size of ``vectors``, and only then the reads (``vectors``, then ``texts``).

The server logs counts and timings only, never texts, vectors or keywords.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from typing import Annotated

import anyio
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import topics_fit

_WARMUP_OFF = frozenset({"false", "0", "no", "off"})

# Request-size limits. The handler judges each part against its own before reading it; the guard
# holds a whole body to their sum.
_META_MAX_BYTES = 1024 * 1024
_TEXTS_BYTES_PER_POINT = 16_384
_TEXTS_BYTES_SLACK = 65_536
_TEXTS_MAX_BYTES = 512 * 1024 * 1024
_VALUES_PER_POINT = 1024
# Room for the multipart boundaries and part headers around the three parts.
_FRAMING_SLACK_BYTES = 65_536
# An admitted fit's whole body must arrive within this many seconds.
_BODY_TIMEOUT_S = 300.0

_FIT_PATH = "/topics/fit"
# Marks the scope of the request that holds the fit slot.
_ADMITTED_KEY = "topics.fit_admitted"

# uvicorn puts its handler on this logger; a logger of our own would drop the INFO line.
_log = logging.getLogger("uvicorn.error")


def _positive_int_env(name: str, default: int) -> int:
    """Read an integer env var that must be at least 1.

    A limit below 1 would refuse every fit, which would only show up as a stream of 413s, so a
    bad value stops the server at startup instead.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        The parsed value.

    Raises:
        ValueError: When the value is not an integer, or is below 1.
    """
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


MAX_POINTS: int = _positive_int_env("TOPICS_MAX_POINTS", 150_000)

if os.environ.get("TOPICS_WARMUP", "true").strip().lower() not in _WARMUP_OFF:
    topics_fit.self_check()

app = FastAPI(title="vllm-service topics", version="1.0")

# The one fit slot, held by the guard from the first byte of a fit's body until its response is sent.
_fit_lock = threading.Lock()


def _max_request_bytes() -> int:
    """Return the largest request body a fit can need.

    That is ``meta``'s limit, plus the most ``texts`` may hold at ``TOPICS_MAX_POINTS`` points, plus
    the largest ``vectors`` part the ``n * dim`` cap allows, plus the multipart framing:
    1,152,385,024 bytes at the default. It is worked out on each request, from the current limits.

    Returns:
        The size in bytes.
    """
    texts = min(MAX_POINTS * _TEXTS_BYTES_PER_POINT + _TEXTS_BYTES_SLACK, _TEXTS_MAX_BYTES)
    vectors = MAX_POINTS * _VALUES_PER_POINT * 4
    return _META_MAX_BYTES + texts + vectors + _FRAMING_SLACK_BYTES


def _content_length(scope: Scope) -> int | None:
    """Return the declared ``Content-Length``, or ``None`` when it is absent or not plain digits.

    ``int()`` would also take ``+5``, ``1_0`` or padding; such a value only skips the early
    refusal, since the body is counted as it arrives either way.

    Args:
        scope: The request's ASGI scope.

    Returns:
        The declared length, if it is a plain decimal number.
    """
    for name, value in scope["headers"]:
        if name == b"content-length":
            return int(value) if value.isdigit() else None
    return None


def _route_path(scope: Scope) -> str:
    """Return the path the routes are matched on: ``scope["path"]`` without ``root_path``.

    This is Starlette's own rule, so the guard and the routes agree on which request is the fit.

    Args:
        scope: The request's ASGI scope.

    Returns:
        The path the router matches.
    """
    path: str = scope["path"]
    root_path: str = scope.get("root_path", "")
    if not root_path or not path.startswith(root_path):
        return path
    if path == root_path:
        return ""
    return path[len(root_path) :] if path[len(root_path)] == "/" else path


def _bounded_receive(receive: Receive, limit: int, deadline: float | None) -> Receive:
    """Wrap ``receive`` so the body is counted against ``limit`` and, for a fit, held to ``deadline``.

    Every body byte counts, whatever length the request declared: h11 frames a body that carries
    both ``Content-Length`` and ``Transfer-Encoding`` as chunked. The refusals are raised as
    ``HTTPException`` from inside FastAPI's form parsing, which passes that type on (anything else
    becomes a 400) once Starlette has closed the parts it spooled. After a refusal the request
    reads as disconnected, so nothing more of the body is pulled.

    Args:
        receive: The server's ``receive``.
        limit: The most body bytes the request may send.
        deadline: When the whole body must have arrived, on ``anyio.current_time()``'s clock, or
            ``None`` for no deadline.

    Returns:
        The wrapped ``receive``.
    """
    received = 0
    complete = False
    refused = False

    async def bounded() -> Message:
        nonlocal received, complete, refused
        if refused:
            return {"type": "http.disconnect"}
        if complete or deadline is None:
            message = await receive()
        else:
            arrived: Message | None = None
            remaining = deadline - anyio.current_time()
            if remaining > 0:
                with anyio.move_on_after(remaining):
                    arrived = await receive()
            if arrived is None:
                refused = True
                raise HTTPException(
                    status_code=408,
                    detail=f"the request body did not arrive within {_BODY_TIMEOUT_S:g} s",
                    headers={"Connection": "close"},
                )
            message = arrived
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                refused = True
                raise HTTPException(status_code=413, detail=f"the request body is over the limit of {limit} bytes")
            complete = not message.get("more_body", False)
        return message

    return bounded


class _UploadGuard:
    """Admit one fit at a time and bound every request body, before any of the body is read.

    FastAPI parses a multipart body in full before the handler runs, so the handler's checks come
    too late to spare the upload. This middleware runs first, in this order:

    1. A declared ``Content-Length`` over :func:`_max_request_bytes` is answered with a 413.
    2. A ``POST /topics/fit`` takes the fit slot, or is answered with a 503 and ``Retry-After``
       while a fit holds it. The admitted request's scope is marked with ``_ADMITTED_KEY``.
    3. The body is counted as it arrives (:func:`_bounded_receive`): a 413 past the limit and, for
       an admitted fit, a 408 once ``_BODY_TIMEOUT_S`` has passed since admission.

    Steps 1 and 2 answer without calling ``receive()``, so an ``Expect: 100-continue`` client is
    never asked for its body, and without ``Connection: close``, so uvicorn reads and drops what
    the client sends anyway instead of resetting a connection its answer may still be on. The slot
    is released once the response has been sent and the fit's thread has ended.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the application.

        Args:
            app: The application behind the guard.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI call.

        Args:
            scope: The connection scope.
            receive: The server's ``receive``.
            send: The server's ``send``.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = _max_request_bytes()
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            detail = f"the request body of {declared} bytes is over the limit of {limit}"
            await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
            return
        if scope["method"] != "POST" or _route_path(scope) != _FIT_PATH:
            await self.app(scope, _bounded_receive(receive, limit, None), send)
            return
        if not _fit_lock.acquire(blocking=False):
            busy = {"detail": "a topic-model fit is already running"}
            await JSONResponse(busy, status_code=503, headers={"Retry-After": "30"})(scope, receive, send)
            return
        try:
            scope[_ADMITTED_KEY] = True
            deadline = anyio.current_time() + _BODY_TIMEOUT_S
            await self.app(scope, _bounded_receive(receive, limit, deadline), send)
        finally:
            _fit_lock.release()


app.add_middleware(_UploadGuard)


class FitMeta(BaseModel):
    """The ``meta`` part: the shape of the other two parts, and the fit's settings.

    Attributes:
        n: Number of points: the length of ``texts`` and the row count of ``vectors``.
        dim: Vector dimension, at most 8192.
        seed: UMAP's random state, within the range numpy accepts as a seed.
        min_topic_size: HDBSCAN's ``min_cluster_size``, of which 2 is the least it accepts.
        nr_topics: Topic-reduction target, or 0 for none. BERTopic counts the outlier class as
            one of these, so 1 is refused.
        top_n_words: Keywords kept per topic.
        min_df: The fewest texts a word must occur in to become a keyword.
        stop_words: Words never used as keywords: at most 10000, each of at most 64 characters.
    """

    n: int = Field(ge=1)
    dim: int = Field(ge=1, le=8192)
    seed: int = Field(ge=0, lt=2**32)
    min_topic_size: int = Field(ge=2)
    nr_topics: int = Field(ge=0)
    top_n_words: int = Field(ge=1, le=30)
    min_df: int = Field(ge=1)
    stop_words: list[Annotated[str, Field(max_length=64)]] = Field(max_length=10_000)

    @field_validator("nr_topics")
    @classmethod
    def _reject_single_topic(cls, value: int) -> int:
        """Refuse ``nr_topics == 1``, which crashes BERTopic whenever the fit has outliers."""
        if value == 1:
            raise ValueError("nr_topics must be 0 (no reduction) or at least 2: the outlier class counts as a topic")
        return value


class Topic(BaseModel):
    """One topic of the response."""

    id: int
    size: int
    keywords: list[tuple[str, float]]


class FitResponse(BaseModel):
    """The response of ``POST /topics/fit``."""

    labels: list[int]
    outliers: int
    topics: list[Topic]
    versions: dict[str, str]
    runtime: dict[str, object]
    timings: dict[str, float]


def _require_size_at_most(part: UploadFile, limit: int, name: str) -> None:
    """Refuse a part larger than ``limit`` bytes, judged by its size on disk before it is read.

    Args:
        part: The uploaded part.
        limit: The most bytes it may hold.
        name: The part's name, for the error message.

    Raises:
        HTTPException: 413, if the part is larger than ``limit`` or its size is unknown.
    """
    if part.size is None or part.size > limit:
        raise HTTPException(status_code=413, detail=f"{name} holds {part.size} bytes, over the limit of {limit}")


def _parse_meta(part: UploadFile) -> FitMeta:
    """Parse and validate the ``meta`` part, which may hold at most 1 MiB.

    Args:
        part: The uploaded ``meta.json``.

    Returns:
        The validated meta.

    Raises:
        HTTPException: 413, if the part is over 1 MiB; 400, if it is not JSON, not an object,
            or has a missing or out-of-range field.
    """
    _require_size_at_most(part, _META_MAX_BYTES, "meta")
    try:
        return FitMeta.model_validate_json(part.file.read())
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, error['loc'])) or 'meta'}: {error['msg']}" for error in exc.errors())
        raise HTTPException(status_code=400, detail=f"invalid meta.json: {problems}") from None


def _read_vectors(part: UploadFile, meta: FitMeta) -> np.ndarray:
    """Read the ``vectors`` part as an ``n x dim`` float32 array.

    The byte count is checked against the part's size on disk, before anything is read, so a
    part of the wrong size is refused without being loaded.

    Args:
        part: The uploaded ``vectors.f32``.
        meta: The validated meta.

    Returns:
        The vectors, a read-only view of the bytes read.

    Raises:
        HTTPException: 422, if the byte count is not ``n * dim * 4`` or a value is not finite.
    """
    expected = meta.n * meta.dim * 4
    if part.size != expected:
        raise HTTPException(status_code=422, detail=f"vectors holds {part.size} bytes, expected n*dim*4 = {expected}")
    vectors = np.frombuffer(part.file.read(), dtype="<f4").reshape(meta.n, meta.dim)
    if not np.isfinite(vectors).all():
        raise HTTPException(status_code=422, detail="vectors must be finite: found NaN or infinity")
    return vectors


def _read_texts(part: UploadFile, n: int) -> list[str]:
    """Read the ``texts`` part as a list of exactly ``n`` strings.

    Args:
        part: The uploaded ``texts.json``.
        n: The expected number of texts.

    Returns:
        The texts.

    Raises:
        HTTPException: 422, if the part is not a JSON array of ``n`` strings.
    """
    try:
        texts = json.loads(part.file.read())
    except (ValueError, RecursionError):
        raise HTTPException(status_code=422, detail="texts must be a JSON array of strings") from None
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise HTTPException(status_code=422, detail="texts must be a JSON array of strings")
    if len(texts) != n:
        raise HTTPException(status_code=422, detail=f"texts holds {len(texts)} items, expected n = {n}")
    return texts


@app.post(_FIT_PATH)
def fit(
    request: Request,
    meta: Annotated[UploadFile, File()],
    texts: Annotated[UploadFile, File()],
    vectors: Annotated[UploadFile, File()],
) -> FitResponse:
    """Fit a topic model to the posted vectors.

    Declared ``def`` (not ``async``) on purpose: FastAPI then runs it on the threadpool, so a fit
    that takes minutes cannot starve ``/health`` on the event loop. ``_UploadGuard`` has given this
    request the one fit slot and bounded its body before FastAPI parsed it; a request that reaches
    the handler without that admission is refused, so no fit runs outside the slot.

    Args:
        request: The request, whose scope carries the guard's admission.
        meta: ``meta.json``, the settings and the shape of the other parts.
        texts: ``texts.json``, one text per vector.
        vectors: ``vectors.f32``, the vectors as little-endian float32.

    Returns:
        FitResponse: Each input's topic, and each topic's size and keywords.

    Raises:
        HTTPException: 400 for a bad ``meta``, 413 for a request over the size limits, 422 for
            inconsistent or unusable inputs, 500 if the request was not admitted or the fit fails.
    """
    if not request.scope.get(_ADMITTED_KEY):
        _log.error("topics fit refused: the request reached the handler without the fit slot")
        raise HTTPException(status_code=500, detail="the request was not admitted as the running fit")
    settings = _parse_meta(meta)
    if settings.n > MAX_POINTS:
        raise HTTPException(status_code=413, detail=f"n={settings.n} exceeds TOPICS_MAX_POINTS={MAX_POINTS}")
    budget = MAX_POINTS * _VALUES_PER_POINT
    if settings.n * settings.dim > budget:
        raise HTTPException(
            status_code=413,
            detail=f"n*dim={settings.n * settings.dim} exceeds TOPICS_MAX_POINTS*{_VALUES_PER_POINT}={budget}",
        )
    texts_limit = min(settings.n * _TEXTS_BYTES_PER_POINT + _TEXTS_BYTES_SLACK, _TEXTS_MAX_BYTES)
    _require_size_at_most(texts, texts_limit, "texts")
    params = topics_fit.FitParams(
        seed=settings.seed,
        min_topic_size=settings.min_topic_size,
        nr_topics=settings.nr_topics,
        top_n_words=settings.top_n_words,
        min_df=settings.min_df,
        stop_words=tuple(settings.stop_words),
    )
    minimum = topics_fit.min_points(params)
    if settings.n < minimum:
        raise HTTPException(
            status_code=422,
            detail=f"need at least {minimum} points for min_topic_size={settings.min_topic_size}, got {settings.n}",
        )
    points = _read_vectors(vectors, settings)
    documents = _read_texts(texts, settings.n)
    try:
        result = topics_fit.fit_topics(documents, points, params)
    except Exception as exc:
        # An error message can quote the input, so log the type and the frames, not the message.
        frames = "".join(traceback.format_tb(exc.__traceback__))
        _log.error("topics fit failed n=%d dim=%d error=%s\n%s", settings.n, settings.dim, type(exc).__name__, frames)
        raise HTTPException(status_code=500, detail="topic-model fit failed") from None
    _log.info(
        "topics fit n=%d dim=%d topics=%d outliers=%d fit_s=%.2f",
        settings.n,
        settings.dim,
        len(result.topics),
        result.outliers,
        result.fit_s,
    )
    return FitResponse(
        labels=result.labels,
        outliers=result.outliers,
        topics=[Topic(id=topic.id, size=topic.size, keywords=topic.keywords) for topic in result.topics],
        versions=topics_fit.versions(),
        runtime=topics_fit.runtime(),
        timings={"fit_s": result.fit_s},
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "versions": topics_fit.versions()}
