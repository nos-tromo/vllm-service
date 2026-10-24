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

        Errors: 400 for a malformed or out-of-range ``meta``; 413 when ``n`` exceeds
        ``TOPICS_MAX_POINTS``; 422 for a missing part, a text count other than ``n``, a vector
        byte count other than ``n * dim * 4``, a NaN or infinity, or fewer points than the fit
        needs; 503 with ``Retry-After`` while another fit runs (one at a time, never queued);
        500 when the fit itself fails.

    GET /health
        Liveness probe; returns ``{"status": "ok", "versions": {...}}``. The server only starts
        answering once the startup self-check has passed.

Configuration:

    TOPICS_MAX_POINTS   Largest accepted ``n``. Default 150000: a fit peaks near 3.6 GB per 100k
                        points at 1024 dimensions. Must be a positive integer.
    TOPICS_WARMUP       Run ``topics_fit.self_check()`` at import, which compiles the numba code
                        and fits a synthetic set twice. On unless set to false, 0, no or off.

Starlette spools each uploaded part to a temporary file, which moves to disk beyond 1 MiB, so a
request body is received without being held in memory. A part is read into memory only once the
checks before it have passed: ``meta`` first, then ``n`` against the limit, then ``vectors``, then
``texts``.

The server logs counts and timings only, never texts, vectors or keywords.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, field_validator

import topics_fit

_WARMUP_OFF = frozenset({"false", "0", "no", "off"})

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

# A fit holds the whole request in memory for minutes, so only one runs at a time.
_fit_lock = threading.Lock()


class FitMeta(BaseModel):
    """The ``meta`` part: the shape of the other two parts, and the fit's settings.

    Attributes:
        n: Number of points: the length of ``texts`` and the row count of ``vectors``.
        dim: Vector dimension.
        seed: UMAP's random state, within the range numpy accepts as a seed.
        min_topic_size: HDBSCAN's ``min_cluster_size``, of which 2 is the least it accepts.
        nr_topics: Topic-reduction target, or 0 for none. BERTopic counts the outlier class as
            one of these, so 1 is refused.
        top_n_words: Keywords kept per topic.
        min_df: The fewest texts a word must occur in to become a keyword.
        stop_words: Words never used as keywords.
    """

    n: int = Field(ge=1)
    dim: int = Field(ge=1)
    seed: int = Field(ge=0, lt=2**32)
    min_topic_size: int = Field(ge=2)
    nr_topics: int = Field(ge=0)
    top_n_words: int = Field(ge=1, le=30)
    min_df: int = Field(ge=1)
    stop_words: list[str] = Field(max_length=10_000)

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


def _parse_meta(part: UploadFile) -> FitMeta:
    """Parse and validate the ``meta`` part.

    Args:
        part: The uploaded ``meta.json``.

    Returns:
        The validated meta.

    Raises:
        HTTPException: 400, if the part is not JSON, not an object, or has a missing or
            out-of-range field.
    """
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


@app.post("/topics/fit")
def fit(
    meta: Annotated[UploadFile, File()],
    texts: Annotated[UploadFile, File()],
    vectors: Annotated[UploadFile, File()],
) -> FitResponse:
    """Fit a topic model to the posted vectors.

    Declared ``def`` (not ``async``) on purpose: FastAPI then runs it on the threadpool, so a fit
    that takes minutes cannot starve ``/health`` on the event loop. The lock is taken before
    anything is read, so a request that arrives during a fit is refused at once rather than
    queued behind it, and only one request's vectors are ever in memory.

    Args:
        meta: ``meta.json``, the settings and the shape of the other parts.
        texts: ``texts.json``, one text per vector.
        vectors: ``vectors.f32``, the vectors as little-endian float32.

    Returns:
        FitResponse: Each input's topic, and each topic's size and keywords.

    Raises:
        HTTPException: 503 while another fit runs, 400 for a bad ``meta``, 413 for too many
            points, 422 for inconsistent or unusable inputs, 500 if the fit fails.
    """
    if not _fit_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503, detail="a topic-model fit is already running", headers={"Retry-After": "30"}
        )
    try:
        settings = _parse_meta(meta)
        if settings.n > MAX_POINTS:
            raise HTTPException(status_code=413, detail=f"n={settings.n} exceeds TOPICS_MAX_POINTS={MAX_POINTS}")
        points = _read_vectors(vectors, settings)
        documents = _read_texts(texts, settings.n)
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
        try:
            result = topics_fit.fit_topics(documents, points, params)
        except Exception as exc:
            # An error message can quote the input, so log the type and the frames, not the message.
            frames = "".join(traceback.format_tb(exc.__traceback__))
            _log.error(
                "topics fit failed n=%d dim=%d error=%s\n%s", settings.n, settings.dim, type(exc).__name__, frames
            )
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
    finally:
        _fit_lock.release()


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "versions": topics_fit.versions()}
