"""HTTP-contract tests for the topics fit server in ``src/topics_server.py``.

The fit is stubbed, so these tests need only numpy and FastAPI and run in the root eval
environment; the fit itself is covered by ``test_topics_fit.py``. The live-server tests also need
uvicorn, which only the topics environment has, and are skipped without it. All data is synthetic.
"""

import asyncio
import contextlib
import importlib.util
import io
import itertools
import json
import logging
import select
import socket
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

SRC = Path(__file__).resolve().parents[2] / "src"

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(SRC))

with pytest.MonkeyPatch.context() as _patch:
    # The startup self-check fits BERTopic, which this environment does not have.
    _patch.setenv("TOPICS_WARMUP", "false")
    _patch.delenv("TOPICS_MAX_POINTS", raising=False)
    import topics_server

from topics_fit import FitParams, FitResult, TopicInfo, min_points  # noqa: E402

client = TestClient(topics_server.app)

# The real fit, kept before any test replaces it with a stub.
REAL_FIT_TOPICS = topics_server.topics_fit.fit_topics

N = 40
DIM = 8
LOGGER = topics_server._log.name
META_FIELDS = ("n", "dim", "seed", "min_topic_size", "nr_topics", "top_n_words", "min_df", "stop_words")
META_MAX_BYTES = 1024 * 1024
TEXTS_LIMIT = N * 16_384 + 65_536  # the texts part's size limit for N points

FitCalls = list[tuple[list[str], np.ndarray, FitParams]]


def _meta(**overrides: object) -> dict[str, object]:
    """Build a valid ``meta`` object for ``N`` points of ``DIM`` dimensions, with any field replaced.

    Args:
        **overrides: Fields to replace.

    Returns:
        The meta object, ready for JSON.
    """
    meta: dict[str, object] = {
        "n": N,
        "dim": DIM,
        "seed": 7,
        "min_topic_size": 5,
        "nr_topics": 0,
        "top_n_words": 10,
        "min_df": 1,
        "stop_words": [],
    }
    return meta | overrides


def _vectors(n: int = N, dim: int = DIM) -> np.ndarray:
    """Build ``n`` synthetic float32 vectors.

    Args:
        n: The number of points.
        dim: The vector dimension.

    Returns:
        An ``n x dim`` little-endian float32 array.
    """
    return np.random.default_rng(3).normal(size=(n, dim)).astype("<f4")


def _texts(n: int = N) -> list[str]:
    """Build ``n`` synthetic texts."""
    return [f"synthetic text {i}" for i in range(n)]


def _blobs() -> tuple[list[str], np.ndarray]:
    """Build three well-separated blobs of 60 points in 32 dimensions, each speaking its own invented vocabulary.

    Returns:
        The texts and the L2-normalised little-endian float32 vectors.
    """
    rng = np.random.default_rng(7)
    prefixes = ("alpha", "beta", "gamma")
    centres = rng.normal(scale=4.0, size=(len(prefixes), 32))
    vectors = np.concatenate([centre + rng.normal(size=(60, 32)) for centre in centres])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    texts = [
        " ".join(f"{prefix}{word}" for word in rng.integers(0, 10, size=6)) for prefix in prefixes for _ in range(60)
    ]
    return texts, vectors.astype("<f4")


def _part(value: object) -> bytes:
    """Serialize a JSON part as UTF-8; bytes pass through so a test can send malformed content."""
    if isinstance(value, bytes):
        return value
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _padded_meta(size: int) -> bytes:
    """Build a valid meta part of exactly ``size`` bytes: the JSON, then trailing spaces.

    Args:
        size: The part's size in bytes.

    Returns:
        The padded meta part.
    """
    raw = _part(_meta())
    return raw + b" " * (size - len(raw))


def _texts_part(n: int, size: int) -> bytes:
    """Build a valid texts part of ``n`` strings and exactly ``size`` bytes, by padding the first string.

    Args:
        n: The number of texts.
        size: The part's size in bytes; at least that of the unpadded array.

    Returns:
        The padded texts part.
    """
    texts = _texts(n)
    texts[0] += "a" * (size - len(_part(texts)))
    part = _part(texts)
    assert len(part) == size
    return part


def _files(meta: object, texts: object, vectors: bytes | None) -> dict[str, tuple[str, bytes, str]]:
    """Build the three file parts, leaving out any given as ``None``.

    Args:
        meta: The ``meta`` object, or raw bytes.
        texts: The texts, or raw bytes.
        vectors: The raw vector bytes.

    Returns:
        The parts, as httpx's ``files`` argument takes them.
    """
    files: dict[str, tuple[str, bytes, str]] = {}
    if meta is not None:
        files["meta"] = ("meta.json", _part(meta), "application/json")
    if texts is not None:
        files["texts"] = ("texts.json", _part(texts), "application/json")
    if vectors is not None:
        files["vectors"] = ("vectors.f32", vectors, "application/octet-stream")
    return files


def _post(meta: object, texts: object, vectors: bytes | None) -> httpx.Response:
    """POST the three parts as file uploads, leaving out any given as ``None``.

    Args:
        meta: The ``meta`` object, or raw bytes.
        texts: The texts, or raw bytes.
        vectors: The raw vector bytes.

    Returns:
        The response.
    """
    return client.post("/topics/fit", files=_files(meta, texts, vectors))


def _post_valid(n: int = N, dim: int = DIM, **meta: object) -> httpx.Response:
    """POST a consistent request of ``n`` points, with any meta field replaced."""
    return _post(_meta(n=n, dim=dim, **meta), _texts(n), _vectors(n, dim).tobytes())


def _post_while_busy(meta: object, texts: object, vectors: bytes | None) -> tuple[httpx.Response, bool]:
    """POST while another request holds the fit lock, as one does for the length of its upload and fit.

    Args:
        meta: The ``meta`` object, or raw bytes.
        texts: The texts, or raw bytes.
        vectors: The raw vector bytes.

    Returns:
        The response, and whether the lock was still held once it arrived.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    topics_server._fit_lock.acquire()
    try:
        # A request that waited for the lock would hang; bound it so that fails fast instead.
        response = pool.submit(_post, meta, texts, vectors).result(timeout=10)
        return response, topics_server._fit_lock.locked()
    finally:
        topics_server._fit_lock.release()
        pool.shutdown()


@pytest.fixture(autouse=True)
def fit_calls(monkeypatch: pytest.MonkeyPatch) -> FitCalls:
    """Replace the fit with a one-topic stub, and record each call's arguments.

    Returns:
        The recorded ``(texts, vectors, params)`` of every call.
    """
    calls: FitCalls = []

    def fake_fit(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        calls.append((list(texts), vectors, params))
        n = len(texts)
        return FitResult(labels=[0] * n, topics=[TopicInfo(0, n, [("w", 1.0)])], outliers=0, fit_s=0.01)

    monkeypatch.setattr(topics_server.topics_fit, "fit_topics", fake_fit)
    return calls


@pytest.fixture(autouse=True)
def lock_is_free() -> Iterator[None]:
    """Fail a test that leaves the fit lock held, and free it so later tests are not refused.

    The lock is released just after a response has been sent, so a client may hold its answer
    while the lock is still taken; only a lock still held two seconds later counts as left.
    """
    yield
    deadline = time.monotonic() + 2
    while topics_server._fit_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    leaked = topics_server._fit_lock.locked()
    if leaked:
        topics_server._fit_lock.release()
    assert not leaked, "the fit lock was left held"


def _load_fresh(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    """Execute ``topics_server.py`` once more as a separate module, with these variables set.

    Args:
        monkeypatch: The test's monkeypatch, which undoes the variables and the registration.
        **env: Environment variables to set; the module's other settings are left unset.

    Returns:
        The freshly executed module.
    """
    for name in ("TOPICS_WARMUP", "TOPICS_MAX_POINTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("topics_server_fresh", SRC / "topics_server.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_fit_returns_contract() -> None:
    """A valid request gets the documented 200 body."""
    response = _post_valid()
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"labels", "outliers", "topics", "versions", "runtime", "timings"}
    assert len(body["labels"]) == N
    assert body["outliers"] == 0
    assert body["topics"][0] == {"id": 0, "size": N, "keywords": [["w", 1.0]]}
    assert body["timings"] == {"fit_s": 0.01}
    assert body["versions"] == topics_server.topics_fit.versions()
    assert body["runtime"] == topics_server.topics_fit.runtime()


def test_meta_and_inputs_reach_the_fit(fit_calls: FitCalls) -> None:
    """The fit gets the texts, the vectors bit for bit, and the settings from meta."""
    texts = ["größe über alles", "日本語のテキスト", "plain ascii", *_texts(N - 3)]
    vectors = _vectors()
    meta = _meta(seed=11, min_topic_size=6, nr_topics=4, top_n_words=7, min_df=2, stop_words=["der", "die"])
    assert _post(meta, texts, vectors.tobytes()).status_code == 200
    ((got_texts, got_vectors, params),) = fit_calls
    assert got_texts == texts
    assert got_vectors.shape == (N, DIM)
    assert got_vectors.dtype == np.float32
    assert np.array_equal(got_vectors, vectors)
    assert params == FitParams(
        seed=11, min_topic_size=6, nr_topics=4, top_n_words=7, min_df=2, stop_words=("der", "die")
    )


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe\x00", id="not-utf8"),
        pytest.param(b"[1, 2]", id="not-an-object"),
        pytest.param(b"[" * 100_000, id="nested-too-deep"),
        pytest.param({name: value for name, value in _meta().items() if name != "dim"}, id="missing-dim"),
    ],
)
def test_malformed_meta_is_400(meta: object, fit_calls: FitCalls) -> None:
    """A meta part that is not a JSON object with every field is a 400, whatever it holds."""
    assert _post(meta, _texts(), _vectors().tobytes()).status_code == 400
    assert fit_calls == []


@pytest.mark.parametrize("field", META_FIELDS)
def test_missing_meta_field_is_400(field: str, fit_calls: FitCalls) -> None:
    """Every setting is required: the same bytes and settings must give the same fit."""
    meta = {name: value for name, value in _meta().items() if name != field}
    response = _post(meta, _texts(), _vectors().tobytes())
    assert response.status_code == 400
    assert field in response.json()["detail"]
    assert fit_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 0),
        ("n", 40.5),
        ("n", "forty"),
        ("dim", 0),
        pytest.param("dim", 8193, id="dim-8193"),
        pytest.param("dim", 1_000_000, id="dim-a-million"),
        ("seed", -1),
        ("seed", 2**32),
        ("min_topic_size", 1),
        ("nr_topics", -1),
        ("top_n_words", 0),
        ("top_n_words", 31),
        ("min_df", 0),
        ("stop_words", "the"),
        ("stop_words", [1, 2]),
        ("stop_words", ["x"] * 10_001),
        pytest.param("stop_words", ["x" * 65], id="stop-word-65-characters"),
        pytest.param("stop_words", ["fine", "x" * 500_000], id="stop-word-half-a-million-characters"),
    ],
)
def test_out_of_range_meta_is_400(field: str, value: object, fit_calls: FitCalls) -> None:
    """A setting outside its range is a 400 naming the field, not a failure inside the fit."""
    response = _post(_meta(**{field: value}), _texts(), _vectors().tobytes())
    assert response.status_code == 400
    assert field in response.json()["detail"]
    assert fit_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 0),
        ("seed", 2**32 - 1),
        ("min_topic_size", 2),
        ("nr_topics", 0),
        ("nr_topics", 2),
        ("top_n_words", 1),
        ("top_n_words", 30),
        ("min_df", 1),
        ("stop_words", ["x"] * 10_000),
        pytest.param("stop_words", ["x" * 64], id="stop-word-64-characters"),
    ],
)
def test_meta_bounds_are_inclusive(field: str, value: object, fit_calls: FitCalls) -> None:
    """The limits themselves are accepted."""
    assert _post(_meta(**{field: value}), _texts(), _vectors().tobytes()).status_code == 200
    assert len(fit_calls) == 1


def test_dim_limit_is_inclusive(fit_calls: FitCalls) -> None:
    """A dimension of 8192 is accepted with its vectors; 8193 is a 400 whatever the vectors."""
    assert _post_valid(dim=8192).status_code == 200
    assert len(fit_calls) == 1
    assert _post_valid(dim=8193).status_code == 400
    assert len(fit_calls) == 1


def test_nr_topics_one_is_400(fit_calls: FitCalls) -> None:
    """BERTopic counts the outlier class as a topic, so 1 would crash the fit: only 0 or 2+ are accepted."""
    response = _post_valid(nr_topics=1)
    assert response.status_code == 400
    assert "nr_topics" in response.json()["detail"]
    assert fit_calls == []


def test_too_many_points_is_413(monkeypatch: pytest.MonkeyPatch, fit_calls: FitCalls) -> None:
    """More points than ``TOPICS_MAX_POINTS`` is refused, and the fit never runs."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 10)
    response = _post_valid(n=11)
    assert response.status_code == 413
    assert "10" in response.json()["detail"]
    assert fit_calls == []


def test_the_limit_itself_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """``n == TOPICS_MAX_POINTS`` is within the limit; one more is not."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    assert _post_valid(n=16).status_code == 200
    assert _post_valid(n=17).status_code == 413


def test_413_is_decided_from_meta_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The large parts are not read for an oversize fit: garbage in them still gets a 413."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 10)
    assert _post(_meta(n=11), b"not json", b"\x00").status_code == 413


def test_n_times_dim_over_budget_is_413(monkeypatch: pytest.MonkeyPatch, fit_calls: FitCalls) -> None:
    """``n * dim`` may not exceed ``TOPICS_MAX_POINTS * 1024``, decided from meta alone like the point limit."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    response = _post(_meta(n=16, dim=1025), b"not json", b"\x00")  # 16 * 1025 is 16 over 16 * 1024
    assert response.status_code == 413
    assert "TOPICS_MAX_POINTS" in response.json()["detail"]
    assert fit_calls == []
    assert _post_valid(n=16, dim=1024).status_code == 200  # exactly the budget
    assert len(fit_calls) == 1


def test_a_wide_dimension_cannot_buy_more_memory(fit_calls: FitCalls) -> None:
    """With the default limit, the most points of the widest dimension are refused from meta alone."""
    assert _post(_meta(n=150_000, dim=8192), b"not json", b"\x00").status_code == 413
    assert fit_calls == []


def test_oversize_meta_is_413(fit_calls: FitCalls) -> None:
    """A meta part over 1 MiB is refused from its size alone, unread; one of exactly 1 MiB is read as usual."""
    response = _post(b"x" * (META_MAX_BYTES + 1), _texts(), _vectors().tobytes())
    assert response.status_code == 413
    assert _post(_meta(stop_words=["x" * 2_000_000]), _texts(), _vectors().tobytes()).status_code == 413
    # Even a meta that would be valid is refused for its size: it is judged before it is parsed.
    assert _post(_padded_meta(META_MAX_BYTES + 1), _texts(), _vectors().tobytes()).status_code == 413
    assert fit_calls == []
    assert _post(_padded_meta(META_MAX_BYTES), _texts(), _vectors().tobytes()).status_code == 200
    assert len(fit_calls) == 1


def test_a_part_of_unknown_size_is_refused() -> None:
    """A part whose size is not known cannot be judged, so it is refused rather than read."""
    part = UploadFile(file=io.BytesIO(b"{}"))  # a part built by hand, not parsed from a request, has no size
    with pytest.raises(HTTPException) as refused:
        topics_server._require_size_at_most(part, 1024, "meta")
    assert refused.value.status_code == 413


def test_oversize_texts_is_413(fit_calls: FitCalls) -> None:
    """The texts part may hold ``n * 16384 + 65536`` bytes; more is refused from its size alone."""
    # Garbage in the other parts: the texts size is judged before either is read or parsed.
    assert _post(_meta(), b"x" * (TEXTS_LIMIT + 1), b"\x00").status_code == 413
    assert _post(_meta(), _texts_part(N, TEXTS_LIMIT + 1), _vectors().tobytes()).status_code == 413
    assert fit_calls == []
    assert _post(_meta(), _texts_part(N, TEXTS_LIMIT), _vectors().tobytes()).status_code == 200
    assert len(fit_calls) == 1


def test_texts_limit_is_capped(monkeypatch: pytest.MonkeyPatch, fit_calls: FitCalls) -> None:
    """However many points there are, the texts part never gets more than 512 MiB."""
    assert topics_server._TEXTS_MAX_BYTES == 512 * 1024 * 1024
    monkeypatch.setattr(topics_server, "_TEXTS_MAX_BYTES", 1_000)
    assert _post(_meta(), _texts_part(N, 1_001), _vectors().tobytes()).status_code == 413
    assert fit_calls == []
    assert _post(_meta(), _texts_part(N, 1_000), _vectors().tobytes()).status_code == 200
    assert len(fit_calls) == 1


@pytest.mark.parametrize("length", [N * DIM * 4 - 4, N * DIM * 4 + 4, N * DIM * 4 - 1, 0])
def test_vector_size_mismatch_is_422(length: int, fit_calls: FitCalls) -> None:
    """A vectors part that is not exactly ``n * dim * 4`` bytes is a 422 naming the expected size."""
    response = _post(_meta(), _texts(), bytes(length))
    assert response.status_code == 422
    assert str(N * DIM * 4) in response.json()["detail"]
    assert fit_calls == []


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_vectors_are_422(bad: float, fit_calls: FitCalls) -> None:
    """A NaN or an infinity anywhere in the vectors is a 422 before any fit work."""
    vectors = _vectors()
    vectors[N // 2, DIM - 1] = bad
    assert _post(_meta(), _texts(), vectors.tobytes()).status_code == 422
    assert fit_calls == []


@pytest.mark.parametrize(
    "texts",
    [
        pytest.param(_texts(N - 1), id="one-short"),
        pytest.param(_texts(N + 1), id="one-over"),
        pytest.param([], id="empty"),
        pytest.param({"0": "text"}, id="object"),
        pytest.param("text", id="string"),
        pytest.param([*_texts(N - 1), 5], id="number-among-texts"),
        pytest.param([*_texts(N - 1), None], id="null-among-texts"),
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe\x00", id="not-utf8"),
        pytest.param(b"[" * 100_000, id="nested-too-deep"),
    ],
)
def test_text_count_mismatch_is_422(texts: object, fit_calls: FitCalls) -> None:
    """The texts must be a JSON array of exactly ``n`` strings."""
    assert _post(_meta(), texts, _vectors().tobytes()).status_code == 422
    assert fit_calls == []


@pytest.mark.parametrize(
    ("min_topic_size", "minimum"),
    [
        pytest.param(2, 16, id="floor-of-16"),
        pytest.param(10, 20, id="twice-the-topic-size"),
        pytest.param(30, 60, id="large-topic-size"),
    ],
)
def test_too_few_points_is_422(min_topic_size: int, minimum: int, fit_calls: FitCalls) -> None:
    """Fewer points than the fit can use is a 422 naming the minimum; the minimum itself is accepted."""
    params = FitParams(seed=7, min_topic_size=min_topic_size, nr_topics=0, top_n_words=10, min_df=1, stop_words=())
    assert min_points(params) == minimum
    response = _post_valid(n=minimum - 1, min_topic_size=min_topic_size)
    assert response.status_code == 422
    assert str(minimum) in response.json()["detail"]
    assert fit_calls == []
    assert _post_valid(n=minimum, min_topic_size=min_topic_size).status_code == 200
    assert len(fit_calls) == 1


def test_too_few_points_is_decided_from_meta_alone(fit_calls: FitCalls) -> None:
    """The minimum needs only meta, so it is checked before either large part is judged or read."""
    oversize_texts = b"x" * (15 * 16_384 + 65_536 + 1)  # over the texts limit for 15 points
    for texts in (b"not json", oversize_texts):
        response = _post(_meta(n=15), texts, b"\x00")
        assert response.status_code == 422
        assert "at least 16" in response.json()["detail"]
    assert fit_calls == []


@pytest.mark.parametrize("missing", ["meta", "texts", "vectors"])
def test_missing_part_is_422(missing: str, fit_calls: FitCalls) -> None:
    """Each of the three parts is required."""
    response = _post(
        None if missing == "meta" else _meta(),
        None if missing == "texts" else _texts(),
        None if missing == "vectors" else _vectors().tobytes(),
    )
    assert response.status_code == 422
    assert fit_calls == []


def test_non_multipart_body_is_422(fit_calls: FitCalls) -> None:
    """A body that is not multipart carries none of the parts."""
    assert client.post("/topics/fit", json=_meta()).status_code == 422
    assert fit_calls == []


def test_multipart_without_a_boundary_is_400(fit_calls: FitCalls) -> None:
    """A multipart body that names no boundary cannot be split into parts: Starlette refuses it with a 400."""
    response = client.post("/topics/fit", content=b"", headers={"content-type": "multipart/form-data"})
    assert response.status_code == 400
    assert fit_calls == []


def test_busy_returns_503(fit_calls: FitCalls) -> None:
    """A request during a fit is refused at once, never queued, and leaves the running fit's lock held."""
    response, still_held = _post_while_busy(_meta(), _texts(), _vectors().tobytes())
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert still_held
    assert fit_calls == []
    assert _post_valid().status_code == 200


@pytest.mark.parametrize(
    ("meta", "texts", "vectors"),
    [
        pytest.param(b"not json", b"x", b"\x00", id="malformed-meta"),
        pytest.param(_meta(n=10_000_000), b"x", b"\x00", id="too-many-points"),
        pytest.param(_meta(), _texts(), b"\x00", id="wrong-vector-size"),
    ],
)
def test_busy_is_decided_before_the_request_is_read(
    meta: object, texts: object, vectors: bytes, fit_calls: FitCalls
) -> None:
    """The slot is checked before anything is read: a request that would be refused for its content gets the 503."""
    response, _ = _post_while_busy(meta, texts, vectors)
    assert response.status_code == 503
    assert fit_calls == []


@pytest.mark.parametrize("error", [RuntimeError, ValueError, MemoryError])
def test_fit_failure_is_500(error: type[Exception], monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit that raises is a 500 with a fixed message, and the lock is free for the next request."""

    def failing_fit(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        raise error(f"cannot fit {texts[0]!r}")

    monkeypatch.setattr(topics_server.topics_fit, "fit_topics", failing_fit)
    texts = [f"quokka-lantern {i}" for i in range(N)]
    response = _post(_meta(), texts, _vectors().tobytes())
    assert response.status_code == 500
    assert response.json() == {"detail": "topic-model fit failed"}
    assert "quokka" not in response.text
    assert not topics_server._fit_lock.locked()


def test_logs_carry_counts_only(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The log line has counts and a timing, and none of the request's texts or the fit's keywords."""

    def keyword_fit(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        return FitResult(
            labels=[0] * len(texts), topics=[TopicInfo(0, len(texts), [("zebra", 0.5)])], outliers=0, fit_s=0.01
        )

    monkeypatch.setattr(topics_server.topics_fit, "fit_topics", keyword_fit)
    texts = [f"quokka-lantern {i}" for i in range(N)]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _post(_meta(), texts, _vectors().tobytes()).status_code == 200
    assert "topics fit n=40 dim=8 topics=1 outliers=0 fit_s=0.01" in caplog.messages
    assert "quokka" not in caplog.text
    assert "zebra" not in caplog.text


def test_failed_fit_logs_the_error_type_not_its_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An error message can quote input text, so the log names the type and the sizes only."""

    def failing_fit(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        raise RuntimeError(f"cannot fit {texts[0]!r}")

    monkeypatch.setattr(topics_server.topics_fit, "fit_topics", failing_fit)
    texts = [f"quokka-lantern {i}" for i in range(N)]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _post(_meta(), texts, _vectors().tobytes()).status_code == 500
    assert "RuntimeError" in caplog.text
    assert "n=40 dim=8" in caplog.text
    assert "quokka" not in caplog.text


def test_real_fit_through_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vectors the server hands over, a read-only view of the request bytes, work with the real fit."""
    pytest.importorskip("bertopic")
    monkeypatch.setattr(topics_server.topics_fit, "fit_topics", REAL_FIT_TOPICS)
    texts, vectors = _blobs()
    n, dim = vectors.shape
    response = _post(_meta(n=n, dim=dim, min_topic_size=10, top_n_words=5), texts, vectors.tobytes())
    assert response.status_code == 200
    body = response.json()
    assert len(body["labels"]) == n
    assert sum(topic["size"] for topic in body["topics"]) + body["outliers"] == n
    top_keywords = [topic["keywords"][0][0].rstrip("0123456789") for topic in body["topics"]]
    assert sorted(top_keywords) == ["alpha", "beta", "gamma"]


def test_health_reports_versions() -> None:
    """The health probe answers ok and lists the fit libraries' versions."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["versions"]) == {"bertopic", "umap-learn", "hdbscan", "scikit-learn", "numpy", "numba"}
    assert body == {"status": "ok", "versions": topics_server.topics_fit.versions()}


def test_max_points_defaults_to_150000() -> None:
    """The default is 150000 points: about 5.4 GB at 1024 dimensions with 2,000-character texts, extrapolated."""
    assert topics_server.MAX_POINTS == 150_000


def test_max_points_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TOPICS_MAX_POINTS`` sets the limit once, at import."""
    assert _load_fresh(monkeypatch, TOPICS_WARMUP="false", TOPICS_MAX_POINTS="2500").MAX_POINTS == 2500


@pytest.mark.parametrize("value", ["0", "-5", "many", "1.5", ""])
def test_max_points_must_be_a_positive_integer(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A bad limit stops the server at startup, naming the variable, rather than silently refusing every fit."""
    with pytest.raises(ValueError, match="TOPICS_MAX_POINTS"):
        _load_fresh(monkeypatch, TOPICS_WARMUP="false", TOPICS_MAX_POINTS=value)


@pytest.mark.parametrize(
    ("env", "runs"),
    [
        pytest.param({}, 1, id="default-on"),
        pytest.param({"TOPICS_WARMUP": "true"}, 1, id="true"),
        pytest.param({"TOPICS_WARMUP": "false"}, 0, id="false"),
        pytest.param({"TOPICS_WARMUP": "0"}, 0, id="zero"),
    ],
)
def test_self_check_runs_at_import_unless_disabled(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], runs: int
) -> None:
    """The determinism self-check runs when the module loads, unless ``TOPICS_WARMUP`` turns it off."""
    calls: list[int] = []
    monkeypatch.setattr(topics_server.topics_fit, "self_check", lambda: calls.append(1))
    _load_fresh(monkeypatch, **env)
    assert len(calls) == runs


def test_failed_self_check_stops_the_server_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit that is not deterministic must keep the service from ever answering health."""

    def failing_self_check() -> None:
        raise RuntimeError("topic-model fit is not deterministic")

    monkeypatch.setattr(topics_server.topics_fit, "self_check", failing_self_check)
    with pytest.raises(RuntimeError, match="not deterministic"):
        _load_fresh(monkeypatch, TOPICS_WARMUP="true")


# The upload guard: what is refused before any of the body is read, and how much of a body is read at all.

# The largest body a fit can need: meta's 1 MiB, the texts limit at TOPICS_MAX_POINTS points, the
# n * dim float32 values at their TOPICS_MAX_POINTS * 1024 cap, and 64 KiB of multipart framing.
DEFAULT_CEILING = 1_048_576 + 536_870_912 + 150_000 * 1024 * 4 + 65_536  # 1,152,385,024 bytes
CEILING_AT_16_POINTS = 1_048_576 + (16 * 16_384 + 65_536) + 16 * 1024 * 4 + 65_536  # 1,507,328 bytes
CHUNK = 65_536
BOUNDARY_TYPE = "multipart/form-data; boundary=b"
# The opening of a ``vectors`` file part, for a body that never closes it.
VECTORS_PART_HEAD = (
    b"--b\r\n"
    b'Content-Disposition: form-data; name="vectors"; filename="vectors.f32"\r\n'
    b"Content-Type: application/octet-stream\r\n\r\n"
)
HTTP_IMPLEMENTATIONS = pytest.mark.parametrize("http", ["httptools", "h11"])


def _valid_upload() -> tuple[bytes, list[tuple[str, str]]]:
    """Encode a consistent request of ``N`` points as httpx sends it.

    Returns:
        The body, and the ``content-type`` and ``content-length`` headers that go with it.
    """
    request = httpx.Request("POST", "http://topics/topics/fit", files=_files(_meta(), _texts(), _vectors().tobytes()))
    content = request.read()
    return content, [("content-type", request.headers["content-type"]), ("content-length", str(len(content)))]


def _chunks(content: bytes, size: int = CHUNK) -> list[bytes]:
    """Split a body into the pieces a server hands over one ``receive()`` at a time."""
    return [content[start : start + size] for start in range(0, len(content), size)]


def _scope(
    headers: Sequence[tuple[str, str]] = (), method: str = "POST", path: str = "/topics/fit", root_path: str = ""
) -> Scope:
    """Build the ASGI scope of one HTTP/1.1 request as uvicorn passes it.

    Args:
        headers: Header names and values.
        method: The request method.
        path: The path, ``root_path`` included, as uvicorn passes it.
        root_path: The ASGI ``root_path``.

    Returns:
        The scope.
    """
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "root_path": root_path,
        "query_string": b"",
        "headers": [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers],
        "client": ("127.0.0.1", 40000),
        "server": ("127.0.0.1", 8000),
        "state": {},
    }


class _Body:
    """A request body handed to the app one ``receive()`` at a time, recording what the app pulls.

    Attributes:
        calls: How many times the app called ``receive()``.
        pulled: How many body bytes it was handed.
        held_at_end: Whether the fit slot was held when the body ran out; ``None`` until then.
    """

    def __init__(
        self, chunks: Iterable[bytes], then: str = "end", gate: asyncio.Event | None = None, pause_after: int = 0
    ) -> None:
        """Set out the body.

        Args:
            chunks: The body, in the pieces the app is to receive.
            then: What follows the last piece: ``"end"`` closes the body, ``"stall"`` sends nothing
                more, ``"disconnect"`` reports the client gone.
            gate: If given, the piece after the first ``pause_after`` waits until it is set.
            pause_after: How many pieces go out before the gate.
        """
        self._chunks = iter(chunks)
        self._then = then
        self._gate = gate
        self._pause_after = pause_after
        self._ended = False
        self.calls = 0
        self.pulled = 0
        self.held_at_end: bool | None = None

    async def receive(self) -> Message:
        """Hand the app its next message.

        Returns:
            The next ASGI ``receive`` message.
        """
        self.calls += 1
        if self._gate is not None and self.calls == self._pause_after + 1:
            await self._gate.wait()
        chunk = next(self._chunks, None)
        if chunk is not None:
            self.pulled += len(chunk)
            return {"type": "http.request", "body": chunk, "more_body": True}
        if self.held_at_end is None:
            self.held_at_end = topics_server._fit_lock.locked()
        if self._then == "stall":
            await asyncio.Event().wait()
        if self._then == "disconnect" or self._ended:
            return {"type": "http.disconnect"}
        self._ended = True
        return {"type": "http.request", "body": b"", "more_body": False}


@dataclass
class _Answer:
    """What the app sent back.

    Attributes:
        status: The status code.
        headers: The headers, with names lower-cased.
        body: The body.
    """

    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        """Decode the body as JSON.

        Returns:
            The decoded body.
        """
        return json.loads(self.body)


async def _exchange(scope: Scope, body: _Body) -> _Answer:
    """Run one request through the app, guard included, and collect what it sends back.

    Args:
        scope: The request's ASGI scope.
        body: The request body.

    Returns:
        The response.
    """
    start: dict[str, Any] = {}
    parts: list[bytes] = []

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            start.update(message)
        elif message["type"] == "http.response.body":
            parts.append(message.get("body", b""))

    await topics_server.app(scope, body.receive, send)
    headers = {name.decode("latin-1").lower(): value.decode("latin-1") for name, value in start.get("headers", [])}
    return _Answer(start["status"], headers, b"".join(parts))


def _drive(scope: Scope, body: _Body) -> _Answer:
    """Run one request through the app in an event loop of its own, failing one that hangs.

    Args:
        scope: The request's ASGI scope.
        body: The request body.

    Returns:
        The response.
    """
    return asyncio.run(asyncio.wait_for(_exchange(scope, body), timeout=5))


@contextlib.contextmanager
def _fit_running() -> Iterator[None]:
    """Hold the fit slot for the block, as a running fit does."""
    topics_server._fit_lock.acquire()
    try:
        yield
    finally:
        topics_server._fit_lock.release()


def _wait_for_free_slot() -> None:
    """Wait up to two seconds for the slot, which is freed just after a response has been sent."""
    deadline = time.monotonic() + 2
    while topics_server._fit_lock.locked():
        assert time.monotonic() < deadline, "the fit slot was not freed"
        time.sleep(0.01)


def test_busy_is_refused_before_any_body_is_read(fit_calls: FitCalls) -> None:
    """While a fit holds the slot, a new request gets its 503 before the app pulls a byte of its body."""
    content, headers = _valid_upload()
    body = _Body(_chunks(content))
    with _fit_running():
        answer = _drive(_scope(headers), body)
    assert answer.status == 503
    assert answer.headers["retry-after"] == "30"
    assert answer.json() == {"detail": "a topic-model fit is already running"}
    assert body.calls == 0
    assert fit_calls == []


def test_declared_oversize_is_refused_unread(fit_calls: FitCalls) -> None:
    """A declared length over the ceiling is a 413 with nothing read, decided before the slot is checked."""
    headers = [("content-type", BOUNDARY_TYPE), ("content-length", str(DEFAULT_CEILING + 1))]
    body = _Body([VECTORS_PART_HEAD])
    assert _drive(_scope(headers), body).status == 413
    assert body.calls == 0
    with _fit_running():
        busy_body = _Body([VECTORS_PART_HEAD])
        assert _drive(_scope(headers), busy_body).status == 413
    assert busy_body.calls == 0
    assert fit_calls == []


def test_the_ceiling_itself_is_not_refused() -> None:
    """A declared length of exactly the ceiling is not refused for its size."""
    headers = [("content-type", BOUNDARY_TYPE), ("content-length", str(DEFAULT_CEILING))]
    assert _drive(_scope(headers), _Body([b"--b--\r\n"])).status != 413


def test_streamed_oversize_is_cut_off(monkeypatch: pytest.MonkeyPatch, fit_calls: FitCalls) -> None:
    """A body sent without a length is cut off once it passes the ceiling, not read to its end."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    body = _Body(itertools.chain([VECTORS_PART_HEAD], itertools.repeat(bytes(CHUNK), 256)))  # 16 MiB on offer
    answer = _drive(_scope([("content-type", BOUNDARY_TYPE)]), body)
    assert answer.status == 413
    assert CEILING_AT_16_POINTS < body.pulled <= CEILING_AT_16_POINTS + CHUNK
    assert fit_calls == []


def test_a_streamed_body_of_exactly_the_ceiling_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body of exactly the ceiling is read to its end, not refused for its size."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    body = _Body([VECTORS_PART_HEAD, bytes(CEILING_AT_16_POINTS - len(VECTORS_PART_HEAD))])
    assert _drive(_scope([("content-type", BOUNDARY_TYPE)]), body).status != 413
    assert body.pulled == CEILING_AT_16_POINTS


def test_an_unreadable_declared_length_is_left_to_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``Content-Length`` that is not plain digits does not break the guard; the body is counted instead."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    headers = [("content-type", BOUNDARY_TYPE), ("content-length", "many")]
    body = _Body(itertools.chain([VECTORS_PART_HEAD], itertools.repeat(bytes(CHUNK), 256)))
    assert _drive(_scope(headers), body).status == 413
    assert body.pulled <= CEILING_AT_16_POINTS + CHUNK


def test_a_declared_length_does_not_stop_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body that runs past its declared length is still counted, as when h11 frames it chunked."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    headers = [("content-type", BOUNDARY_TYPE), ("content-length", "100")]
    body = _Body(itertools.chain([VECTORS_PART_HEAD], itertools.repeat(bytes(CHUNK), 256)))
    assert _drive(_scope(headers), body).status == 413
    assert body.pulled <= CEILING_AT_16_POINTS + CHUNK


def test_stalled_upload_gets_408_and_frees_the_slot(monkeypatch: pytest.MonkeyPatch, fit_calls: FitCalls) -> None:
    """A fit whose body stops arriving is refused at the deadline and closed, and the next fit is admitted."""
    monkeypatch.setattr(topics_server, "_BODY_TIMEOUT_S", 0.2)
    content, headers = _valid_upload()
    started = time.monotonic()
    answer = _drive(_scope(headers), _Body([content[:1024]], then="stall"))
    assert answer.status == 408
    assert answer.headers["connection"] == "close"
    assert time.monotonic() - started < 2
    assert fit_calls == []
    assert _post_valid().status_code == 200


def test_slot_is_held_for_the_whole_upload(fit_calls: FitCalls) -> None:
    """The slot is taken when a fit's upload begins, so a request arriving mid-upload is refused unread."""
    content, headers = _valid_upload()

    async def run() -> tuple[_Answer, _Answer, _Body, bool]:
        gate = asyncio.Event()
        first = _Body(_chunks(content, 256), gate=gate, pause_after=1)
        second = _Body(_chunks(content))
        upload = asyncio.create_task(_exchange(_scope(headers), first))
        while first.calls < 2:  # the first piece is in; the second waits at the gate
            await asyncio.sleep(0.01)
        held = topics_server._fit_lock.locked()
        refused = await _exchange(_scope(headers), second)
        gate.set()
        return await upload, refused, second, held

    admitted, refused, second, held = asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert held
    assert refused.status == 503
    assert second.calls == 0
    assert admitted.status == 200
    assert len(fit_calls) == 1


def test_disconnect_mid_upload_frees_the_slot(fit_calls: FitCalls) -> None:
    """A client that leaves mid-upload held the slot until it left, and frees it then; nothing is fitted."""
    content, headers = _valid_upload()
    body = _Body([content[:1024]], then="disconnect")
    _drive(_scope(headers), body)
    assert body.held_at_end is True
    assert not topics_server._fit_lock.locked()
    assert fit_calls == []


def test_health_is_never_gated() -> None:
    """The health probe answers while a fit holds the slot."""
    with _fit_running():
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "root_path", "status"),
    [
        pytest.param("POST", "/topics/fit/", "", 307, id="trailing-slash-redirects"),
        pytest.param("GET", "/topics/fit", "", 405, id="get"),
        pytest.param("HEAD", "/topics/fit", "", 405, id="head"),
        pytest.param("POST", "/svc/topics/fit", "/svc", 503, id="fit-under-a-root-path"),
    ],
)
def test_only_the_fit_route_waits_for_the_slot(method: str, path: str, root_path: str, status: int) -> None:
    """While busy, only a POST that routes to the fit gets the 503, and none of these reads the body."""
    body = _Body([VECTORS_PART_HEAD])
    with _fit_running():
        answer = _drive(_scope([("content-type", BOUNDARY_TYPE)], method, path, root_path), body)
    assert answer.status == status
    assert body.calls == 0


def test_a_fit_under_a_root_path_is_admitted(fit_calls: FitCalls) -> None:
    """A fit served under a ``root_path`` is matched the way the routes match it, and runs."""
    content, headers = _valid_upload()
    answer = _drive(_scope(headers, path="/svc/topics/fit", root_path="/svc"), _Body(_chunks(content)))
    assert answer.status == 200
    assert len(fit_calls) == 1


def test_the_fit_is_the_only_route_that_reads_a_body() -> None:
    """The guard admits ``POST /topics/fit`` alone, so a new route that reads a body needs a decision there."""
    body_routes = {
        (method, route.path)
        for route in topics_server.app.routes
        if isinstance(route, APIRoute) and route.body_field is not None
        for method in route.methods or ()
    }
    assert body_routes == {("POST", "/topics/fit")}


def test_a_fit_that_skipped_admission_is_refused(monkeypatch: pytest.MonkeyPatch, fit_calls: FitCalls) -> None:
    """Should the guard and the routes disagree on the fit's path, the fit refuses rather than run unadmitted."""
    monkeypatch.setattr(topics_server, "_FIT_PATH", "/elsewhere")
    assert _post_valid().status_code == 500
    assert fit_calls == []


@contextlib.contextmanager
def _serving(http: str) -> Iterator[int]:
    """Serve the app with uvicorn on a free local port for the length of the block.

    Args:
        http: uvicorn's HTTP implementation, ``httptools`` or ``h11``.

    Yields:
        The port.
    """
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip(http)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # A small receive buffer keeps what the kernel holds in flight close to what the app has pulled.
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, CHUNK)
    listener.bind(("127.0.0.1", 0))
    config = uvicorn.Config(
        topics_server.app,
        http=http,
        lifespan="off",
        log_config=None,  # uvicorn's own logging config would stop its logger reaching caplog
        access_log=False,
        timeout_keep_alive=1,
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    try:
        while not server.started:
            assert thread.is_alive(), "uvicorn stopped before it served"
            assert time.monotonic() < deadline, "uvicorn did not start within 5 s"
            time.sleep(0.01)
        yield listener.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(5)
        if thread.is_alive():
            server.force_exit = True
            thread.join(5)
        listener.close()


def _head(method: str, target: str, headers: Sequence[tuple[str, str]]) -> bytes:
    """Encode an HTTP/1.1 request line and its headers, up to the blank line before the body."""
    lines = [f"{method} {target} HTTP/1.1", "Host: topics", *(f"{name}: {value}" for name, value in headers)]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _read_response(conn: socket.socket) -> tuple[int, dict[str, str], bytes]:
    """Read one response off a socket.

    Args:
        conn: The connected socket.

    Returns:
        The status, the headers with names lower-cased, and the body, read to its Content-Length.
    """
    data = b""
    while b"\r\n\r\n" not in data:
        received = conn.recv(CHUNK)
        if not received:
            raise ConnectionError("the connection closed before a whole response head arrived")
        data += received
    head, body = data.split(b"\r\n\r\n", 1)
    status_line, *lines = head.decode("latin-1").split("\r\n")
    headers = {name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in lines)}
    length = int(headers.get("content-length", "0"))
    while len(body) < length:
        received = conn.recv(CHUNK)
        if not received:
            break
        body += received
    return int(status_line.split()[1]), headers, body[:length]


@HTTP_IMPLEMENTATIONS
def test_live_busy_answer_arrives_before_the_body(http: str) -> None:
    """A busy fit's 503 arrives after 1 KiB of a declared 8 MiB; the rest is drained, and the connection reused."""
    declared = 8 * 1024 * 1024
    with _serving(http) as port, _fit_running(), socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(_head("POST", "/topics/fit", [("Content-Type", BOUNDARY_TYPE), ("Content-Length", str(declared))]))
        conn.sendall(bytes(1024))
        status, headers, _ = _read_response(conn)
        assert status == 503
        assert headers["retry-after"] == "30"
        conn.sendall(bytes(declared - 1024))
        conn.sendall(_head("GET", "/health", []))
        assert _read_response(conn)[0] == 200


@HTTP_IMPLEMENTATIONS
def test_live_declared_oversize_is_refused_unread(http: str) -> None:
    """A declared length over the ceiling gets its 413 before a byte of the body is sent."""
    headers = [("Content-Type", BOUNDARY_TYPE), ("Content-Length", str(DEFAULT_CEILING + 1))]
    with _serving(http) as port, socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(_head("POST", "/topics/fit", headers))
        assert _read_response(conn)[0] == 413


@HTTP_IMPLEMENTATIONS
def test_live_chunked_oversize_is_cut_off(http: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunked body gets its 413 soon after it passes the ceiling, long before the sender would stop."""
    monkeypatch.setattr(topics_server, "MAX_POINTS", 16)
    frame = f"{CHUNK:x}\r\n".encode("ascii") + bytes(CHUNK) + b"\r\n"
    sent = 0
    with _serving(http) as port, socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CHUNK)
        conn.sendall(_head("POST", "/topics/fit", [("Content-Type", BOUNDARY_TYPE), ("Transfer-Encoding", "chunked")]))
        conn.sendall(f"{len(VECTORS_PART_HEAD):x}\r\n".encode("ascii") + VECTORS_PART_HEAD + b"\r\n")
        while sent < 32 * 1024 * 1024 and not select.select([conn], [], [], 0)[0]:
            conn.sendall(frame)
            sent += CHUNK
        status = _read_response(conn)[0]
    assert status == 413
    assert sent <= CEILING_AT_16_POINTS + 8 * 1024 * 1024


@HTTP_IMPLEMENTATIONS
def test_live_expect_continue_is_not_invited_when_busy(http: str) -> None:
    """A client that waits for ``100 Continue`` is answered with the 503 instead, and sends no body."""
    headers = [("Content-Type", BOUNDARY_TYPE), ("Content-Length", str(8 * 1024 * 1024)), ("Expect", "100-continue")]
    with _serving(http) as port, _fit_running(), socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(_head("POST", "/topics/fit", headers))
        assert _read_response(conn)[0] == 503


@HTTP_IMPLEMENTATIONS
def test_live_stalled_upload_frees_the_slot(http: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An upload that stops gets a 408 that closes its connection, and the next fit is admitted."""
    monkeypatch.setattr(topics_server, "_BODY_TIMEOUT_S", 0.5)
    headers = [("Content-Type", BOUNDARY_TYPE), ("Content-Length", str(8 * 1024 * 1024))]
    with _serving(http) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            conn.sendall(_head("POST", "/topics/fit", headers))
            conn.sendall(VECTORS_PART_HEAD)
            status, answer_headers, _ = _read_response(conn)
            closed = conn.recv(1) == b""
        assert status == 408
        assert answer_headers["connection"] == "close"
        assert closed
        _wait_for_free_slot()
        files = _files(_meta(), _texts(), _vectors().tobytes())
        assert httpx.post(f"http://127.0.0.1:{port}/topics/fit", files=files, timeout=5).status_code == 200


@HTTP_IMPLEMENTATIONS
def test_live_httpx_reads_the_busy_answer(http: str) -> None:
    """httpx, the transport of the docint client, reads the 503 that arrives while it is still uploading."""
    files = _files(_meta(), _texts(), bytes(8 * 1024 * 1024))
    with _serving(http) as port, _fit_running():
        response = httpx.post(f"http://127.0.0.1:{port}/topics/fit", files=files, timeout=5)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
