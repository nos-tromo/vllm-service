"""HTTP-contract tests for the topics fit server in ``src/topics_server.py``.

The fit is stubbed, so these tests need only numpy and FastAPI and run in the root eval
environment; the fit itself is covered by ``test_topics_fit.py``. All data is synthetic.
"""

import importlib.util
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

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

N = 40
DIM = 8
LOGGER = topics_server._log.name
META_FIELDS = ("n", "dim", "seed", "min_topic_size", "nr_topics", "top_n_words", "min_df", "stop_words")

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


def _part(value: object) -> bytes:
    """Serialize a JSON part as UTF-8; bytes pass through so a test can send malformed content."""
    if isinstance(value, bytes):
        return value
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _post(meta: object, texts: object, vectors: bytes | None) -> httpx.Response:
    """POST the three parts as file uploads, leaving out any given as ``None``.

    Args:
        meta: The ``meta`` object, or raw bytes.
        texts: The texts, or raw bytes.
        vectors: The raw vector bytes.

    Returns:
        The response.
    """
    files: dict[str, tuple[str, bytes, str]] = {}
    if meta is not None:
        files["meta"] = ("meta.json", _part(meta), "application/json")
    if texts is not None:
        files["texts"] = ("texts.json", _part(texts), "application/json")
    if vectors is not None:
        files["vectors"] = ("vectors.f32", vectors, "application/octet-stream")
    return client.post("/topics/fit", files=files)


def _post_valid(n: int = N, dim: int = DIM, **meta: object) -> httpx.Response:
    """POST a consistent request of ``n`` points, with any meta field replaced."""
    return _post(_meta(n=n, dim=dim, **meta), _texts(n), _vectors(n, dim).tobytes())


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
    """Fail a test that leaves the fit lock held, and free it so later tests are not refused."""
    yield
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
    ],
)
def test_meta_bounds_are_inclusive(field: str, value: object, fit_calls: FitCalls) -> None:
    """The limits themselves are accepted."""
    assert _post(_meta(**{field: value}), _texts(), _vectors().tobytes()).status_code == 200
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


def test_busy_returns_503(fit_calls: FitCalls) -> None:
    """A request during a fit is refused at once, never queued, and leaves the running fit's lock held."""
    pool = ThreadPoolExecutor(max_workers=1)
    topics_server._fit_lock.acquire()
    try:
        # A request that waited for the lock would hang; bound it so that fails fast instead.
        response = pool.submit(_post_valid).result(timeout=10)
        still_held = topics_server._fit_lock.locked()
    finally:
        topics_server._fit_lock.release()
        pool.shutdown()
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert still_held
    assert fit_calls == []
    assert _post_valid().status_code == 200


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


def test_health_reports_versions() -> None:
    """The health probe answers ok and lists the fit libraries' versions."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["versions"]) == {"bertopic", "umap-learn", "hdbscan", "scikit-learn", "numpy", "numba"}
    assert body == {"status": "ok", "versions": topics_server.topics_fit.versions()}


def test_max_points_defaults_to_150000() -> None:
    """A fit's peak memory sets the default: about 5.4 GB at 150000 points of 1024 dimensions."""
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
