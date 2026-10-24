"""Tests for the deterministic topic-model fit in ``src/topics_fit.py``.

The fit needs BERTopic, which only the separately locked topics project installs, so the
module is skipped anywhere else (including the root project's eval environment). Run it with
``uv run --project docker/topics --group dev pytest eval/tests/test_topics_fit.py``.

All data is synthetic: Gaussian blobs, each with its own invented vocabulary.
"""

import importlib.util
import os
import socket
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bertopic")

import hdbscan
import umap

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import topics_fit
from topics_fit import FitParams, FitResult, fit_topics, min_points, self_check

PREFIXES = ("alpha", "beta", "gamma")
PARAMS = FitParams(seed=7, min_topic_size=10, nr_topics=0, top_n_words=10, min_df=1, stop_words=())

Blobs = tuple[list[str], np.ndarray, list[int]]


def _make_blobs(
    n_per: int = 60, dim: int = 32, k: int = 3, seed: int = 7, prefixes: tuple[str, ...] = PREFIXES
) -> Blobs:
    """Build ``k`` separated Gaussian blobs whose texts draw on one invented vocabulary each.

    Args:
        n_per: Points per blob.
        dim: Vector dimension.
        k: Number of blobs.
        seed: Seed for both the vectors and the texts.
        prefixes: Word prefix per blob; blob ``i`` speaks ``prefixes[i]0`` … ``prefixes[i]9``.

    Returns:
        The texts, the L2-normalised float32 vectors, and each point's planted blob.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4.0, size=(k, dim))
    vectors = np.concatenate([centre + rng.normal(size=(n_per, dim)) for centre in centres])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    planted = [blob for blob in range(k) for _ in range(n_per)]
    texts = [" ".join(f"{prefixes[blob]}{word}" for word in rng.integers(0, 10, size=6)) for blob in planted]
    return texts, vectors.astype(np.float32), planted


@pytest.fixture
def blobs() -> Blobs:
    """The default planted set: three blobs of 60 points in 32 dimensions."""
    return _make_blobs()


def _majority(labels: list[int], planted: list[int], blob: int) -> tuple[int, float]:
    """Return the most common label among one blob's points and the share of the blob it holds.

    Args:
        labels: Fitted label per point.
        planted: Planted blob per point.
        blob: The blob to inspect.

    Returns:
        The label and the fraction of the blob's points carrying it.
    """
    members = [label for label, origin in zip(labels, planted, strict=True) if origin == blob]
    label, count = Counter(members).most_common(1)[0]
    return label, count / len(members)


def _blob_of(result: FitResult, planted: list[int], topic_id: int) -> int:
    """Return the planted blob that most of a topic's points come from.

    Args:
        result: The fit.
        planted: Planted blob per point.
        topic_id: The topic to inspect.

    Returns:
        The blob index.
    """
    members = [origin for label, origin in zip(result.labels, planted, strict=True) if label == topic_id]
    return Counter(members).most_common(1)[0][0]


def _keyword_words(result: FitResult) -> set[str]:
    """Return every keyword of every topic in a fit.

    Args:
        result: The fit.

    Returns:
        The set of keyword strings.
    """
    return {word for topic in result.topics for word, _ in topic.keywords}


def test_fit_is_deterministic(blobs: Blobs) -> None:
    """Fitting the same input twice gives the same labels and the same topics."""
    texts, vectors, _ = blobs
    first = fit_topics(texts, vectors, PARAMS)
    second = fit_topics(texts, vectors, PARAMS)
    assert first.labels == second.labels
    assert first.topics == second.topics


def test_recovers_planted_blobs(blobs: Blobs) -> None:
    """Each planted blob lands, mostly whole, in a topic of its own."""
    texts, vectors, planted = blobs
    result = fit_topics(texts, vectors, PARAMS)
    assert len(result.topics) >= 2
    majorities = [_majority(result.labels, planted, blob) for blob in range(3)]
    for label, share in majorities:
        assert label != -1
        assert share >= 0.8
    assert len({label for label, _ in majorities}) == 3
    assert [topic.id for topic in result.topics] == sorted({label for label in result.labels if label >= 0})
    assert [topic.size for topic in result.topics] == [result.labels.count(topic.id) for topic in result.topics]
    assert result.outliers == result.labels.count(-1)


@pytest.mark.parametrize("prefixes", [PREFIXES, ("größe", "привет", "κόσμος")], ids=["ascii", "non-ascii"])
def test_keywords_come_from_texts(prefixes: tuple[str, ...]) -> None:
    """A topic's top keyword is a word of its blob's vocabulary, whatever the script."""
    texts, vectors, planted = _make_blobs(prefixes=prefixes)
    result = fit_topics(texts, vectors, PARAMS)
    assert result.topics
    for topic in result.topics:
        assert topic.keywords[0][0].startswith(prefixes[_blob_of(result, planted, topic.id)])
        for _, score in topic.keywords:
            assert type(score) is float
            assert score == round(score, 6)


def test_min_points(blobs: Blobs) -> None:
    """The minimum is twice the topic size, never below 16, and smaller inputs are refused."""
    assert min_points(PARAMS) == 20
    assert min_points(replace(PARAMS, min_topic_size=5)) == 16
    texts, vectors, _ = blobs
    with pytest.raises(ValueError, match="at least 20"):
        fit_topics(texts[:15], vectors[:15], PARAMS)


def test_length_mismatch_raises(blobs: Blobs) -> None:
    """Texts and vectors must pair up one to one."""
    texts, vectors, _ = blobs
    with pytest.raises(ValueError, match="texts"):
        fit_topics(texts[:-1], vectors, PARAMS)


def test_identical_vectors_do_not_crash(blobs: Blobs, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical rows come back as outliers with no topics, without UMAP or HDBSCAN running."""

    def _must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("UMAP and HDBSCAN must not run on too few distinct vectors")

    monkeypatch.setattr(umap.UMAP, "fit_transform", _must_not_run)
    monkeypatch.setattr(hdbscan.HDBSCAN, "fit", _must_not_run)
    texts, vectors, _ = blobs
    result = fit_topics(texts[:80], np.repeat(vectors[:1], 80, axis=0), PARAMS)
    assert set(result.labels) <= {-1} | {topic.id for topic in result.topics}
    assert result.labels == [-1] * 80
    assert result.topics == []
    assert result.outliers == 80


@pytest.mark.parametrize("others", [0, 18], ids=["identical", "19-distinct"])
def test_mostly_identical_rows_yield_no_spurious_topics(blobs: Blobs, others: int) -> None:
    """Fewer distinct vectors than the minimum get no topics, not ones invented among duplicates."""
    texts, vectors, _ = blobs
    rows = np.concatenate([np.repeat(vectors[:1], 2000 - others, axis=0), vectors[1 : 1 + others]])
    result = fit_topics((texts * 12)[:2000], rows, PARAMS)
    assert result.labels == [-1] * 2000
    assert result.topics == []
    assert result.outliers == 2000


def test_signed_zeros_do_not_make_vectors_distinct(blobs: Blobs) -> None:
    """Rows differing only in the sign of a zero are one vector, as ``np.unique`` counts them."""
    texts, vectors, _ = blobs
    positive = vectors[:19].copy()
    positive[:, 0] = 0.0
    negative = positive.copy()
    negative[:, 0] = -0.0
    rows = np.concatenate([positive, negative] * 50)
    assert len(np.unique(rows, axis=0)) == 19
    result = fit_topics((texts * 11)[:1900], rows, PARAMS)
    assert result.topics == []
    assert result.outliers == 1900


def test_fit_runs_at_exactly_the_minimum_distinct_vectors(blobs: Blobs, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard stops inputs below the minimum, not at it: 20 distinct vectors still reach UMAP."""
    fitted: list[int] = []
    fit_transform = umap.UMAP.fit_transform

    def _spy(self: umap.UMAP, data: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        fitted.append(len(data))
        return fit_transform(self, data, *args, **kwargs)

    monkeypatch.setattr(umap.UMAP, "fit_transform", _spy)
    texts, vectors, _ = blobs
    fit_topics(texts[:20], vectors[:20], PARAMS)
    assert fitted == [20]


@pytest.mark.parametrize("blank", ["", " \n\t "], ids=["empty", "whitespace"])
def test_empty_texts_yield_empty_keywords(blobs: Blobs, blank: str) -> None:
    """Texts without a single word still cluster, and their topics carry no keywords."""
    texts, vectors, _ = blobs
    result = fit_topics([blank] * len(texts), vectors, PARAMS)
    assert result.topics
    assert all(topic.keywords == [] for topic in result.topics)
    assert result.labels == fit_topics(texts, vectors, PARAMS).labels


def test_topic_of_empty_texts_has_no_keywords(blobs: Blobs) -> None:
    """When only one blob's texts are empty, only that blob's topic loses its keywords."""
    texts, vectors, planted = blobs
    texts = ["" if blob == 0 else text for text, blob in zip(texts, planted, strict=True)]
    result = fit_topics(texts, vectors, PARAMS)
    silent, _ = _majority(result.labels, planted, 0)
    for topic in result.topics:
        assert (topic.keywords == []) == (topic.id == silent)


def test_single_word_vocabulary_yields_empty_keywords(blobs: Blobs) -> None:
    """A vocabulary of one word is too small for c-TF-IDF; the fit still returns its topics."""
    texts, vectors, _ = blobs
    result = fit_topics(["same"] * len(texts), vectors, PARAMS)
    assert result.topics
    assert all(topic.keywords == [] for topic in result.topics)


def test_min_df_counts_texts(blobs: Blobs) -> None:
    """``min_df`` counts the texts a word occurs in, not the topics, however often it repeats."""
    texts, vectors, _ = blobs
    texts = [texts[0] + " solo" * 100, *texts[1:]]
    assert "solo" in _keyword_words(fit_topics(texts, vectors, PARAMS))
    assert "solo" not in _keyword_words(fit_topics(texts, vectors, replace(PARAMS, min_df=2)))


def test_no_network_during_fit(blobs: Blobs, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit opens no network connection."""
    attempts: list[object] = []

    def _refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("network access is disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    texts, vectors, _ = blobs
    result = fit_topics(texts, vectors, PARAMS)
    assert result.topics
    assert attempts == []


def test_no_torch_or_sentence_transformers() -> None:
    """The locked environment holds neither torch, sentence-transformers nor plotly."""
    for module in ("torch", "sentence_transformers", "plotly"):
        assert importlib.util.find_spec(module) is None, module


def test_versions_and_runtime_report_the_stack() -> None:
    """The reported versions and runtime cover the fit's libraries and its thread settings."""
    versions = topics_fit.versions()
    assert list(versions) == ["bertopic", "umap-learn", "hdbscan", "scikit-learn", "numpy", "numba"]
    assert versions["bertopic"] == "0.17.4"
    runtime = topics_fit.runtime()
    assert set(runtime) == {"machine", "python", "threads"}
    thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    assert runtime["threads"] == {name: os.environ.get(name) for name in thread_vars}


def test_self_check_passes() -> None:
    """The startup self-check fits its synthetic set twice and finds the results identical."""
    assert self_check() is None
