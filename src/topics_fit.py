"""Deterministic topic-model fit over caller-supplied vectors.

The fit is BERTopic (UMAP -> HDBSCAN -> c-TF-IDF) with no embedding model: the vectors
arrive with the request, so nothing is embedded or downloaded. ``topics_server.py`` wraps
this module in the HTTP contract.

bertopic, umap, hdbscan and scikit-learn are imported inside the functions that use them,
so this module imports with only the standard library and numpy.

Determinism: the same texts, vectors, settings, image and CPU class give the same labels
and keywords. UMAP is seeded, which makes it single-threaded; HDBSCAN computes core
distances on one thread; and the image pins the thread variables in ``THREAD_VARS`` to 1,
which :func:`runtime` reports.

Text handling: the c-TF-IDF vocabulary is fitted here, on the individual texts, and handed
to BERTopic fixed. Left to BERTopic, the vectorizer is fitted after clustering to each
topic's joined texts, so ``min_df`` would count topics rather than texts, and a setting
above the number of topics found would empty the vocabulary mid-fit. A vocabulary of fewer
than two words (BERTopic's c-TF-IDF raises on one) is replaced by a two-word sentinel,
which keeps the clustering untouched and leaves every topic without keywords.

Degenerate input: when fewer than :func:`min_points` of the vectors are distinct, every
input is returned as an outlier without running UMAP or HDBSCAN, which would otherwise
invent clusters among the duplicates.
"""

from __future__ import annotations

import os
import platform
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import SupportsFloat

import numpy as np

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")

_PACKAGES = ("bertopic", "umap-learn", "hdbscan", "scikit-learn", "numpy", "numba")

# Plain lowercase ASCII, so tokenizing leaves both words exactly as written.
_SENTINEL_WORDS = ("topicsemptyone", "topicsemptytwo")


@dataclass(frozen=True)
class FitParams:
    """Settings for one fit.

    Attributes:
        seed: UMAP's random state.
        min_topic_size: HDBSCAN's ``min_cluster_size``.
        nr_topics: BERTopic's topic-reduction target, or 0 for no reduction. BERTopic
            counts the outlier class as one of these topics.
        top_n_words: Keywords kept per topic.
        min_df: The fewest texts a word must occur in to become a keyword.
        stop_words: Words never used as keywords.
    """

    seed: int
    min_topic_size: int
    nr_topics: int
    top_n_words: int
    min_df: int
    stop_words: tuple[str, ...]


@dataclass(frozen=True)
class TopicInfo:
    """One topic.

    Attributes:
        id: The topic id, ``>= 0``.
        size: How many inputs carry the id.
        keywords: ``(word, c-TF-IDF score)`` pairs, best first, scores rounded to 6 places.
    """

    id: int
    size: int
    keywords: list[tuple[str, float]]


@dataclass(frozen=True)
class FitResult:
    """The outcome of one fit.

    Attributes:
        labels: The topic id of each input, in input order; ``-1`` marks an outlier.
        topics: The topics with an id ``>= 0``, in id order.
        outliers: How many inputs are labelled ``-1``.
        fit_s: Wall time of the fit, in seconds.
    """

    labels: list[int]
    topics: list[TopicInfo]
    outliers: int
    fit_s: float


def min_points(params: FitParams) -> int:
    """Return the fewest inputs a fit accepts with these settings.

    UMAP needs more points than its 15 neighbours, and HDBSCAN needs room for two clusters
    of ``min_topic_size``.

    Args:
        params: The fit settings.

    Returns:
        The minimum number of inputs.
    """
    return max(16, 2 * params.min_topic_size)


def fit_topics(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
    """Fit BERTopic to precomputed vectors and return the labels and keywords.

    Inputs with fewer distinct vectors than :func:`min_points` come back as all outliers,
    with no topics.

    Args:
        texts: One text per vector, the c-TF-IDF input.
        vectors: An ``n x dim`` array; row ``i`` belongs to ``texts[i]``.
        params: The fit settings.

    Returns:
        The labels, the topics and the fit's wall time.

    Raises:
        ValueError: If ``texts`` and ``vectors`` differ in length, or hold fewer than
            :func:`min_points` inputs.
    """
    if len(texts) != len(vectors):
        raise ValueError(f"got {len(texts)} texts for {len(vectors)} vectors")
    minimum = min_points(params)
    if len(vectors) < minimum:
        raise ValueError(
            f"need at least {minimum} points for min_topic_size={params.min_topic_size}, got {len(vectors)}"
        )

    import hdbscan
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    started = time.perf_counter()
    if _distinct_rows(vectors, minimum) < minimum:
        n = len(vectors)
        return FitResult(labels=[-1] * n, topics=[], outliers=n, fit_s=time.perf_counter() - started)

    docs = list(texts)
    vocabulary = _vocabulary(docs, params)
    dropped: frozenset[str] = frozenset()
    if vocabulary is None:
        docs = [" ".join(_SENTINEL_WORDS)] * len(docs)
        vocabulary = {word: index for index, word in enumerate(sorted(_SENTINEL_WORDS))}
        dropped = frozenset(_SENTINEL_WORDS)

    model = BERTopic(
        embedding_model=None,
        # The default, "english", strips every character outside [A-Za-z0-9 ] before c-TF-IDF.
        language=None,
        # n_jobs=1 is what the seed forces anyway; spelling it out avoids UMAP's warning.
        umap_model=UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=params.seed, n_jobs=1
        ),
        hdbscan_model=hdbscan.HDBSCAN(
            min_cluster_size=params.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=1,
            prediction_data=False,
        ),
        vectorizer_model=CountVectorizer(vocabulary=vocabulary),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        nr_topics=params.nr_topics or None,
        top_n_words=params.top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )
    predicted, _ = model.fit_transform(docs, embeddings=vectors)
    labels = [int(label) for label in predicted]
    sizes = Counter(labels)
    topics = [
        TopicInfo(id=topic_id, size=sizes[topic_id], keywords=_keywords(model.get_topic(topic_id) or [], dropped))
        for topic_id in sorted(sizes)
        if topic_id >= 0
    ]
    return FitResult(labels=labels, topics=topics, outliers=sizes[-1], fit_s=time.perf_counter() - started)


def _distinct_rows(vectors: np.ndarray, enough: int) -> int:
    """Count the distinct rows of ``vectors``, stopping once ``enough`` have been seen.

    The count equals ``np.unique(vectors, axis=0)``'s for finite input. Stopping early means
    ordinary input costs a few rows, and an all-duplicate one a single pass, whereas
    ``np.unique`` sorts every row.

    Args:
        vectors: The input rows.
        enough: The count at which to stop.

    Returns:
        The number of distinct rows, capped at ``enough``.
    """
    seen: set[bytes] = set()
    for row in vectors:
        seen.add((row + 0.0).tobytes())  # + 0.0 turns -0.0 into 0.0, which np.unique counts as equal
        if len(seen) >= enough:
            break
    return len(seen)


def _vocabulary(docs: list[str], params: FitParams) -> dict[str, int] | None:
    """Fit the c-TF-IDF vocabulary to the individual texts.

    Args:
        docs: The texts.
        params: The fit settings; ``stop_words`` and ``min_df`` apply.

    Returns:
        Word -> column index, or ``None`` when fewer than two words survive.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words=list(params.stop_words) or None, min_df=params.min_df)
    try:
        vectorizer.fit(docs)
    except ValueError:  # no word survives the stop words and min_df
        return None
    vocabulary = {str(word): int(index) for word, index in vectorizer.vocabulary_.items()}
    return vocabulary if len(vocabulary) >= 2 else None


def _keywords(ranked: Sequence[tuple[str, SupportsFloat]], dropped: frozenset[str]) -> list[tuple[str, float]]:
    """Round BERTopic's ranked words, dropping its blank padding and the sentinel.

    Args:
        ranked: ``(word, score)`` pairs as BERTopic returns them, best first; the scores are
            numpy scalars.
        dropped: Words to leave out.

    Returns:
        The keywords, scores as plain floats rounded to 6 decimal places.
    """
    return [(word, round(float(score), 6)) for word, score in ranked if word and word not in dropped]


def versions() -> dict[str, str]:
    """Report the installed versions of the fit's libraries.

    Returns:
        Distribution name -> version, with ``"absent"`` for one that is not installed.
    """
    found: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = "absent"
    return found


def runtime() -> dict[str, object]:
    """Report what, besides the input, a fit's result depends on.

    Returns:
        The CPU architecture, the Python version and the thread variables (``None`` when
        unset).
    """
    return {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARS},
    }


def self_check() -> None:
    """Fit a fixed synthetic set twice and require identical results.

    Raises:
        RuntimeError: If the two fits disagree on the labels or the keywords.
    """
    texts, vectors = _synthetic_blobs()
    params = FitParams(seed=7, min_topic_size=10, nr_topics=0, top_n_words=10, min_df=1, stop_words=())
    first = fit_topics(texts, vectors, params)
    second = fit_topics(texts, vectors, params)
    if first.labels != second.labels or first.topics != second.topics:
        raise RuntimeError("topic-model fit is not deterministic: two fits of the self-check set differ")


def _synthetic_blobs() -> tuple[list[str], np.ndarray]:
    """Build the self-check set: three separated blobs of 60 points with a vocabulary each.

    Returns:
        The texts and the L2-normalised float32 vectors.
    """
    rng = np.random.default_rng(7)
    prefixes = ("alpha", "beta", "gamma")
    centres = rng.normal(scale=4.0, size=(len(prefixes), 32))
    vectors = np.concatenate([centre + rng.normal(size=(60, 32)) for centre in centres])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    texts = [
        " ".join(f"{prefix}{word}" for word in rng.integers(0, 10, size=6)) for prefix in prefixes for _ in range(60)
    ]
    return texts, vectors.astype(np.float32)
