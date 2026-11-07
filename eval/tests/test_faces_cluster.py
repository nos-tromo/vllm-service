"""Tests for the per-video face-track clustering in ``src/faces_cluster.py``.

All data is synthetic: basis directions of a 128-dimensional space (SFace's
feature size) stand for different people, and small seeded noise around one
direction for further faces of the same person.
"""

import sys
from pathlib import Path

import numpy as np

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from faces_cluster import cluster_tracks, track_vector

DIM = 128
_RNG = np.random.default_rng(0)


def _unit(axis: int) -> np.ndarray:
    """Return the unit vector along one axis: a person of its own.

    Args:
        axis: The axis index.

    Returns:
        A unit vector orthogonal to every other axis's.
    """
    vector = np.zeros(DIM)
    vector[axis] = 1.0
    return vector


def _near(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector close to ``vector``: another face of the same person.

    Args:
        vector: A unit vector.

    Returns:
        ``vector`` plus small seeded noise, renormalized.
    """
    noisy = vector + _RNG.normal(0.0, 0.02, DIM)
    return noisy / np.linalg.norm(noisy)


def test_similar_tracks_merge_and_labels_follow_first_track() -> None:
    """Tracks of the same person share a label; labels count up from the first track."""
    a, b = _unit(0), _unit(1)
    labels = cluster_tracks(
        [a, _near(a), b, _near(b)],
        [frozenset({0}), frozenset({5}), frozenset({1}), frozenset({6})],
        threshold=0.42,
    )
    assert labels == [0, 0, 1, 1]


def test_labels_are_numbered_by_first_appearance() -> None:
    """The first track's cluster is 0 even when a later pair is the closer one."""
    a, b = _unit(0), _unit(1)
    labels = cluster_tracks(
        [b, a, a, _near(b)], [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})], threshold=0.42
    )
    assert labels == [0, 1, 1, 0]


def test_cannot_link_keeps_cooccurring_tracks_apart() -> None:
    """Two tracks on screen at the same time are two people, however alike."""
    a = _unit(0)
    labels = cluster_tracks([a, _near(a)], [frozenset({3, 4}), frozenset({4, 5})], threshold=0.42)
    assert labels == [0, 1]


def test_constraint_survives_merges() -> None:
    """A merged cluster inherits its members' constraints."""
    # t1 co-occurs with t2, so whichever of them t0 joins, the other stays out.
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a), _near(a)], [frozenset({0}), frozenset({9}), frozenset({9, 10})], threshold=0.42
    )
    assert labels[1] != labels[2]
    assert labels[0] in (labels[1], labels[2])


def test_constraints_follow_each_shared_sample() -> None:
    """Only tracks that share a sample are kept apart: t0~t1 and t1~t2 overlap, t0 and t2 do not."""
    a = _unit(0)
    labels = cluster_tracks([a, a, a], [frozenset({1, 2}), frozenset({2, 3}), frozenset({3, 4})], threshold=0.42)
    assert labels == [0, 1, 0]


def test_one_sample_shared_by_many_tracks_keeps_them_all_apart() -> None:
    """Fifty look-alike tracks on screen together are fifty people."""
    a = _unit(0)
    labels = cluster_tracks([a] * 50, [frozenset({7, index + 100}) for index in range(50)], threshold=0.42)
    assert labels == list(range(50))


def test_none_vector_is_unassigned() -> None:
    """A track without a decodable face gets ``-1`` and takes no label number."""
    assert cluster_tracks([None, _unit(0), None, _unit(1)], [frozenset({i}) for i in range(4)], threshold=0.42) == [
        -1,
        0,
        -1,
        1,
    ]


def test_empty_input_gives_no_labels() -> None:
    """No tracks, no labels."""
    assert cluster_tracks([], [], threshold=0.42) == []


def test_below_threshold_stays_split() -> None:
    """Two different people stay two clusters."""
    assert cluster_tracks([_unit(0), _unit(1)], [frozenset({0}), frozenset({1})], threshold=0.42) == [0, 1]


def test_average_linkage_not_single_linkage() -> None:
    """A cluster joins another only when the average similarity clears the threshold.

    ``m`` sits between two people: close enough to ``a`` to merge with it
    (single link 0.71), but the merged cluster's average to ``b`` is below
    the threshold although ``m`` alone would clear it.
    """
    a, b = _unit(0), _unit(1)
    m = (a + b) / np.linalg.norm(a + b)
    labels = cluster_tracks([a, m, b], [frozenset({0}), frozenset({1}), frozenset({2})], threshold=0.42)
    assert labels == [0, 0, 1]


def test_outlier_faces_do_not_drag_the_track() -> None:
    """A face of someone else in a track is dropped before averaging."""
    a, b = _unit(0), _unit(1)
    vec = track_vector(np.stack([a, _near(a), b]))
    assert vec is not None and float(vec @ a) > 0.95
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-9


def test_track_vector_of_no_faces_is_none() -> None:
    """A track with no face rows has no vector."""
    assert track_vector(np.zeros((0, DIM))) is None


def test_track_vector_skips_zero_rows() -> None:
    """An all-zero feature row (nothing to normalize) is skipped, not turned into NaN."""
    vec = track_vector(np.stack([np.zeros(DIM), _unit(0)]))
    assert vec is not None and np.allclose(vec, _unit(0))
    assert track_vector(np.zeros((2, DIM))) is None
