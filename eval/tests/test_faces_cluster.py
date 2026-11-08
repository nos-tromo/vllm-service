"""Tests for the per-video face-track clustering in ``src/faces_cluster.py``.

All data is synthetic: basis directions of a 128-dimensional space (SFace's
feature size) stand for different people, and small seeded noise around one
direction for further faces of the same person.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from faces_cluster import cluster_tracks, group_tracks, track_vector

DIM = 128
GROUP = 0.42
JOIN = 0.25
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
        threshold=GROUP,
        join_threshold=JOIN,
    )
    assert labels == [0, 0, 1, 1]


def test_labels_are_numbered_by_first_appearance() -> None:
    """The first track's cluster is 0 even when a later pair is the closer one."""
    a, b = _unit(0), _unit(1)
    labels = cluster_tracks(
        [b, a, a, _near(b)],
        [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})],
        threshold=GROUP,
        join_threshold=JOIN,
    )
    assert labels == [0, 1, 1, 0]


def test_cannot_link_keeps_cooccurring_tracks_apart() -> None:
    """Two tracks on screen at the same time are two people, however alike."""
    a = _unit(0)
    labels = cluster_tracks([a, _near(a)], [frozenset({3, 4}), frozenset({4, 5})], threshold=GROUP, join_threshold=JOIN)
    assert labels == [0, 1]


def test_constraint_survives_merges() -> None:
    """A merged cluster inherits its members' constraints."""
    # t1 co-occurs with t2, so whichever of them t0 joins, the other stays out.
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a), _near(a)],
        [frozenset({0}), frozenset({9}), frozenset({9, 10})],
        threshold=GROUP,
        join_threshold=JOIN,
    )
    assert labels[1] != labels[2]
    assert labels[0] in (labels[1], labels[2])


def test_constraints_follow_each_shared_sample() -> None:
    """Only tracks that share a sample are kept apart: t0~t1 and t1~t2 overlap, t0 and t2 do not."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, a, a], [frozenset({1, 2}), frozenset({2, 3}), frozenset({3, 4})], threshold=GROUP, join_threshold=JOIN
    )
    assert labels == [0, 1, 0]


def test_one_sample_shared_by_many_tracks_keeps_them_all_apart() -> None:
    """Fifty look-alike tracks on screen together are fifty people."""
    a = _unit(0)
    labels = cluster_tracks(
        [a] * 50, [frozenset({7, index + 100}) for index in range(50)], threshold=GROUP, join_threshold=JOIN
    )
    assert labels == list(range(50))


def test_none_vector_is_unassigned() -> None:
    """A track without a decodable face gets ``-1`` and takes no label number."""
    assert cluster_tracks(
        [None, _unit(0), None, _unit(1)], [frozenset({i}) for i in range(4)], threshold=GROUP, join_threshold=JOIN
    ) == [
        -1,
        0,
        -1,
        1,
    ]


def test_empty_input_gives_no_labels() -> None:
    """No tracks, no labels."""
    assert cluster_tracks([], [], threshold=GROUP, join_threshold=JOIN) == []


def test_below_threshold_stays_split() -> None:
    """Two different people stay two clusters."""
    assert cluster_tracks(
        [_unit(0), _unit(1)], [frozenset({0}), frozenset({1})], threshold=GROUP, join_threshold=JOIN
    ) == [0, 1]


def test_average_linkage_not_single_linkage() -> None:
    """A cluster joins another only when the average similarity clears the threshold.

    ``m`` sits between two people: close enough to ``a`` to merge with it
    (single link 0.71), but the merged cluster's average to ``b`` is below
    the threshold although ``m`` alone would clear it.
    """
    a, b = _unit(0), _unit(1)
    m = (a + b) / np.linalg.norm(a + b)
    labels = cluster_tracks(
        [a, m, b], [frozenset({0}), frozenset({1}), frozenset({2, 3})], threshold=GROUP, join_threshold=JOIN
    )
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


def _toward(vector: np.ndarray, cosine: float, axis: int) -> np.ndarray:
    """Return a unit vector at a given cosine to ``vector``, the rest along an unused axis.

    Args:
        vector: A unit vector.
        cosine: The cosine the result has to ``vector``.
        axis: An axis no person uses, for the remainder.

    Returns:
        A unit vector: a less reliable face (a profile, say) of the same person.
    """
    return cosine * vector + np.sqrt(1.0 - cosine**2) * _unit(axis)


def test_weak_track_joins_the_person_it_matches() -> None:
    """A track without a good face joins the one person it is close enough to."""
    a, b = _unit(0), _unit(1)
    labels = cluster_tracks(
        [a, _near(a), b, _toward(a, 0.7, 100)],
        [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, True, True, False],
    )
    assert labels == [0, 0, 1, 0]


def test_weak_track_below_the_join_threshold_stays_unassigned() -> None:
    """A weak track close to nobody is left on its own."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _toward(a, 0.2, 100)],
        [frozenset({0, 1}), frozenset({2})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, False],
    )
    assert labels == [0, -1]


def test_weak_track_between_the_thresholds_joins() -> None:
    """A weak track too far to cluster with a person (0.3 < 0.42) still joins it."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _toward(a, 0.3, 100)],
        [frozenset({0, 1}), frozenset({2})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, False],
    )
    assert labels == [0, 0]


def test_weak_track_never_bridges_two_people() -> None:
    """A weak track close to two people joins neither and does not merge them.

    ``a`` and ``b`` are two people (cosine 0.3, below the threshold); ``w`` is
    0.81 to each. Clustered as an equal, it pulls them into one cluster.
    """
    a = _unit(0)
    b = _toward(a, 0.3, 1)
    w = (a + b) / np.linalg.norm(a + b)
    samples = [frozenset({0, 10}), frozenset({1, 11}), frozenset({2})]
    assert cluster_tracks([a, b, w], samples, threshold=GROUP, join_threshold=JOIN) == [0, 0, 0]
    assert cluster_tracks([a, b, w], samples, threshold=GROUP, join_threshold=JOIN, anchors=[True, True, False]) == [
        0,
        1,
        -1,
    ]


def test_weak_track_on_screen_with_the_person_does_not_join_it() -> None:
    """A weak track sharing a sample with a person is someone else."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a)], [frozenset({0, 1}), frozenset({1})], threshold=GROUP, join_threshold=JOIN, anchors=[True, False]
    )
    assert labels == [0, -1]


def test_weak_track_joins_the_only_person_it_may_join() -> None:
    """Of two people above the threshold, one on screen with the track is not a candidate."""
    a = _unit(0)
    b = _toward(a, 0.3, 1)
    w = (a + b) / np.linalg.norm(a + b)
    labels = cluster_tracks(
        [a, b, w],
        [frozenset({0, 9}), frozenset({1, 2}), frozenset({9})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, True, False],
    )
    assert labels == [0, 1, 1]


def test_two_weak_tracks_on_screen_together_do_not_both_join_one_person() -> None:
    """The closer of two co-occurring weak tracks joins; the other is then someone else."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _toward(a, 0.6, 100), _toward(a, 0.9, 101)],
        [frozenset({0, 1}), frozenset({5}), frozenset({5})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, False, False],
    )
    assert labels == [0, -1, 0]


def test_weak_tracks_alone_form_no_person() -> None:
    """Without an anchor there is no person to join, however alike the weak tracks are."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a)], [frozenset({0}), frozenset({1})], threshold=GROUP, join_threshold=JOIN, anchors=[False, False]
    )
    assert labels == [-1, -1]


def test_weak_track_takes_its_persons_number() -> None:
    """People are numbered by their first anchor; a weak track earlier in the list takes no number."""
    a, b = _unit(0), _unit(1)
    labels = cluster_tracks(
        [_toward(a, 0.7, 100), b, a],
        [frozenset({0}), frozenset({1}), frozenset({2, 3})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[False, True, True],
    )
    assert labels == [1, 0, 1]


def test_all_anchors_cluster_as_before() -> None:
    """Every track an anchor is the same as passing no roles."""
    a, b = _unit(0), _unit(1)
    vectors = [a, _near(a), b, None, _near(b), _toward(a, 0.5, 100)]
    samples = [frozenset({i}) for i in range(5)] + [frozenset({1, 2})]
    assert cluster_tracks(vectors, samples, threshold=GROUP, join_threshold=JOIN) == cluster_tracks(
        vectors, samples, threshold=GROUP, join_threshold=JOIN, anchors=[True] * len(vectors)
    )


def test_anchors_must_match_the_tracks() -> None:
    """One role per track."""
    with pytest.raises(ValueError, match="anchors"):
        cluster_tracks([_unit(0)], [frozenset({0})], threshold=GROUP, join_threshold=JOIN, anchors=[True, False])


def _mix(cosines: dict[int, float], rest: int) -> np.ndarray:
    """Return a unit vector with given cosines to some axes, the rest along an unused one.

    Args:
        cosines: Axis to the cosine the result has to that axis's unit vector.
        rest: An axis no person uses, for the remainder.

    Returns:
        The unit vector.
    """
    vector = np.zeros(DIM)
    for axis, cosine in cosines.items():
        vector[axis] = cosine
    vector[rest] = np.sqrt(1.0 - float(np.sum(vector**2)))
    return vector


def test_anchor_seen_once_joins_a_person_between_the_thresholds() -> None:
    """A small or blurred face that founded a person of its own joins the person it resembles."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a), _toward(a, 0.35, 100)],
        [frozenset({0}), frozenset({1}), frozenset({2})],
        threshold=GROUP,
        join_threshold=JOIN,
    )
    assert labels == [0, 0, 0]


def test_anchor_seen_once_below_the_join_threshold_stays_a_person() -> None:
    """A face unlike every person stays a person of its own."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a), _toward(a, 0.2, 100)],
        [frozenset({0}), frozenset({1}), frozenset({2})],
        threshold=GROUP,
        join_threshold=JOIN,
    )
    assert labels == [0, 0, 1]


@pytest.mark.parametrize(("to_b", "expected"), [(0.30, [0, 1, 2]), (0.20, [0, 1, 0])], ids=["no-lead", "clear-lead"])
def test_a_piece_joins_only_with_a_clear_lead(to_b: float, expected: list[int]) -> None:
    """Close to two people, a piece joins the closer only when it leads by the margin."""
    a, b = _unit(0), _unit(1)
    x = _mix({0: 0.35, 1: to_b}, 100)
    labels = cluster_tracks(
        [a, b, x], [frozenset({0, 1}), frozenset({2, 3}), frozenset({4})], threshold=GROUP, join_threshold=JOIN
    )
    assert labels == expected


def test_closeness_counts_the_persons_closest_anchors() -> None:
    """A profile matches a person's profile views, not the mean of every view.

    The person has three frontal anchors ``f`` and two profile anchors ``p``
    (cosine 0.45, so all five cluster). The weak track is 0.4 to the frontal
    ones and 0 to the profiles: 0.24 averaged over all five, below the join
    threshold, but 0.4 over its three closest.
    """
    f = _unit(0)
    p = _toward(f, 0.45, 1)
    beta = -0.45 * 0.4 / (1 - 0.45**2)
    alpha = 0.4 - 0.45 * beta
    c = alpha * f + beta * p
    c = c + np.sqrt(1.0 - float(c @ c)) * _unit(100)
    assert float(np.mean([c @ f] * 3 + [c @ p] * 2)) < JOIN
    labels = cluster_tracks(
        [f, f, f, p, p, c],
        [frozenset({index}) for index in range(6)],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True] * 5 + [False],
    )
    assert labels == [0] * 6


def test_established_people_between_the_thresholds_stay_apart() -> None:
    """Joining never merges two people each seen more than once."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _toward(a, 0.35, 1)], [frozenset({0, 1}), frozenset({2, 3})], threshold=GROUP, join_threshold=JOIN
    )
    assert labels == [0, 1]


def test_a_person_seen_once_takes_no_one() -> None:
    """Only a person seen in at least two samples may be joined."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _near(a)], [frozenset({0}), frozenset({1})], threshold=GROUP, join_threshold=JOIN, anchors=[True, False]
    )
    assert labels == [0, -1]


def test_a_piece_on_screen_with_a_joined_one_does_not_join_too() -> None:
    """Once a closer piece joined at a sample, another piece at that sample is someone else."""
    a = _unit(0)
    labels = cluster_tracks(
        [a, _toward(a, 0.9, 100), _toward(a, 0.35, 101)],
        [frozenset({0, 1}), frozenset({5}), frozenset({5})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, False, True],
    )
    assert labels == [0, 0, 1]


def test_a_joined_piece_draws_in_no_one() -> None:
    """A piece close only to a joined piece, not to the person's anchors, stays on its own."""
    a = _unit(0)
    w1 = _toward(a, 0.5, 100)
    w2 = _mix({0: 0.1, 100: (0.9 - 0.05) / np.sqrt(0.75)}, 101)
    assert float(w1 @ w2) == pytest.approx(0.9)
    labels = cluster_tracks(
        [a, w1, w2],
        [frozenset({0, 1}), frozenset({2}), frozenset({3})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, False, False],
    )
    assert labels == [0, 0, -1]


def test_labels_stay_contiguous_after_a_join() -> None:
    """A joined anchor's former person leaves no gap; the first anchor still numbers."""
    a = _unit(0)
    labels = cluster_tracks(
        [_toward(a, 0.35, 100), _unit(1), a, _near(a)],
        [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})],
        threshold=GROUP,
        join_threshold=JOIN,
    )
    assert labels == [0, 1, 0, 0]


def test_group_tracks_counts_the_pieces_and_the_joins() -> None:
    """Weak tracks and anchors seen once are pieces; the counts say how many joined."""
    a = _unit(0)
    grouping = group_tracks(
        [a, _toward(a, 0.9, 100), _unit(1), _toward(a, 0.35, 101), None],
        [frozenset({0, 1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})],
        threshold=GROUP,
        join_threshold=JOIN,
        anchors=[True, False, False, True, False],
    )
    assert grouping.labels == [0, 0, -1, 0, -1]
    assert (grouping.pieces, grouping.joined) == (3, 2)
