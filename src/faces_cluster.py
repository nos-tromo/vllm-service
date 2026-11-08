"""Per-video grouping of face tracks by their SFace features.

``faces_server.py`` wraps this module in the ``/faces/group`` contract. Pure
NumPy, so it imports and tests without OpenCV.

A track is a run of detections the caller already linked within one shot; it
arrives as up to three faces. :func:`track_vector` turns a track's L2-normalized
face features into one vector, dropping faces far from the track's medoid (a
track that runs on past a missed cut holds someone else's face).
:func:`cluster_tracks` then groups the tracks by exact average-linkage
agglomerative clustering on cosine similarity, with one constraint: two tracks
that share a sample index were on screen at the same time, so they are two
people and never end up in one cluster, however alike they look.

Tracks come in two roles. An anchor track holds at least one face the caller
judged good (confident, large, roughly frontal); only anchors are clustered,
so only they make up people. A weak track holds no such face, a profile say,
whose features are less reliable.

Grouping runs in two stages. First the anchors are clustered as above; a
cluster seen in at least :data:`MIN_PERSON_SAMPLES` samples is an established
person. Then every other piece (a weak track, or an anchor cluster seen only
once, often a small or blurred face) may join one established person
(:func:`_join`). Its closeness to a person is the mean of its
:data:`JOIN_TOP_K` highest cosines to the person's anchor tracks, so a person
filmed from many angles is matched by the views that resemble the piece rather
than diluted by the others. It joins the closest person it was never on screen
with when that closeness is at least the join threshold and leads the next
such person by :data:`JOIN_MARGIN`. A piece never founds a person by joining
and never links two, joined pieces do not count as anchors (so nothing
chains), and established people are never merged here. A weak track that
joins no one stays on its own; so does an anchor cluster, as a person of its
own.

Nothing here is kept between calls; the vectors live only as long as the
request that computed them.
"""

import math
from dataclasses import dataclass

import numpy as np

#: Faces whose cosine to their track's medoid is below this are dropped.
OUTLIER_COS = 0.3
#: Samples an anchor cluster needs to be an established person others may join.
MIN_PERSON_SAMPLES = 2
#: How many of a person's anchor tracks a joining piece is compared with: its closest.
JOIN_TOP_K = 3
#: How far the closest person must lead the next for a piece to join it.
JOIN_MARGIN = 0.10


def _normalized_rows(embeddings: np.ndarray) -> np.ndarray:
    """Return the rows of ``embeddings`` that can be normalized, normalized.

    Args:
        embeddings: One feature per row.

    Returns:
        The finite, non-zero rows scaled to unit length, in order.

    Raises:
        ValueError: If ``embeddings`` is not two-dimensional.
    """
    rows = np.asarray(embeddings, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("embeddings must be a 2-D array, one face per row")
    norms = np.linalg.norm(rows, axis=1)
    usable = np.isfinite(norms) & (norms > 0)
    return rows[usable] / norms[usable, None]


def track_vector(embeddings: np.ndarray, *, outlier_cos: float = OUTLIER_COS) -> np.ndarray | None:
    """Return one unit vector for a track from its faces' features.

    The medoid is the face with the highest mean cosine to the others (the
    first on a tie); faces whose cosine to it is below ``outlier_cos`` are
    dropped, and the mean of the rest is renormalized.

    Args:
        embeddings: The track's face features, one per row.
        outlier_cos: The cosine to the medoid below which a face is dropped.

    Returns:
        The track's unit vector, or ``None`` when no row can be normalized.

    Raises:
        ValueError: If ``embeddings`` is not two-dimensional.
    """
    rows = _normalized_rows(embeddings)
    if len(rows) == 0:
        return None
    similarity = rows @ rows.T
    count = len(rows)
    mean_to_others = (similarity.sum(axis=1) - np.diag(similarity)) / max(count - 1, 1)
    medoid = int(np.argmax(mean_to_others))
    kept = similarity[medoid] >= outlier_cos
    kept[medoid] = True
    mean = rows[kept].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm == 0.0:
        return rows[medoid]
    return mean / norm


def _cooccurring(assigned: list[int], samples: list[frozenset[int]]) -> list[np.ndarray]:
    """Return each distinct set of tracks that share a sample index.

    Built from an index of sample to tracks, so the cost follows the number of
    sample indices sent, not the number of track pairs times their sizes.

    Args:
        assigned: The tracks to cluster, as indices into ``samples``.
        samples: The sample indices of every track.

    Returns:
        The positions in ``assigned`` of each group of two or more tracks on
        screen together, each group once.
    """
    on_screen: dict[int, list[int]] = {}
    for position, index in enumerate(assigned):
        for sample in samples[index]:
            on_screen.setdefault(sample, []).append(position)
    groups = {tuple(members) for members in on_screen.values() if len(members) > 1}
    return [np.asarray(members) for members in sorted(groups)]


def _average_linkage(
    present: dict[int, np.ndarray], samples: list[frozenset[int]], seeds: list[int], threshold: float
) -> np.ndarray:
    """Cluster the seed tracks by average linkage under cannot-link constraints.

    Clusters start as single tracks. Each step merges the pair with the highest
    average pairwise cosine, while that average is at least ``threshold`` and
    no track of one shares a sample index with a track of the other. The
    similarity matrix carries ``-inf`` for every pair that may not merge;
    the average-linkage update keeps that, since a merged cluster may not join
    anything either of its parts could not.

    Args:
        present: The unit vector of every track that has one, by track index.
        samples: The sample indices of every track.
        seeds: The tracks to cluster, as indices; each has a vector.
        threshold: The lowest average cosine at which two clusters merge.

    Returns:
        Per position in ``seeds``, the position of the seed whose cluster it ended in.
    """
    count = len(seeds)
    matrix = np.stack([present[index] for index in seeds])
    similarity = matrix @ matrix.T
    for members in _cooccurring(seeds, samples):
        similarity[np.ix_(members, members)] = -np.inf
    np.fill_diagonal(similarity, -np.inf)

    sizes = np.ones(count)
    owner = np.arange(count)
    for _ in range(count - 1):
        flat = int(np.argmax(similarity))
        keep, drop = divmod(flat, count)
        if not similarity[keep, drop] >= threshold:
            break
        keep, drop = min(keep, drop), max(keep, drop)
        merged = (sizes[keep] * similarity[keep] + sizes[drop] * similarity[drop]) / (sizes[keep] + sizes[drop])
        similarity[keep, :] = merged
        similarity[:, keep] = merged
        similarity[drop, :] = -np.inf
        similarity[:, drop] = -np.inf
        similarity[keep, keep] = -np.inf
        sizes[keep] += sizes[drop]
        owner[owner == drop] = keep
    return owner


def _closeness(piece: np.ndarray, person: np.ndarray) -> float:
    """Return how close a piece is to a person: the mean of its top cosines.

    Args:
        piece: The piece's track vectors, one per row.
        person: The person's anchor track vectors, one per row.

    Returns:
        The mean of the :data:`JOIN_TOP_K` highest cosines between a track of
        the piece and an anchor of the person (all of them when there are fewer).
    """
    cosines = np.sort((piece @ person.T).reshape(-1))[::-1]
    return float(np.mean(cosines[:JOIN_TOP_K]))


def _join(
    present: dict[int, np.ndarray],
    samples: list[frozenset[int]],
    people: list[list[int]],
    pieces: list[list[int]],
    join_threshold: float,
) -> dict[int, int]:
    """Decide which pieces join which established person.

    A person is allowed for a piece when no track of the person, joined pieces
    included, shares a sample index with it. The piece joins its closest
    allowed person (:func:`_closeness`) when that closeness is at least
    ``join_threshold`` and at least :data:`JOIN_MARGIN` above the next allowed
    person's. Pieces go closest first, so of two pieces on screen together the
    closer one joins and the other is then someone else. Closeness counts the
    person's anchors only, so a joined piece never draws in the next.

    Args:
        present: The unit vector of every track that has one, by track index.
        samples: The sample indices of every track.
        people: The established people, each as its anchor tracks.
        pieces: The pieces that may join, each as its tracks (all with a vector).
        join_threshold: The lowest closeness at which a piece joins a person.

    Returns:
        For each piece that joins, its position in ``pieces`` mapped to the
        person's position in ``people``.
    """
    if not people or not pieces:
        return {}
    anchors_of = [np.stack([present[index] for index in members]) for members in people]
    on_screen = [set().union(*(samples[index] for index in members)) for members in people]
    seen = [frozenset().union(*(samples[index] for index in members)) for members in pieces]
    closeness = [
        [_closeness(np.stack([present[index] for index in members]), person) for person in anchors_of]
        for members in pieces
    ]
    joined: dict[int, int] = {}
    for position in sorted(range(len(pieces)), key=lambda piece: (-max(closeness[piece]), pieces[piece][0])):
        allowed = sorted(
            (
                (value, person)
                for person, value in enumerate(closeness[position])
                if not on_screen[person] & seen[position]
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not allowed:
            continue
        best, person = allowed[0]
        runner_up = allowed[1][0] if len(allowed) > 1 else -math.inf
        if best >= join_threshold and best - runner_up >= JOIN_MARGIN:
            joined[position] = person
            on_screen[person] |= seen[position]
    return joined


@dataclass(frozen=True)
class Grouping:
    """The outcome of grouping one video's tracks.

    Attributes:
        labels: One label per track, as :func:`cluster_tracks` returns them.
        pieces: How many pieces could join a person: weak tracks with a vector
            and anchor clusters seen in fewer than :data:`MIN_PERSON_SAMPLES` samples.
        joined: How many of them joined one.
    """

    labels: list[int]
    pieces: int
    joined: int


def group_tracks(
    vectors: list[np.ndarray | None],
    samples: list[frozenset[int]],
    *,
    threshold: float,
    join_threshold: float,
    anchors: list[bool] | None = None,
) -> Grouping:
    """Group tracks into people and say how many pieces joined one.

    The anchor tracks are clustered by average linkage under cannot-link
    constraints (:func:`_average_linkage`); a cluster seen in at least
    :data:`MIN_PERSON_SAMPLES` samples is an established person. The weak
    tracks and the smaller anchor clusters may then each join one (:func:`_join`).

    Args:
        vectors: One unit vector per track, or ``None`` for a track without one.
        samples: The sample indices each track was on screen at, one set per track.
        threshold: The lowest average cosine at which two anchor clusters merge.
        join_threshold: The lowest closeness at which a piece joins a person.
        anchors: Per track, whether it is an anchor; ``None`` makes every
            track one.

    Returns:
        The labels: people numbered ``0, 1, …`` in the order of their first
        anchor track (a joined anchor counts), ``-1`` for a track without a
        vector and for a weak track that joined no one; and the piece counts.

    Raises:
        ValueError: If ``vectors``, ``samples`` and ``anchors`` differ in length.
    """
    if len(vectors) != len(samples):
        raise ValueError("vectors and samples must have one entry per track")
    roles = [True] * len(vectors) if anchors is None else list(anchors)
    if len(roles) != len(vectors):
        raise ValueError("anchors must have one entry per track")
    present = {
        index: np.asarray(vector, dtype=np.float64) for index, vector in enumerate(vectors) if vector is not None
    }
    seeds = [index for index in present if roles[index]]
    clusters: dict[int, list[int]] = {}
    if seeds:
        owner = _average_linkage(present, samples, seeds, threshold)
        for position, index in enumerate(seeds):
            clusters.setdefault(int(owner[position]), []).append(index)
    people: list[list[int]] = []
    pieces: list[list[int]] = []
    for members in clusters.values():
        sample_count = sum(len(samples[index]) for index in members)
        (people if sample_count >= MIN_PERSON_SAMPLES else pieces).append(members)
    pieces.extend([index] for index in present if not roles[index])
    joined = _join(present, samples, people, pieces, join_threshold)

    groups = [list(members) for members in people]
    for position, members in enumerate(pieces):
        if position in joined:
            groups[joined[position]].extend(members)
        elif roles[members[0]]:
            groups.append(list(members))
    labels = [-1] * len(vectors)
    for label, members in enumerate(sorted(groups, key=lambda group: min(i for i in group if roles[i]))):
        for index in members:
            labels[index] = label
    return Grouping(labels=labels, pieces=len(pieces), joined=len(joined))


def cluster_tracks(
    vectors: list[np.ndarray | None],
    samples: list[frozenset[int]],
    *,
    threshold: float,
    join_threshold: float,
    anchors: list[bool] | None = None,
) -> list[int]:
    """Group tracks into people: anchors by clustering, the rest by joining one.

    Args:
        vectors: One unit vector per track, or ``None`` for a track without one.
        samples: The sample indices each track was on screen at, one set per track.
        threshold: The lowest average cosine at which two anchor clusters merge.
        join_threshold: The lowest closeness at which a piece joins a person.
        anchors: Per track, whether it is an anchor; ``None`` makes every
            track one.

    Returns:
        One label per track (:func:`group_tracks`).

    Raises:
        ValueError: If ``vectors``, ``samples`` and ``anchors`` differ in length.
    """
    return group_tracks(vectors, samples, threshold=threshold, join_threshold=join_threshold, anchors=anchors).labels
