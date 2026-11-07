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

Nothing here is kept between calls; the vectors live only as long as the
request that computed them.
"""

import numpy as np

#: Faces whose cosine to their track's medoid is below this are dropped.
OUTLIER_COS = 0.3


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
        assigned: The tracks that have a vector, as indices into ``samples``.
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


def cluster_tracks(vectors: list[np.ndarray | None], samples: list[frozenset[int]], *, threshold: float) -> list[int]:
    """Group tracks by average-linkage clustering under cannot-link constraints.

    Clusters start as single tracks. Each step merges the pair with the highest
    average pairwise cosine, while that average is at least ``threshold`` and
    no track of one shares a sample index with a track of the other. The
    similarity matrix carries ``-inf`` for every pair that may not merge;
    the average-linkage update keeps that, since a merged cluster may not join
    anything either of its parts could not.

    Args:
        vectors: One unit vector per track, or ``None`` for a track without one.
        samples: The sample indices each track was on screen at, one set per track.
        threshold: The lowest average cosine at which two clusters merge.

    Returns:
        One label per track: clusters numbered ``0, 1, …`` in the order of
        their first track; ``-1`` for a track without a vector.

    Raises:
        ValueError: If ``vectors`` and ``samples`` differ in length.
    """
    if len(vectors) != len(samples):
        raise ValueError("vectors and samples must have one entry per track")
    assigned = [index for index, vector in enumerate(vectors) if vector is not None]
    labels = [-1] * len(vectors)
    count = len(assigned)
    if count == 0:
        return labels

    matrix = np.stack([np.asarray(vectors[index], dtype=np.float64) for index in assigned])
    similarity = matrix @ matrix.T
    for members in _cooccurring(assigned, samples):
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

    numbering: dict[int, int] = {}
    for position, index in enumerate(assigned):
        cluster = int(owner[position])
        labels[index] = numbering.setdefault(cluster, len(numbering))
    return labels
