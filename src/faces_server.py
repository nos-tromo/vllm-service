"""Face detection and per-video grouping server (YuNet and SFace through OpenCV).

Wraps two OpenCV Zoo ONNX models in a small FastAPI app so a consumer (Nextext)
can find the faces in sampled video frames and group them by person within one
video, without running a model of its own. CPU only.

Endpoints:

    POST /faces/detect
        Body JSON ``{"images": ["<base64 JPEG/PNG>", ...], "min_score": 0.7}``:
        one to eight images, standard base64 without a ``data:`` prefix;
        ``min_score`` in ``[0, 1]`` is optional (default 0.7). Returns
        ``{"results": [{"faces": [{"box": [x, y, w, h], "score": s,
        "landmarks": [[x, y], ... five]}, ...]}, {"error": "undecodable"}, ...]}``,
        one result per image, in order, faces by descending score.
        Coordinates are fractions of the submitted image's width and height,
        clipped to ``[0, 1]``; the landmarks are YuNet's order (right eye,
        left eye, nose tip, right mouth corner, left mouth corner). An image
        whose long side exceeds 1280 px is downscaled first. Detection runs at
        the image's own scale and at half scale (YuNet finds faces of roughly
        10-300 px), and the two sets are merged by non-maximum suppression at
        IoU 0.4, keeping the higher score.

    POST /faces/group
        Body JSON ``{"tracks": [{"samples": [3, 4, 5], "faces": [{"image":
        "<base64 JPEG/PNG>", "landmarks": [[x, y], ... five]}]}, ...]}``: at
        most 500 tracks of at most three faces each; a face is a crop with its
        five landmarks in the crop's pixels, ``samples`` the indices of the
        frames the track was on screen in. Returns ``{"labels": [0, 0, 1, -1]}``,
        one label per track. Per face, SFace ``alignCrop`` + ``feature``,
        L2-normalized; per track, faces far from the track's medoid are dropped
        and the rest averaged; tracks that share a sample index are never
        merged; average-linkage clustering merges while the best pair's mean
        cosine is at least ``FACE_GROUP_THRESHOLD`` (``faces_cluster.py``).
        Labels are numbered by each group's first track; a track without a
        decodable face gets ``-1``.

    GET /health
        Liveness probe; returns ``{"status": "ok", "detect_model": ...,
        "embed_model": ...}``.

Errors: 400 for a malformed payload or one over a count limit; 413 when the
declared ``Content-Length``, or the body as it is read, exceeds 32 MiB; 422
when no image of a detect request decodes; 503 with ``Retry-After: 5`` when
the models are busy with another request for longer than 30 s (requests are
served one at a time, never queued past that).

Unknown top-level keys are ignored: the router drops keys named like its own
parameters before forwarding, so the contract's keys (``images``,
``min_score``, ``tracks``) avoid them, and a key a client adds beside them does
not fail the request.

Configuration:

    FACE_DETECT_MODEL, FACE_DETECT_MODEL_FILE, FACE_DETECT_MODEL_REVISION
        The detection model's Hub repository, file and commit (defaults
        ``opencv/face_detection_yunet``, ``face_detection_yunet_2023mar.onnx``,
        unset).
    FACE_EMBED_MODEL, FACE_EMBED_MODEL_FILE, FACE_EMBED_MODEL_REVISION
        The same for the feature model (``opencv/face_recognition_sface``,
        ``face_recognition_sface_2021dec.onnx``, unset).
    FACE_GROUP_THRESHOLD
        The lowest mean cosine at which two groups merge (default 0.42); a
        value that is not a number in ``[0, 1]`` stops the server at startup.

Both files come from the ``huggingface-cache`` volume (``HF_HUB_OFFLINE=1``): a
revision variable that is set loads that commit, unset or empty the cache's
``refs/main``. They are loaded once at import, so the first request is hot.

Privacy: features exist only inside a ``/faces/group`` request and are dropped
with it; no response, log line or error message carries a feature, an image or
base64 text. The server logs counts and timings only.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Annotated, TypeVar

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from faces_cluster import cluster_tracks, track_vector

MAX_IMAGES = 8
MAX_SIDE = 1280
MAX_TRACKS = 500
MAX_FACES_PER_TRACK = 3
MAX_BODY_BYTES = 32 * 1024 * 1024
#: Seconds a request waits for the models before it is told to retry.
LOCK_WAIT_S = 30.0
RETRY_AFTER_S = 5
NMS_IOU = 0.4
DEFAULT_MIN_SCORE = 0.7
DEFAULT_GROUP_THRESHOLD = 0.42
#: Neither pass runs on an image whose short side is below this: no face fits.
MIN_DETECT_SIDE = 16
#: YuNet's own settings at load; the score threshold is set per request.
_YUNET_NMS = 0.3
_YUNET_TOP_K = 5000

#: The columns of a YuNet row that hold x coordinates (box, then five landmarks), and y.
_X_COLUMNS = [0, 4, 6, 8, 10, 12]
_Y_COLUMNS = [1, 5, 7, 9, 11, 13]
_WIDTH_COLUMN = 2
_HEIGHT_COLUMN = 3
_SCORE_COLUMN = 14
_ROW_WIDTH = 15

# uvicorn puts its handler on this logger; a logger of our own would drop the INFO line.
_log = logging.getLogger("uvicorn.error")

_T = TypeVar("_T")


def _setting(name: str, default: str) -> str:
    """Return an environment variable, or ``default`` when it is unset or blank.

    Args:
        name: The variable.
        default: The value used when it is unset or blank.

    Returns:
        The value without surrounding whitespace.
    """
    return os.environ.get(name, "").strip() or default


def _revision(name: str) -> str | None:
    """Return a revision variable's commit, as the router records it.

    Args:
        name: The variable.

    Returns:
        The value without surrounding whitespace, or ``None`` when it is unset or blank.
    """
    return os.environ.get(name, "").strip() or None


def _group_threshold() -> float:
    """Read ``FACE_GROUP_THRESHOLD``.

    A value outside ``[0, 1]`` would merge everyone or no one, which would only
    show up in the labels, so it stops the server at startup instead.

    Returns:
        The threshold.

    Raises:
        ValueError: When the value is not a number in ``[0, 1]``.
    """
    raw = _setting("FACE_GROUP_THRESHOLD", str(DEFAULT_GROUP_THRESHOLD))
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"FACE_GROUP_THRESHOLD must be a number in [0, 1], got {raw!r}")
    return value


def _model_file(repo_id: str, filename: str, revision: str | None) -> str:
    """Return the local path of a model file in the Hub cache.

    Args:
        repo_id: The Hub repository.
        filename: The file in it.
        revision: The commit to load, or ``None`` for the cache's ``refs/main``.

    Returns:
        The path ``hf_hub_download`` resolves (offline: from the cache only).
    """
    kwargs: dict[str, str] = {"revision": revision} if revision else {}
    path: str = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return path


DETECT_MODEL = _setting("FACE_DETECT_MODEL", "opencv/face_detection_yunet")
DETECT_MODEL_FILE = _setting("FACE_DETECT_MODEL_FILE", "face_detection_yunet_2023mar.onnx")
EMBED_MODEL = _setting("FACE_EMBED_MODEL", "opencv/face_recognition_sface")
EMBED_MODEL_FILE = _setting("FACE_EMBED_MODEL_FILE", "face_recognition_sface_2021dec.onnx")
GROUP_THRESHOLD = _group_threshold()

detector = cv2.FaceDetectorYN.create(
    _model_file(DETECT_MODEL, DETECT_MODEL_FILE, _revision("FACE_DETECT_MODEL_REVISION")),
    "",
    (320, 320),
    DEFAULT_MIN_SCORE,
    _YUNET_NMS,
    _YUNET_TOP_K,
)
recognizer = cv2.FaceRecognizerSF.create(
    _model_file(EMBED_MODEL, EMBED_MODEL_FILE, _revision("FACE_EMBED_MODEL_REVISION")), ""
)

# Neither OpenCV model is safe to call from two threads at once, and the detector's
# input size is per call; one request at a time holds both.
_model_lock = threading.Lock()

app = FastAPI(title="vllm-service faces", version="1.0")


class _Request(BaseModel):
    """Common settings: unknown keys are ignored, and NaN or infinity is malformed."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class DetectRequest(_Request):
    """A ``/faces/detect`` body."""

    images: Annotated[list[str], Field(min_length=1, max_length=MAX_IMAGES)]
    min_score: Annotated[float, Field(ge=0.0, le=1.0)] = DEFAULT_MIN_SCORE


class GroupFace(_Request):
    """One face crop of a track, with its five landmarks in crop pixels."""

    image: str
    landmarks: Annotated[list[tuple[float, float]], Field(min_length=5, max_length=5)]


class GroupTrack(_Request):
    """One track: the frames it was on screen in and up to three of its faces."""

    samples: list[int]
    faces: Annotated[list[GroupFace], Field(max_length=MAX_FACES_PER_TRACK)]


class GroupRequest(_Request):
    """A ``/faces/group`` body."""

    tracks: Annotated[list[GroupTrack], Field(max_length=MAX_TRACKS)]


async def _read_body(request: Request) -> bytes:
    """Read the request body, refusing it once it is over the limit.

    A declared length over the limit is refused before a byte is read; the body
    is also counted as it arrives, so a chunked or understated one is cut off.

    Args:
        request: The request.

    Returns:
        The body.

    Raises:
        HTTPException: 413 when the body is over ``MAX_BODY_BYTES``.
    """
    too_large = HTTPException(status_code=413, detail=f"the request body is over the limit of {MAX_BODY_BYTES} bytes")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise too_large
    return bytes(body)


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _parse(model: type[_ModelT], body: bytes) -> _ModelT:
    """Validate a JSON body against a request model.

    The error names the field and the rule it broke, never the value sent.

    Args:
        model: The request model.
        body: The raw body.

    Returns:
        The validated request.

    Raises:
        HTTPException: 400 when the body is not valid JSON or breaks the contract.
    """
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False, include_context=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "body"
        raise HTTPException(status_code=400, detail=f"malformed request: {where}: {first['msg']}") from None


def _with_models(work: Callable[[], _T]) -> _T:
    """Run ``work`` while holding the models, or refuse when they stay busy.

    Args:
        work: The CPU work, run in the calling (worker) thread.

    Returns:
        What ``work`` returns.

    Raises:
        HTTPException: 503 with ``Retry-After`` when the models are not free within ``LOCK_WAIT_S``.
    """
    if not _model_lock.acquire(timeout=LOCK_WAIT_S):
        raise HTTPException(
            status_code=503,
            detail="the face models are busy; retry later",
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )
    try:
        return work()
    finally:
        _model_lock.release()


def _decode(data: str) -> np.ndarray | None:
    """Decode one base64 image to a BGR array.

    Args:
        data: Standard base64 of a JPEG, PNG or other format OpenCV reads.

    Returns:
        The image, or ``None`` when it does not decode.
    """
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if image is None or image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    return image


def _downscale(image: np.ndarray) -> np.ndarray:
    """Shrink an image whose long side exceeds ``MAX_SIDE`` to that side.

    Args:
        image: The decoded image.

    Returns:
        The image itself, or a downscaled copy.
    """
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale >= 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _detect_rows(image: np.ndarray) -> np.ndarray:
    """Run YuNet over one image at its own size.

    Args:
        image: The image.

    Returns:
        One 15-column row per face, in the image's pixels.
    """
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, rows = detector.detect(image)
    if rows is None:
        return np.zeros((0, _ROW_WIDTH))
    return np.asarray(rows, dtype=np.float64).reshape(-1, _ROW_WIDTH)


def _nms(rows: np.ndarray, iou: float) -> list[int]:
    """Return the rows non-maximum suppression keeps, best score first.

    Args:
        rows: YuNet rows.
        iou: A row is suppressed by a better-scored one it overlaps by more than this.

    Returns:
        The indices of the kept rows.
    """
    x0, y0 = rows[:, 0], rows[:, 1]
    widths = np.maximum(rows[:, _WIDTH_COLUMN], 0.0)
    heights = np.maximum(rows[:, _HEIGHT_COLUMN], 0.0)
    x1, y1 = x0 + widths, y0 + heights
    areas = widths * heights
    suppressed = np.zeros(len(rows), dtype=bool)
    kept: list[int] = []
    for index in np.argsort(-rows[:, _SCORE_COLUMN], kind="stable"):
        if suppressed[index]:
            continue
        kept.append(int(index))
        overlap_w = np.clip(np.minimum(x1[index], x1) - np.maximum(x0[index], x0), 0.0, None)
        overlap_h = np.clip(np.minimum(y1[index], y1) - np.maximum(y0[index], y0), 0.0, None)
        intersection = overlap_w * overlap_h
        union = areas[index] + areas - intersection
        ratio = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppressed |= ratio > iou
    return kept


def _face_json(row: np.ndarray, width: int, height: int) -> dict[str, object]:
    """Express one YuNet row as fractions of the image it was found in.

    Args:
        row: The row, in the image's pixels.
        width: The image's width.
        height: The image's height.

    Returns:
        The face as the contract returns it, clipped to ``[0, 1]``.
    """
    left, right = np.clip([row[0], row[0] + row[_WIDTH_COLUMN]], 0.0, width)
    top, bottom = np.clip([row[1], row[1] + row[_HEIGHT_COLUMN]], 0.0, height)
    box = [left / width, top / height, (right - left) / width, (bottom - top) / height]
    landmarks = [
        [round(float(np.clip(row[x] / width, 0.0, 1.0)), 4), round(float(np.clip(row[y] / height, 0.0, 1.0)), 4)]
        for x, y in zip(_X_COLUMNS[1:], _Y_COLUMNS[1:], strict=True)
    ]
    return {
        "box": [round(float(value), 4) for value in box],
        "score": round(float(row[_SCORE_COLUMN]), 4),
        "landmarks": landmarks,
    }


def _faces_in(image: np.ndarray, min_score: float) -> list[dict[str, object]]:
    """Find the faces in one image at full and half scale.

    Args:
        image: The decoded image.
        min_score: The lowest score returned.

    Returns:
        The faces, best score first, as fractions of the image.
    """
    image = _downscale(image)
    height, width = image.shape[:2]
    if min(height, width) < MIN_DETECT_SIDE:
        return []
    detector.setScoreThreshold(min_score)
    found = [_detect_rows(image)]
    half_width, half_height = width // 2, height // 2
    if min(half_width, half_height) >= MIN_DETECT_SIDE:
        half = _detect_rows(cv2.resize(image, (half_width, half_height), interpolation=cv2.INTER_AREA))
        half[:, [*_X_COLUMNS, _WIDTH_COLUMN]] *= width / half_width
        half[:, [*_Y_COLUMNS, _HEIGHT_COLUMN]] *= height / half_height
        found.append(half)
    rows = np.concatenate(found)
    rows = rows[rows[:, _SCORE_COLUMN] >= min_score]
    return [_face_json(rows[index], width, height) for index in _nms(rows, NMS_IOU)]


def _detect_all(images: list[str], min_score: float) -> list[dict[str, object]]:
    """Detect the faces in each image, one decoded image in memory at a time.

    Args:
        images: The base64 images.
        min_score: The lowest score returned.

    Returns:
        One result per image, in order.
    """
    results: list[dict[str, object]] = []
    for data in images:
        image = _decode(data)
        if image is None:
            results.append({"error": "undecodable"})
        else:
            results.append({"faces": _faces_in(image, min_score)})
    return results


def _face_feature(face: GroupFace) -> np.ndarray | None:
    """Return one crop's SFace feature, L2-normalized.

    Args:
        face: The crop and its landmarks.

    Returns:
        The feature, or ``None`` when the crop does not decode or cannot be aligned.
    """
    image = _decode(face.image)
    if image is None:
        return None
    height, width = image.shape[:2]
    # alignCrop reads a YuNet row; only its landmarks place the crop.
    row = np.zeros(_ROW_WIDTH, dtype=np.float32)
    row[_WIDTH_COLUMN], row[_HEIGHT_COLUMN], row[_SCORE_COLUMN] = width, height, 1.0
    row[4:14] = np.asarray(face.landmarks, dtype=np.float32).reshape(-1)
    try:
        feature = np.asarray(recognizer.feature(recognizer.alignCrop(image, row)), dtype=np.float64).reshape(-1)
    except cv2.error:
        return None
    norm = float(np.linalg.norm(feature))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return feature / norm


def _group_all(tracks: list[GroupTrack]) -> tuple[list[int], int]:
    """Group the tracks by person.

    Args:
        tracks: The tracks.

    Returns:
        One label per track, and how many faces yielded a feature.
    """
    vectors: list[np.ndarray | None] = []
    embedded = 0
    for track in tracks:
        features = [feature for feature in map(_face_feature, track.faces) if feature is not None]
        embedded += len(features)
        vectors.append(track_vector(np.stack(features)) if features else None)
    samples = [frozenset(track.samples) for track in tracks]
    return cluster_tracks(vectors, samples, threshold=GROUP_THRESHOLD), embedded


@app.post("/faces/detect")
async def detect(request: Request) -> dict[str, list[dict[str, object]]]:
    """Detect the faces in up to eight images.

    Args:
        request: The request; its body is read here, against the size limit.

    Returns:
        One result per image, in order.

    Raises:
        HTTPException: 400, 413, 422 or 503, as the module docstring lists.
    """
    started = time.monotonic()
    payload = _parse(DetectRequest, await _read_body(request))
    results = await anyio.to_thread.run_sync(_with_models, lambda: _detect_all(payload.images, payload.min_score))
    undecodable = sum(1 for result in results if "error" in result)
    faces = sum(len(faces) for result in results if isinstance(faces := result.get("faces"), list))
    _log.info(
        "faces.detect images=%d undecodable=%d faces=%d ms=%d",
        len(results),
        undecodable,
        faces,
        round((time.monotonic() - started) * 1000),
    )
    if undecodable == len(results):
        raise HTTPException(status_code=422, detail="no image of the request could be decoded")
    return {"results": results}


@app.post("/faces/group")
async def group(request: Request) -> dict[str, list[int]]:
    """Group up to 500 face tracks by person within one video.

    Args:
        request: The request; its body is read here, against the size limit.

    Returns:
        One label per track.

    Raises:
        HTTPException: 400, 413 or 503, as the module docstring lists.
    """
    started = time.monotonic()
    payload = _parse(GroupRequest, await _read_body(request))
    labels, embedded = await anyio.to_thread.run_sync(_with_models, lambda: _group_all(payload.tracks))
    _log.info(
        "faces.group tracks=%d faces=%d embedded=%d unassigned=%d groups=%d ms=%d",
        len(labels),
        sum(len(track.faces) for track in payload.tracks),
        embedded,
        labels.count(-1),
        len({label for label in labels if label >= 0}),
        round((time.monotonic() - started) * 1000),
    )
    return {"labels": labels}


@app.get("/health")
def health() -> dict[str, str]:
    """Report liveness and the two models.

    Returns:
        The status and the model repositories.
    """
    return {"status": "ok", "detect_model": DETECT_MODEL, "embed_model": EMBED_MODEL}
