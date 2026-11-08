"""Contract tests for the face detection and grouping server, ``src/faces_server.py``.

OpenCV and huggingface_hub are not installed in this env, so both are stubbed
in ``sys.modules`` while the server is imported. No image here is a picture of
anything: a synthetic image is the base64 of a short ``SYNTH <w> <h> <v> <padding>``
tag, which the stubbed ``cv2.imdecode`` turns into a ``h x w`` array filled with
``v``. The fake detector reports ``v`` faces at fixed fractions of whatever
input size it is given, so the same face shows up at full and at half scale;
the fake recognizer maps a crop filled with ``v`` to the unit vector along axis
``v``, so ``v`` names the person in a grouping test.
"""

import base64
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Scope
from typing_extensions import override

SRC = Path(__file__).resolve().parents[2] / "src"
DIM = 128

#: Every ``(w, h)`` the fake detector was set to, across the session.
SET_INPUT_SIZES: list[tuple[int, int]] = []
#: Every ``hf_hub_download`` call: its keyword arguments.
DOWNLOADS: list[dict[str, object]] = []
#: File names the fake cache does not hold: ``hf_hub_download`` raises for them.
MISSING_FILES: set[str] = set()


class _CvError(Exception):
    """Stand-in for ``cv2.error``."""


def _default_faces(width: int, height: int, value: int) -> list[list[float]]:
    """Return ``value`` YuNet rows at fixed fractions of a ``width x height`` input.

    Args:
        width: The input width the detector was set to.
        height: The input height the detector was set to.
        value: The fill value of the image, i.e. how many faces it holds.

    Returns:
        One 15-column row per face: box, five landmarks, score.
    """
    rows = []
    for index in range(value):
        x, y, w, h = 0.1 + 0.3 * index, 0.2, 0.1, 0.2
        landmarks = [
            (x + 0.03, y + 0.07),
            (x + 0.07, y + 0.07),
            (x + 0.05, y + 0.1),
            (x + 0.035, y + 0.14),
            (x + 0.065, y + 0.14),
        ]
        row = [x * width, y * height, w * width, h * height]
        for lx, ly in landmarks:
            row += [lx * width, ly * height]
        rows.append([*row, 0.9])
    return rows


class _Detector:
    """A fake ``cv2.FaceDetectorYN``: rows come from ``faces_for``, which tests may replace."""

    faces_for: Callable[[int, int, int], list[list[float]]] = staticmethod(_default_faces)

    def __init__(self) -> None:
        """Start with no input size."""
        self.size = (0, 0)
        self.score_threshold = 0.0

    def setInputSize(self, size: tuple[int, int]) -> None:
        """Record the input size, as YuNet needs it to match the image."""
        self.size = (int(size[0]), int(size[1]))
        SET_INPUT_SIZES.append(self.size)

    def setScoreThreshold(self, threshold: float) -> None:
        """Record the score threshold."""
        self.score_threshold = float(threshold)

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Return the configured rows for this input, as YuNet does.

        Args:
            image: The image; its size must match the input size set.

        Returns:
            ``(1, rows)``, rows ``None`` when there are none.
        """
        height, width = image.shape[:2]
        assert (width, height) == self.size, "detect called with an image of another size than setInputSize"
        rows = [
            row for row in type(self).faces_for(width, height, int(image.flat[0])) if row[14] >= self.score_threshold
        ]
        return 1, (np.asarray(rows, dtype=np.float32) if rows else None)


class _Recognizer:
    """A fake ``cv2.FaceRecognizerSF``: a crop filled with ``v`` is person ``v``."""

    def alignCrop(self, image: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Check the YuNet-shaped row and return the image as the crop."""
        assert np.asarray(row).reshape(-1).shape == (15,)
        return image

    def feature(self, crop: np.ndarray) -> np.ndarray:
        """Return the unit vector along axis ``v``, scaled (the server normalizes)."""
        vector = np.zeros((1, DIM), dtype=np.float32)
        vector[0, int(crop.flat[0])] = 3.0
        return vector


def _imdecode(buffer: np.ndarray, flags: int) -> np.ndarray | None:
    """Decode a ``SYNTH <w> <h> <v> <padding>`` tag to a filled array; anything else fails as OpenCV does."""
    parts = bytes(buffer).split()
    if len(parts) != 5 or parts[0] != b"SYNTH":
        return None
    width, height, value = (int(part) for part in parts[1:4])
    return np.full((height, width, 3), value, dtype=np.uint8)


def _resize(image: np.ndarray, dsize: tuple[int, int], interpolation: int = 0) -> np.ndarray:
    """Return an array of the requested size, keeping the fill value."""
    width, height = dsize
    return np.full((height, width, 3), image.flat[0], dtype=np.uint8)


def _hf_hub_download(**kwargs: object) -> str:
    """Record the call; return a fake local path, or raise like a cache miss."""
    DOWNLOADS.append(dict(kwargs))
    if kwargs["filename"] in MISSING_FILES:
        # huggingface_hub's LocalEntryNotFoundError is a FileNotFoundError.
        raise FileNotFoundError(f"cannot find {kwargs['filename']} in the local cache")
    return f"/nonexistent/{kwargs['filename']}"


def _stubs() -> dict[str, types.ModuleType]:
    """Build the ``cv2`` and ``huggingface_hub`` stubs."""
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__version__ = "0.0-stub"  # type: ignore[attr-defined]
    cv2_stub.error = _CvError  # type: ignore[attr-defined]
    cv2_stub.IMREAD_COLOR = 1  # type: ignore[attr-defined]
    cv2_stub.INTER_AREA = 3  # type: ignore[attr-defined]
    cv2_stub.imdecode = _imdecode  # type: ignore[attr-defined]
    cv2_stub.resize = _resize  # type: ignore[attr-defined]
    cv2_stub.FaceDetectorYN = types.SimpleNamespace(create=lambda *_a, **_k: _Detector())  # type: ignore[attr-defined]
    cv2_stub.FaceRecognizerSF = types.SimpleNamespace(create=lambda *_a, **_k: _Recognizer())  # type: ignore[attr-defined]
    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]
    return {"cv2": cv2_stub, "huggingface_hub": hub_stub}


def _import_server(name: str) -> types.ModuleType:
    """Execute ``src/faces_server.py`` as a new module over the stubs.

    The stubs are removed from ``sys.modules`` again afterwards (the module
    keeps its own references), so other test files see the real ImportError.

    Args:
        name: The module name to register it under.

    Returns:
        The imported module.
    """
    saved = {key: sys.modules.get(key) for key in ("cv2", "huggingface_hub")}
    sys.modules.update(_stubs())
    sys.path.insert(0, str(SRC))
    try:
        spec = importlib.util.spec_from_file_location(name, SRC / "faces_server.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SRC))
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous


faces_server = _import_server("faces_server")


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client over the server, with the default fake detector."""
    _Detector.faces_for = staticmethod(_default_faces)
    SET_INPUT_SIZES.clear()
    with TestClient(faces_server.app) as test_client:
        yield test_client
    _Detector.faces_for = staticmethod(_default_faces)


#: Makes every synthetic image's base64 long enough for the leak checks to spot.
_PADDING = b"p" * 48


def _b64(raw: bytes) -> str:
    """Base64-encode ``raw`` as the wire carries it."""
    return base64.b64encode(raw).decode("ascii")


def _img(width: int, height: int, faces: int = 1) -> str:
    """A synthetic ``width x height`` image holding ``faces`` faces."""
    return _b64(b"SYNTH %d %d %d %s" % (width, height, faces, _PADDING))


def _face(person: int, side: int = 112) -> dict[str, object]:
    """A synthetic ``side x side`` crop of ``person``, with landmarks in crop pixels."""
    return {
        "image": _b64(b"SYNTH %d %d %d %s" % (side, side, person, _PADDING)),
        "landmarks": [[38.0, 51.0], [73.0, 51.0], [56.0, 71.0], [41.0, 92.0], [70.0, 92.0]],
    }


def _track(person: int, samples: list[int], faces: int = 2) -> dict[str, object]:
    """A track of ``faces`` crops of ``person`` on screen at ``samples``."""
    return {"samples": samples, "faces": [_face(person) for _ in range(faces)]}


# -- /faces/detect ---------------------------------------------------------------


def test_detect_returns_normalized_faces_in_order(client: TestClient) -> None:
    """One result per image, in order; boxes and landmarks are fractions of the image."""
    response = client.post("/faces/detect", json={"images": [_img(640, 360, 1), _img(640, 360, 2)]})
    assert response.status_code == 200
    out = response.json()
    assert [len(result["faces"]) for result in out["results"]] == [1, 2]
    face = out["results"][0]["faces"][0]
    assert all(0.0 <= v <= 1.0 for v in face["box"]) and len(face["box"]) == 4
    assert len(face["landmarks"]) == 5 and all(len(point) == 2 for point in face["landmarks"])
    assert all(0.0 <= v <= 1.0 for point in face["landmarks"] for v in point)
    assert face["box"] == pytest.approx([0.1, 0.2, 0.1, 0.2], abs=1e-3)
    assert face["landmarks"][0] == pytest.approx([0.13, 0.27], abs=1e-3)
    assert face["score"] == pytest.approx(0.9, abs=1e-3)


def test_detect_marks_undecodable_image_and_keeps_others(client: TestClient) -> None:
    """A corrupt image costs only its own result."""
    images = ["broken", _b64(b"not an image at all"), _img(320, 240)]
    out = client.post("/faces/detect", json={"images": images}).json()
    assert out["results"][0] == {"error": "undecodable"}
    assert out["results"][1] == {"error": "undecodable"}
    assert len(out["results"][2]["faces"]) == 1


def test_opencv_error_costs_only_its_own_image(client: TestClient) -> None:
    """An image OpenCV fails on after decoding is marked undecodable; the batch goes on."""

    def failing(width: int, height: int, value: int) -> list[list[float]]:
        if value == 9:
            raise _CvError("synthetic detector failure")
        return _default_faces(width, height, value)

    _Detector.faces_for = staticmethod(failing)
    response = client.post("/faces/detect", json={"images": [_img(640, 360, 9), _img(640, 360, 1)]})
    assert response.status_code == 200
    out = response.json()
    assert out["results"][0] == {"error": "undecodable"}
    assert len(out["results"][1]["faces"]) == 1
    assert faces_server._model_lock.acquire(blocking=False)
    faces_server._model_lock.release()


def test_opencv_error_costs_only_its_own_face(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crop OpenCV fails on is skipped; the track's other faces still count."""
    recognizer = faces_server.recognizer
    feature = recognizer.feature

    def failing(crop: np.ndarray) -> np.ndarray:
        if int(crop.flat[0]) == 99:
            raise _CvError("synthetic recognizer failure")
        return feature(crop)

    monkeypatch.setattr(recognizer, "feature", failing)
    tracks = [{"samples": [0], "faces": [_face(99), _face(2)]}, {"samples": [1], "faces": [_face(99)]}, _track(2, [5])]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [0, -1, 0]}


def test_detect_all_undecodable_is_422(client: TestClient) -> None:
    """A request none of whose images decode is refused."""
    assert client.post("/faces/detect", json={"images": ["broken"]}).status_code == 422


def test_two_scales_map_back_to_normalized_coordinates(client: TestClient) -> None:
    """A large image is downscaled to 1280 px, run at 1x and 0.5x, and the duplicate merged."""
    out = client.post("/faces/detect", json={"images": [_img(2560, 1440)]}).json()
    (face,) = out["results"][0]["faces"]
    assert face["box"] == pytest.approx([0.1, 0.2, 0.1, 0.2], abs=1e-3)
    assert set(SET_INPUT_SIZES) == {(1280, 720), (640, 360)}
    assert max(SET_INPUT_SIZES) <= (1280, 720)


def test_half_scale_finds_a_face_too_large_for_full_scale(client: TestClient) -> None:
    """A face only the half-scale pass sees comes back in the submitted image's fractions."""

    def only_at_half(width: int, height: int, value: int) -> list[list[float]]:
        return _default_faces(width, height, value) if width < 1000 else []

    _Detector.faces_for = staticmethod(only_at_half)
    out = client.post("/faces/detect", json={"images": [_img(1920, 1080)]}).json()
    (face,) = out["results"][0]["faces"]
    assert face["box"] == pytest.approx([0.1, 0.2, 0.1, 0.2], abs=1e-3)
    assert face["landmarks"][4] == pytest.approx([0.165, 0.34], abs=1e-3)


def test_merge_keeps_the_higher_score(client: TestClient) -> None:
    """Of a face seen at both scales, the better-scored detection is the one returned."""

    def better_at_half(width: int, height: int, value: int) -> list[list[float]]:
        rows = _default_faces(width, height, value)
        return [[*row[:14], 0.95 if width < 1000 else 0.8] for row in rows]

    _Detector.faces_for = staticmethod(better_at_half)
    out = client.post("/faces/detect", json={"images": [_img(1280, 720)]}).json()
    (face,) = out["results"][0]["faces"]
    assert face["score"] == pytest.approx(0.95, abs=1e-3)


def test_min_score_drops_weaker_faces(client: TestClient) -> None:
    """Faces below ``min_score`` are not returned."""
    assert client.post("/faces/detect", json={"images": [_img(640, 360)], "min_score": 0.95}).json() == {
        "results": [{"faces": []}]
    }


def test_box_partly_outside_the_image_is_clipped(client: TestClient) -> None:
    """YuNet can report a box past the edge; the response stays within [0, 1]."""

    def off_edge(width: int, height: int, value: int) -> list[list[float]]:
        row = _default_faces(width, height, 1)[0]
        return [[-0.05 * width, -0.05 * height, 0.2 * width, 0.2 * height, *row[4:]]]

    _Detector.faces_for = staticmethod(off_edge)
    out = client.post("/faces/detect", json={"images": [_img(640, 360)]}).json()
    (face,) = out["results"][0]["faces"]
    assert face["box"] == pytest.approx([0.0, 0.0, 0.15, 0.15], abs=1e-3)


@pytest.mark.parametrize(
    ("width", "height", "passes"),
    [(15, 200, []), (31, 100, [(31, 100)]), (32, 32, [(32, 32), (16, 16)])],
    ids=["too-small-for-any-pass", "too-small-for-the-half-pass", "both-passes"],
)
def test_minimum_side_decides_which_passes_run(
    client: TestClient, width: int, height: int, passes: list[tuple[int, int]]
) -> None:
    """No pass runs below 16 px on the short side, and the half pass needs 16 px at half size."""
    out = client.post("/faces/detect", json={"images": [_img(width, height)]}).json()
    assert SET_INPUT_SIZES == passes
    assert len(out["results"][0]["faces"]) == (1 if passes else 0)


@pytest.mark.parametrize(("value", "status"), [(True, 400), ("0.5", 400), (None, 400), (1, 200), (0, 200), (0.5, 200)])
def test_min_score_is_a_number(client: TestClient, value: object, status: int) -> None:
    """``min_score`` takes a JSON number only, never a boolean or a string."""
    response = client.post("/faces/detect", json={"images": [_img(64, 64)], "min_score": value})
    assert response.status_code == status


def test_more_than_eight_images_is_400(client: TestClient) -> None:
    """A batch over the image limit is refused before anything is decoded."""
    assert client.post("/faces/detect", json={"images": [_img(64, 64)] * 9}).status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b'{"images": "one"}',
        b'{"images": []}',
        b'{"images": [1]}',
        b'{"images": ["x"], "min_score": 1.5}',
        b'{"images": ["x"], "min_score": NaN}',
        b"[]",
    ],
    ids=["not-json", "not-a-list", "empty", "not-a-string", "score-range", "score-nan", "not-an-object"],
)
def test_malformed_detect_is_400(client: TestClient, body: bytes) -> None:
    """A malformed payload is a 400 (422 is kept for "nothing decoded")."""
    response = client.post("/faces/detect", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400


def test_body_over_limit_is_413(client: TestClient) -> None:
    """A declared length over 32 MiB is refused before the body is parsed."""
    body = b'{"images": ["' + b"A" * (40 * 1024 * 1024) + b'"]}'
    response = client.post("/faces/detect", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_streamed_body_over_limit_is_413(client: TestClient) -> None:
    """A body without a declared length is counted as it arrives."""

    def chunks() -> Iterator[bytes]:
        yield b'{"tracks": [{"samples": [0], "faces": [{"image": "'
        for _ in range(33):
            yield b"A" * (1024 * 1024)
        yield b'", "landmarks": []}]}]}'

    response = client.post("/faces/group", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_busy_lock_is_503_with_retry_after(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that cannot get the model within the wait is told to come back, not queued."""
    monkeypatch.setattr(faces_server, "LOCK_WAIT_S", 0.01)
    assert faces_server._model_lock.acquire(timeout=1)
    try:
        detect = client.post("/faces/detect", json={"images": [_img(64, 64)]})
        group = client.post("/faces/group", json={"tracks": [_track(0, [0])]})
    finally:
        faces_server._model_lock.release()
    for response in (detect, group):
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"


def _asgi_post(path: str, body: bytes, delays: Sequence[float] = (0.0,)) -> tuple[int, dict[str, str], list[str]]:
    """Send one POST straight to the ASGI app, recording whether it read the body.

    Each call runs its own event loop, so calls from several threads are
    concurrent requests to the one app.

    Args:
        path: The route.
        body: The JSON body.
        delays: The body arrives in ``len(delays)`` pieces, piece ``i`` after
            waiting ``delays[i]`` seconds, as a slow client would send it.

    Returns:
        The status, the response headers, and one ``"receive"`` per body read.
    """
    reads: list[str] = []
    messages: list[Message] = []
    size = -(-len(body) // len(delays))
    pieces = [body[index * size : (index + 1) * size] for index in range(len(delays))]

    async def receive() -> Message:
        index = len(reads)
        reads.append("receive")
        if index >= len(pieces):
            return {"type": "http.disconnect"}
        await anyio.sleep(delays[index])
        return {"type": "http.request", "body": pieces[index], "more_body": index < len(pieces) - 1}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    anyio.run(faces_server.app, scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    headers = {bytes(name).decode(): bytes(value).decode() for name, value in start["headers"]}
    return int(start["status"]), headers, reads


def _detect_body() -> bytes:
    """A one-image detect body."""
    return json.dumps({"images": [_img(64, 64)]}).encode()


def _group_body() -> bytes:
    """A one-track group body."""
    return json.dumps({"tracks": [_track(0, [0])]}).encode()


@pytest.mark.parametrize(("path", "body"), [("/faces/detect", _detect_body), ("/faces/group", _group_body)])
def test_a_fifth_request_is_refused_before_its_body_is_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str, body: Callable[[], bytes]
) -> None:
    """With four requests in flight, one more gets 503 at once and none of its body is read."""
    monkeypatch.setattr(faces_server, "_in_flight", faces_server.MAX_IN_FLIGHT)
    status, headers, reads = _asgi_post(path, body())
    assert status == 503
    assert headers["retry-after"] == "5"
    assert reads == []
    assert faces_server._in_flight == faces_server.MAX_IN_FLIGHT


def test_a_request_is_admitted_while_a_slot_is_free(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three in flight leave room for one more, which returns its slot when done."""
    monkeypatch.setattr(faces_server, "_in_flight", faces_server.MAX_IN_FLIGHT - 1)
    status, _, reads = _asgi_post("/faces/detect", _detect_body())
    assert status == 200
    assert reads
    assert faces_server._in_flight == faces_server.MAX_IN_FLIGHT - 1


def test_a_stalled_body_is_408_and_frees_its_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body that stops arriving mid-stream is cut off after the stall limit; the slot is free again."""
    monkeypatch.setattr(faces_server, "_RECEIVE_STALL_S", 0.2)
    monkeypatch.setattr(faces_server, "_BODY_TIMEOUT_S", 5.0)
    started = time.monotonic()
    status, headers, _ = _asgi_post("/faces/detect", _detect_body(), delays=(0.0, 30.0))
    assert time.monotonic() - started < 2.0
    assert status == 408
    assert headers["connection"] == "close"
    assert faces_server._in_flight == 0
    assert _asgi_post("/faces/detect", _detect_body())[0] == 200


def test_a_body_slower_than_the_total_budget_is_408(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trickle that never stalls long still has to finish within the body budget."""
    monkeypatch.setattr(faces_server, "_RECEIVE_STALL_S", 0.3)
    monkeypatch.setattr(faces_server, "_BODY_TIMEOUT_S", 0.5)
    started = time.monotonic()
    status, headers, reads = _asgi_post("/faces/group", _group_body(), delays=(0.15,) * 20)
    elapsed = time.monotonic() - started
    assert 0.4 < elapsed < 2.0
    assert status == 408
    assert headers["connection"] == "close"
    assert 1 < len(reads) < 20
    assert faces_server._in_flight == 0


def test_a_body_within_both_limits_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pieces that keep coming and finish in time make an ordinary request."""
    monkeypatch.setattr(faces_server, "_RECEIVE_STALL_S", 0.3)
    monkeypatch.setattr(faces_server, "_BODY_TIMEOUT_S", 5.0)
    assert _asgi_post("/faces/detect", _detect_body(), delays=(0.05,) * 5)[0] == 200


def test_four_in_flight_then_a_fifth_is_refused_and_a_later_one_admitted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four real requests held on the models fill the slots; once they finish, a new one is served."""
    monkeypatch.setattr(faces_server, "LOCK_WAIT_S", 10.0)
    statuses: list[int] = []

    def held_request() -> None:
        statuses.append(_asgi_post("/faces/detect", _detect_body())[0])

    assert faces_server._model_lock.acquire(timeout=1)
    held = [threading.Thread(target=held_request) for _ in range(faces_server.MAX_IN_FLIGHT)]
    try:
        for thread in held:
            thread.start()
        give_up = time.monotonic() + 5
        while faces_server._in_flight < faces_server.MAX_IN_FLIGHT and time.monotonic() < give_up:
            time.sleep(0.01)
        assert faces_server._in_flight == faces_server.MAX_IN_FLIGHT
        started = time.monotonic()
        fifth_status, _, fifth_reads = _asgi_post("/faces/group", _group_body())
        assert time.monotonic() - started < 2.0
    finally:
        faces_server._model_lock.release()
        for thread in held:
            thread.join(timeout=15)
    assert fifth_status == 503
    assert fifth_reads == []
    assert statuses == [200] * faces_server.MAX_IN_FLIGHT
    assert faces_server._in_flight == 0
    assert _asgi_post("/faces/detect", _detect_body())[0] == 200


def test_health_runs_on_the_event_loop() -> None:
    """The probe never needs a worker thread, so a busy threadpool cannot starve it."""
    assert inspect.iscoroutinefunction(faces_server.health)


def test_unknown_top_level_keys_are_ignored(client: TestClient) -> None:
    """Keys a client (or the router) adds beside the contract's do not fail the request."""
    body = {"images": [_img(64, 64)], "min_score": 0.5, "extra": {"anything": 1}}
    assert client.post("/faces/detect", json=body).status_code == 200
    assert client.post("/faces/group", json={"tracks": [], "extra": True}).json() == {"labels": []}


# -- /faces/group ----------------------------------------------------------------


def test_group_returns_labels_only(client: TestClient) -> None:
    """Two people, two labels; the response is the labels and nothing else."""
    out = client.post("/faces/group", json={"tracks": [_track(0, [0]), _track(1, [1])]}).json()
    assert out == {"labels": [0, 1]}


def test_group_merges_one_person_across_tracks(client: TestClient) -> None:
    """Tracks of one person at different times share a label."""
    tracks = [_track(3, [0, 1]), _track(5, [2]), _track(3, [4])]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [0, 1, 0]}


def test_group_never_merges_tracks_on_screen_together(client: TestClient) -> None:
    """Two tracks that share a sample are two people, however alike they look."""
    tracks = [_track(3, [0, 1]), _track(3, [1, 2])]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [0, 1]}


def test_group_track_without_a_decodable_face_is_unassigned(client: TestClient) -> None:
    """A track none of whose faces decode gets ``-1``; a track with one good face still counts."""
    broken = {"image": "broken", "landmarks": _face(0)["landmarks"]}
    tracks = [{"samples": [0], "faces": [broken]}, {"samples": [1], "faces": [broken, _face(2)]}, _track(2, [5])]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [-1, 0, 0]}


def test_group_track_without_faces_is_unassigned(client: TestClient) -> None:
    """A track with no faces at all is valid and unassigned."""
    assert client.post("/faces/group", json={"tracks": [{"samples": [0], "faces": []}]}).json() == {"labels": [-1]}


def _weak(person: int, samples: list[int]) -> dict[str, object]:
    """A weak track (no face the client judged good) of ``person`` on screen at ``samples``."""
    return {**_track(person, samples), "anchor": False}


def test_group_weak_track_joins_its_person(client: TestClient) -> None:
    """A weak track takes the label of the person it matches, one seen more than once."""
    tracks = [_track(3, [0, 3]), _track(5, [1]), _weak(3, [2])]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [0, 1, 0]}


def test_group_weak_tracks_found_no_person(client: TestClient) -> None:
    """Weak tracks alone stay unassigned, however alike."""
    tracks = [_weak(3, [0]), _weak(3, [1])]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [-1, -1]}


@pytest.mark.parametrize("flag", [None, True], ids=["missing", "true"])
def test_group_track_is_an_anchor_unless_marked_weak(client: TestClient, flag: bool | None) -> None:
    """Without ``anchor`` a track is an anchor, as before the field existed."""
    second = _track(3, [1]) if flag is None else {**_track(3, [1]), "anchor": flag}
    assert client.post("/faces/group", json={"tracks": [_track(3, [0]), second]}).json() == {"labels": [0, 0]}


@pytest.mark.parametrize("flag", ["no", 0, 1, None], ids=["string", "zero", "one", "null"])
def test_group_anchor_must_be_a_boolean(client: TestClient, flag: object) -> None:
    """``anchor`` is a JSON boolean or absent; anything else is malformed."""
    track = {**_track(3, [0]), "anchor": flag}
    assert client.post("/faces/group", json={"tracks": [track]}).status_code == 400


def test_group_log_counts_weak_tracks_pieces_and_joins(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """The group log line says how many tracks were weak, how many pieces could join, and how many did."""
    caplog.set_level("INFO")
    tracks = [_track(3, [0, 3]), _weak(3, [1]), _weak(4, [2]), _track(6, [4])]
    client.post("/faces/group", json={"tracks": tracks})
    line = next(record.getMessage() for record in caplog.records if "faces.group" in record.getMessage())
    assert "weak=2" in line and "pieces=3" in line and "joined=1" in line


def test_group_passes_the_join_threshold(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Grouping uses the configured join threshold."""
    seen: dict[str, object] = {}
    real = faces_server.group_tracks

    def spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(faces_server, "group_tracks", spy)
    monkeypatch.setattr(faces_server, "JOIN_THRESHOLD", 0.31)
    assert client.post("/faces/group", json={"tracks": [_track(3, [0])]}).status_code == 200
    assert seen["join_threshold"] == 0.31
    assert seen["threshold"] == faces_server.GROUP_THRESHOLD


@pytest.mark.parametrize(
    "tracks",
    [
        [_track(0, [0])] * 501,
        [_track(0, [0], faces=4)],
        [{"samples": [0], "faces": [{"image": "x", "landmarks": [[1.0, 2.0]] * 4}]}],
        [{"samples": [0], "faces": [{"image": "x", "landmarks": [[1.0, 2.0, 3.0]] * 5}]}],
        [{"samples": "0", "faces": []}],
    ],
    ids=["too-many-tracks", "too-many-faces", "four-landmarks", "three-coordinates", "samples-not-a-list"],
)
def test_malformed_group_is_400(client: TestClient, tracks: list[object]) -> None:
    """Over a count limit or malformed, a group request is a 400."""
    assert client.post("/faces/group", json={"tracks": tracks}).status_code == 400


# -- what leaves the server --------------------------------------------------------

_BASE64_RUN = re.compile(rb"[A-Za-z0-9+/]{40,}={0,2}")


def _longest_float_list(value: object) -> int:
    """Return the length of the longest list of numbers anywhere in ``value``."""
    if isinstance(value, dict):
        return max((_longest_float_list(item) for item in value.values()), default=0)
    if isinstance(value, list):
        own = len(value) if value and all(isinstance(item, float) for item in value) else 0
        return max([own, *(_longest_float_list(item) for item in value)])
    return 0


def test_no_response_carries_a_vector_or_image(client: TestClient) -> None:
    """Neither an embedding nor an image (nor its base64) is in any response body."""
    responses = [
        client.post("/faces/detect", json={"images": [_img(1920, 1080, 3), "broken"]}),
        client.post("/faces/detect", json={"images": ["broken"]}),
        client.post("/faces/group", json={"tracks": [_track(0, [0]), _track(1, [1])]}),
        client.post("/faces/group", json={"tracks": [_track(0, [0], faces=4)]}),
        client.get("/health"),
    ]
    for response in responses:
        assert not _BASE64_RUN.search(response.content)
        assert _longest_float_list(json.loads(response.content)) <= 10


def test_no_log_line_carries_a_vector_or_image(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """The server logs counts only."""
    caplog.set_level("DEBUG")
    client.post("/faces/detect", json={"images": [_img(640, 360, 2), "broken"]})
    client.post("/faces/group", json={"tracks": [_track(0, [0]), _track(1, [1])]})
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "SYNTH" not in text and not _BASE64_RUN.search(text.encode())
    assert "faces.detect" in text and "faces.group" in text


def test_health(client: TestClient) -> None:
    """The probe names both models."""
    assert client.get("/health").json() == {
        "status": "ok",
        "detect_model": "opencv/face_detection_yunet",
        "embed_model": "opencv/face_recognition_sface",
    }


# -- per-request bounds ----------------------------------------------------------------

#: OpenCV's decode ceiling the server sets: a 4K frame.
FOUR_K_PIXELS = 3840 * 2160


def test_detect_refuses_a_canvas_over_4k_as_undecodable(client: TestClient) -> None:
    """An image that decodes to more than 3840 x 2160 pixels costs only its own result."""
    out = client.post("/faces/detect", json={"images": [_img(3841, 2160), _img(3840, 2160)]}).json()
    assert out["results"][0] == {"error": "undecodable"}
    assert len(out["results"][1]["faces"]) == 1


def test_decode_cap_is_set_before_opencv_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenCV reads OPENCV_IO_MAX_IMAGE_PIXELS once, as it loads, so the server sets it before importing cv2."""
    seen: list[str | None] = []
    stub = _stubs()["cv2"]

    class _Cv2Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Loads the cv2 stub as a real import, recording the variable at that moment."""

        @override
        def find_spec(
            self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None
        ) -> importlib.machinery.ModuleSpec | None:
            """Claim ``cv2`` only."""
            return importlib.util.spec_from_loader(fullname, self) if fullname == "cv2" else None

        @override
        def exec_module(self, module: types.ModuleType) -> None:
            """Record the variable, then fill the module from the stub."""
            seen.append(os.environ.get("OPENCV_IO_MAX_IMAGE_PIXELS"))
            vars(module).update({key: value for key, value in vars(stub).items() if not key.startswith("__")})

    monkeypatch.delenv("OPENCV_IO_MAX_IMAGE_PIXELS", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Cv2Finder(), *sys.meta_path])
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", _stubs()["huggingface_hub"])
    monkeypatch.syspath_prepend(str(SRC))
    spec = importlib.util.spec_from_file_location("_faces_server_env", SRC / "faces_server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_faces_server_env", module)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("cv2", None)
    assert seen == [str(FOUR_K_PIXELS)]


def test_group_refuses_a_crop_over_512_px(client: TestClient) -> None:
    """A crop larger than 512 px on a side counts as undecodable; a 512 px crop is used."""
    tracks = [{"samples": [0], "faces": [_face(1, side=600)]}, {"samples": [1], "faces": [_face(1, side=512)]}]
    assert client.post("/faces/group", json={"tracks": tracks}).json() == {"labels": [-1, 0]}


def _faceless_track(samples: int) -> dict[str, object]:
    """A track with ``samples`` sample indices and no faces."""
    faces: list[dict[str, object]] = []
    return {"samples": list(range(samples)), "faces": faces}


def test_track_over_3600_samples_is_400(client: TestClient) -> None:
    """A track may name at most 3600 samples."""
    at_limit = client.post("/faces/group", json={"tracks": [_faceless_track(3600)]})
    assert at_limit.json() == {"labels": [-1]}
    over = client.post("/faces/group", json={"tracks": [_faceless_track(3601)]})
    assert over.status_code == 400


def test_request_over_100000_samples_is_400(client: TestClient) -> None:
    """All tracks together may name at most 100,000 samples."""
    full = [_faceless_track(3600) for _ in range(27)]
    at_limit = [*full, _faceless_track(2800)]
    assert client.post("/faces/group", json={"tracks": at_limit}).status_code == 200
    over = [*full, _faceless_track(2801)]
    assert client.post("/faces/group", json={"tracks": over}).status_code == 400


def test_detect_past_its_deadline_is_503_and_frees_the_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that runs past its deadline stops between images with a 503, holding nothing."""
    clock = [0.0]
    monkeypatch.setattr(faces_server, "_now", lambda: clock[0])

    def slow(width: int, height: int, value: int) -> list[list[float]]:
        clock[0] += 100.0
        return _default_faces(width, height, value)

    _Detector.faces_for = staticmethod(slow)
    response = client.post("/faces/detect", json={"images": [_img(640, 360)] * 3})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert SET_INPUT_SIZES == [(640, 360), (320, 180)]
    assert faces_server._model_lock.acquire(blocking=False)
    faces_server._model_lock.release()


def test_group_past_its_deadline_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The grouping loop checks the deadline too."""
    monkeypatch.setattr(faces_server, "REQUEST_DEADLINE_S", 0.0)
    response = client.post("/faces/group", json={"tracks": [_track(0, [0])]})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"


# -- model loading -------------------------------------------------------------------

REVISION_VARS = ("FACE_DETECT_MODEL_REVISION", "FACE_EMBED_MODEL_REVISION")


def test_models_load_at_their_pinned_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each model file comes from its own repository, file name and revision."""
    monkeypatch.setenv("FACE_DETECT_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("FACE_EMBED_MODEL_REVISION", "b" * 40)
    DOWNLOADS.clear()
    _import_server("_faces_server_pinned")
    sys.modules.pop("_faces_server_pinned", None)
    assert DOWNLOADS == [
        {
            "repo_id": "opencv/face_detection_yunet",
            "filename": "face_detection_yunet_2023mar.onnx",
            "revision": "a" * 40,
            "local_files_only": True,
        },
        {
            "repo_id": "opencv/face_recognition_sface",
            "filename": "face_recognition_sface_2021dec.onnx",
            "revision": "b" * 40,
            "local_files_only": True,
        },
    ]


@pytest.mark.parametrize("value", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_models_load_without_revision_when_unset(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Unset, empty or blank, no ``revision`` argument is passed."""
    for variable in REVISION_VARS:
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
    DOWNLOADS.clear()
    _import_server("_faces_server_unpinned")
    sys.modules.pop("_faces_server_unpinned", None)
    assert len(DOWNLOADS) == 2
    assert all("revision" not in call for call in DOWNLOADS)


def test_models_follow_their_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """The repositories and file names are the FACE_*_MODEL{,_FILE} variables."""
    monkeypatch.setenv("FACE_DETECT_MODEL", "example/detector")
    monkeypatch.setenv("FACE_DETECT_MODEL_FILE", "detector.onnx")
    monkeypatch.setenv("FACE_EMBED_MODEL", "example/embedder")
    monkeypatch.setenv("FACE_EMBED_MODEL_FILE", "embedder.onnx")
    for variable in REVISION_VARS:
        monkeypatch.delenv(variable, raising=False)
    DOWNLOADS.clear()
    _import_server("_faces_server_custom")
    sys.modules.pop("_faces_server_custom", None)
    assert DOWNLOADS == [
        {"repo_id": "example/detector", "filename": "detector.onnx", "local_files_only": True},
        {"repo_id": "example/embedder", "filename": "embedder.onnx", "local_files_only": True},
    ]


def test_models_never_download_whatever_hf_hub_offline_says(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HF_HUB_OFFLINE=0`` (left over from a gated first fetch) still loads from the cache only."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    DOWNLOADS.clear()
    _import_server("_faces_server_online_env")
    sys.modules.pop("_faces_server_online_env", None)
    assert len(DOWNLOADS) == 2
    assert all(call.get("local_files_only") is True for call in DOWNLOADS)


@pytest.mark.parametrize(("revision", "named"), [(None, "refs/main"), ("c" * 40, "c" * 40)], ids=["unpinned", "pinned"])
def test_a_missing_model_file_stops_startup_with_what_to_fetch(
    monkeypatch: pytest.MonkeyPatch, revision: str | None, named: str
) -> None:
    """A file missing from the cache names the repository, file and revision to fetch."""
    if revision is None:
        monkeypatch.delenv("FACE_EMBED_MODEL_REVISION", raising=False)
    else:
        monkeypatch.setenv("FACE_EMBED_MODEL_REVISION", revision)
    monkeypatch.setattr(sys.modules[__name__], "MISSING_FILES", {"face_recognition_sface_2021dec.onnx"})
    with pytest.raises(RuntimeError) as raised:
        _import_server("_faces_server_missing")
    sys.modules.pop("_faces_server_missing", None)
    message = str(raised.value)
    assert "face_recognition_sface_2021dec.onnx" in message
    assert "opencv/face_recognition_sface" in message
    assert named in message
    assert "huggingface-cache" in message
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("value", ["abc", "nan", "-0.1", "1.5"])
def test_bad_group_threshold_stops_startup(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A threshold that is not a number in [0, 1] fails the import, not every request."""
    monkeypatch.setenv("FACE_GROUP_THRESHOLD", value)
    with pytest.raises(ValueError, match="FACE_GROUP_THRESHOLD"):
        _import_server("_faces_server_bad_threshold")
    sys.modules.pop("_faces_server_bad_threshold", None)


@pytest.mark.parametrize(("value", "expected"), [(None, 0.25), ("0.3", 0.3), ("0", 0.0), ("0.42", 0.42)])
def test_join_threshold_follows_its_variable(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float
) -> None:
    """``FACE_JOIN_THRESHOLD`` defaults to 0.25 and may go up to the group threshold."""
    monkeypatch.delenv("FACE_GROUP_THRESHOLD", raising=False)
    if value is None:
        monkeypatch.delenv("FACE_JOIN_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("FACE_JOIN_THRESHOLD", value)
    module = _import_server("_faces_server_join_threshold")
    sys.modules.pop("_faces_server_join_threshold", None)
    assert module.JOIN_THRESHOLD == expected


@pytest.mark.parametrize("value", ["abc", "nan", "-0.1", "1.5", "0.5"])
def test_bad_join_threshold_stops_startup(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A join threshold that is not a number in [0, FACE_GROUP_THRESHOLD] fails the import."""
    monkeypatch.delenv("FACE_GROUP_THRESHOLD", raising=False)
    monkeypatch.setenv("FACE_JOIN_THRESHOLD", value)
    with pytest.raises(ValueError, match="FACE_JOIN_THRESHOLD"):
        _import_server("_faces_server_bad_join_threshold")
    sys.modules.pop("_faces_server_bad_join_threshold", None)
