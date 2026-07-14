"""VAD-gating tests for the diarize server (torch-free, /vad mocked).

``diarize_server`` loads the pyannote pipeline at import, so — following
``test_clip_server.py`` — torch and ``diarize_pipeline`` are stubbed in
``sys.modules`` before import, and the module is skipped when the real
pyannote is installed (the eval-run env, where importing would load the
gated checkpoint). The ``/vad`` HTTP call is monkeypatched per-test.
"""

import importlib.util
import io
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("pyannote.audio") is not None:
    pytest.skip(
        "diarize_server unit tests need the torch-free env (real pyannote would load the checkpoint at import)",
        allow_module_level=True,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


class _FakeWaveform:
    """Stand-in for the tensor handed to the pipeline; shape is irrelevant."""

    def unsqueeze(self, dim: int) -> "_FakeWaveform":
        """No-op channel-dim insert."""
        return self


class _FakeAnnotation:
    """Minimal pyannote Annotation: fixed turns for every request."""

    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = True) -> Iterator[tuple[types.SimpleNamespace, None, str]]:
        """Yield (segment, track, label) like pyannote's Annotation."""
        for start, end, speaker in self._turns:
            yield types.SimpleNamespace(start=start, end=end), None, speaker

    def labels(self) -> list[str]:
        """Distinct labels, sorted, like pyannote's Annotation."""
        return sorted({speaker for _, _, speaker in self._turns})


_PIPELINE_TURNS = [(0.0, 4.0, "SPEAKER_00"), (10.0, 12.0, "SPEAKER_01")]


def _fake_pipeline(payload: object, **kwargs: object) -> _FakeAnnotation:
    """The stubbed pyannote pipeline: same two turns for any audio."""
    return _FakeAnnotation(list(_PIPELINE_TURNS))


# Stub the heavy imports before diarize_server is imported. Following
# test_clip_server.py's pattern, the stubs are removed again right after the
# import — diarize_server keeps its own references to them, but other test
# files in this session must still see the real ImportError for `torch` (or
# the real pyannote.metrics/scipy stack, which chokes on a torch stub
# lacking a real `Tensor` attribute), and the real `diarize_pipeline` module
# some of them import directly (e.g. test_diarize_pipeline.py,
# test_smoke_e2e.py) rather than this test's fake single-turn pipeline.
_torch_stub = types.ModuleType("torch")
_torch_stub.from_numpy = lambda array: _FakeWaveform()  # type: ignore[attr-defined]
_inserted_torch = sys.modules.setdefault("torch", _torch_stub) is _torch_stub

_pipeline_stub = types.ModuleType("diarize_pipeline")
_pipeline_stub.DEFAULT_MODEL = "stub/diarize-model"  # type: ignore[attr-defined]
_pipeline_stub.build_pipeline = lambda **kwargs: _fake_pipeline  # type: ignore[attr-defined]
# diarize_pipeline may already be the *real* module in sys.modules (another
# test file imported it first) — force our fake in unconditionally, but
# remember what was there so it can be restored rather than just dropped.
_prior_pipeline_module = sys.modules.get("diarize_pipeline")
sys.modules["diarize_pipeline"] = _pipeline_stub

try:
    import diarize_server  # — must come after the sys.modules stubs
finally:
    if _inserted_torch:
        del sys.modules["torch"]
    if _prior_pipeline_module is not None:
        sys.modules["diarize_pipeline"] = _prior_pipeline_module
    else:
        del sys.modules["diarize_pipeline"]

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(diarize_server.app)

_UPLOAD = {"file": ("clip.wav", io.BytesIO(b"fake-bytes"), "audio/wav")}


@pytest.fixture(autouse=True)
def _decoded_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass ffmpeg: every upload decodes to a second of silence."""
    monkeypatch.setattr(diarize_server, "decode_audio", lambda data: np.zeros(16000, dtype=np.float32))


def test_no_vad_url_returns_ungated_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset DIARIZE_VAD_URL → byte-identical legacy behavior, no /vad call."""
    monkeypatch.delenv("DIARIZE_VAD_URL", raising=False)

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("requests.post must not be called when gating is off")

    monkeypatch.setattr(diarize_server.requests, "post", _explode)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    body = response.json()
    assert [s["start"] for s in body["segments"]] == [0.0, 10.0]
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_kill_switch_disables_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """DIARIZE_VAD_GATE=off wins over a set URL."""
    monkeypatch.setenv("DIARIZE_VAD_URL", "http://vad:8000")
    monkeypatch.setenv("DIARIZE_VAD_GATE", "off")

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("requests.post must not be called when the kill switch is off")

    monkeypatch.setattr(diarize_server.requests, "post", _explode)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    assert len(response.json()["segments"]) == 2


def test_gated_turns_are_cropped_to_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speech timeline [1,3] crops turn one and drops turn two entirely."""
    monkeypatch.setenv("DIARIZE_VAD_URL", "http://vad:8000")
    monkeypatch.delenv("DIARIZE_VAD_GATE", raising=False)
    calls: list[dict[str, object]] = []

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"segments": [{"start": 1.0, "end": 3.0}], "has_speech": True, "sampling_rate": 16000}

    def _fake_post(
        url: str, files: object = None, data: dict[str, object] | None = None, timeout: float | None = None
    ) -> _Response:
        calls.append({"url": url, "data": data, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(diarize_server.requests, "post", _fake_post)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["segments"] == [{"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    assert body["speakers"] == ["SPEAKER_00"]  # SPEAKER_01's turn was outside speech
    assert calls[0]["url"] == "http://vad:8000/vad"
    assert calls[0]["data"] == {"threshold": 0.4, "speech_pad_ms": 100}
    assert calls[0]["timeout"] == 30.0


def test_vad_failure_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """/vad down → ungated turns come back, request still succeeds."""
    monkeypatch.setenv("DIARIZE_VAD_URL", "http://vad:8000")

    def _boom(*args: object, **kwargs: object) -> None:
        raise ConnectionError("vad unreachable")

    monkeypatch.setattr(diarize_server.requests, "post", _boom)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    body = response.json()
    assert len(body["segments"]) == 2
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_tuning_env_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom threshold/pad/timeout env reach the /vad form fields."""
    monkeypatch.setenv("DIARIZE_VAD_URL", "http://vad:8000/")  # trailing slash normalized
    monkeypatch.setenv("DIARIZE_VAD_THRESHOLD", "0.3")
    monkeypatch.setenv("DIARIZE_VAD_PAD_MS", "150")
    monkeypatch.setenv("DIARIZE_VAD_TIMEOUT", "5")
    calls: list[dict[str, object]] = []

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"segments": [], "has_speech": False, "sampling_rate": 16000}

    def _fake_post(
        url: str, files: object = None, data: dict[str, object] | None = None, timeout: float | None = None
    ) -> _Response:
        calls.append({"url": url, "data": data, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(diarize_server.requests, "post", _fake_post)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    assert response.json()["segments"] == []  # no speech at all → everything gated away
    assert calls[0]["url"] == "http://vad:8000/vad"
    assert calls[0]["data"] == {"threshold": 0.3, "speech_pad_ms": 150}
    assert calls[0]["timeout"] == 5.0
