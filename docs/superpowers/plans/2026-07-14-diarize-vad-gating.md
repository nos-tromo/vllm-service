# Diarize Backend VAD Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Crop `/diarize` turns to the Silero `/vad` speech timeline inside the diarize service (fail-open, on by default in the full stack), so every consumer gets the measured −35% false-alarm win.

**Architecture:** A pure interval-cropping helper (`src/diarize_gate.py`) plus request-time env/HTTP glue in `src/diarize_server.py` that POSTs the original upload to the stack's `vad` service and intersects the returned speech segments with the pipeline's turns. Full-stack compose turns it on by setting `DIARIZE_VAD_URL`; unset URL means byte-identical behavior.

**Tech Stack:** FastAPI, `requests` (already in the diarize images via `huggingface_hub`), pytest (torch-free `eval` group; heavy deps stubbed in `sys.modules` following `eval/tests/test_clip_server.py`).

Spec: `docs/superpowers/specs/2026-07-14-diarize-vad-gating-design.md`.

## Global Constraints

- Unset `DIARIZE_VAD_URL` (the code default) → **byte-identical** responses to today; diarize-only shape unchanged.
- Fail-open: any `/vad` failure (connection, non-200, timeout, bad body) logs one warning and returns ungated turns. Never a 5xx from the gate itself.
- Defaults: `DIARIZE_VAD_THRESHOLD=0.4`, `DIARIZE_VAD_PAD_MS=100`, `DIARIZE_VAD_TIMEOUT=30`; `DIARIZE_VAD_GATE` kill switch (`off`/`false`/`no`/`0` disable).
- `/diarize` response JSON shape unchanged; `GET /health` untouched (never calls `/vad`).
- No `depends_on`/healthcheck changes in compose.
- Lint gate: `make verify` (ruff + pyrefly via pre-commit) must pass; new test files must be `git add`ed first (pre-commit is tracked-only).
- Tests run torch-free: `uv run --group eval pytest eval/tests/ -v` — no GPU, no network, no model download.
- Per repo hard rule: no real production/testing data (incl. real clip names) in any committed file.

---

### Task 1: Pure crop helper — `src/diarize_gate.py`

**Files:**
- Create: `src/diarize_gate.py`
- Test: `eval/tests/test_diarize_gate.py`

**Interfaces:**
- Produces: `crop_turns_to_speech(turns: list[tuple[float, float, str]], speech: list[tuple[float, float]]) -> list[tuple[float, float, str]]` — Task 2's server wiring imports exactly this.

- [ ] **Step 1: Write the failing tests**

Create `eval/tests/test_diarize_gate.py`:

```python
"""Tests for the pure VAD-gating interval logic (torch-free)."""

import sys
from pathlib import Path

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diarize_gate import crop_turns_to_speech


def test_turn_inside_speech_is_unchanged() -> None:
    """A turn fully covered by one speech interval passes through as-is."""
    turns = [(1.0, 2.0, "SPEAKER_00")]
    speech = [(0.5, 3.0)]
    assert crop_turns_to_speech(turns, speech) == [(1.0, 2.0, "SPEAKER_00")]


def test_turn_outside_speech_is_dropped() -> None:
    """A turn with no speech overlap (music/noise false alarm) is removed."""
    turns = [(5.0, 7.0, "SPEAKER_00")]
    speech = [(0.0, 4.0)]
    assert crop_turns_to_speech(turns, speech) == []


def test_straddling_turn_is_trimmed() -> None:
    """A turn overhanging the speech edge is cropped to the overlap."""
    turns = [(1.0, 5.0, "SPEAKER_00")]
    speech = [(2.0, 8.0)]
    assert crop_turns_to_speech(turns, speech) == [(2.0, 5.0, "SPEAKER_00")]


def test_turn_spanning_a_gap_splits_in_two() -> None:
    """A turn bridging two speech intervals yields one sub-turn per interval."""
    turns = [(0.0, 10.0, "SPEAKER_00")]
    speech = [(1.0, 3.0), (6.0, 8.0)]
    assert crop_turns_to_speech(turns, speech) == [
        (1.0, 3.0, "SPEAKER_00"),
        (6.0, 8.0, "SPEAKER_00"),
    ]


def test_empty_speech_drops_everything() -> None:
    """No detected speech (pure music upload) → no turns survive."""
    assert crop_turns_to_speech([(0.0, 5.0, "SPEAKER_00")], []) == []


def test_empty_turns_is_noop() -> None:
    """No turns in → no turns out, regardless of speech."""
    assert crop_turns_to_speech([], [(0.0, 5.0)]) == []


def test_zero_length_overlap_is_dropped() -> None:
    """A turn merely touching a speech boundary produces no zero-length turn."""
    turns = [(3.0, 5.0, "SPEAKER_00")]
    speech = [(0.0, 3.0)]
    assert crop_turns_to_speech(turns, speech) == []


def test_unsorted_input_yields_chronological_output() -> None:
    """Output is chronological even when turns and speech arrive unsorted."""
    turns = [(6.0, 7.0, "SPEAKER_01"), (1.0, 2.0, "SPEAKER_00")]
    speech = [(5.0, 8.0), (0.0, 3.0)]
    assert crop_turns_to_speech(turns, speech) == [
        (1.0, 2.0, "SPEAKER_00"),
        (6.0, 7.0, "SPEAKER_01"),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group eval pytest eval/tests/test_diarize_gate.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'diarize_gate'`

- [ ] **Step 3: Write the implementation**

Create `src/diarize_gate.py`:

```python
"""Pure interval logic for VAD-gating diarization turns.

The diarize server (``src/diarize_server.py``) optionally crops the pyannote
pipeline's speaker turns to a Silero VAD speech timeline fetched from the
stack's ``vad`` service, dropping the music/noise the diarizer over-detects
as speech (measured −35% false alarm / −12.5% DER at threshold 0.4 /
pad 100 ms — see eval/reports/2026-07-11-false-alarm-vad-gating.md). This
module holds only the interval intersection: no torch, no HTTP, no env, so
the torch-free eval test group can exercise it directly.
"""

from __future__ import annotations


def crop_turns_to_speech(
    turns: list[tuple[float, float, str]],
    speech: list[tuple[float, float]],
) -> list[tuple[float, float, str]]:
    """Intersect speaker turns with a speech timeline.

    Each turn is cropped to the speech intervals it overlaps: a turn fully
    inside speech is unchanged, one overhanging an edge is trimmed, one
    bridging a silence gap splits into one sub-turn per speech interval,
    and one with no overlap is dropped. Zero-length results are dropped.

    Args:
        turns: ``(start, end, speaker)`` turns in seconds; any order.
        speech: ``(start, end)`` speech intervals in seconds; any order.

    Returns:
        Cropped ``(start, end, speaker)`` turns in chronological order.
    """
    cropped: list[tuple[float, float, str]] = []
    for start, end, speaker in turns:
        for speech_start, speech_end in speech:
            overlap_start = max(start, speech_start)
            overlap_end = min(end, speech_end)
            if overlap_end > overlap_start:
                cropped.append((overlap_start, overlap_end, speaker))
    cropped.sort(key=lambda turn: (turn[0], turn[1], turn[2]))
    return cropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group eval pytest eval/tests/test_diarize_gate.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/diarize_gate.py eval/tests/test_diarize_gate.py
git commit -m "feat(diarize): pure turn-to-speech cropping helper for VAD gating"
```

---

### Task 2: Server wiring — gate config, `/vad` fetch, fail-open crop

**Files:**
- Modify: `src/diarize_server.py` (imports block ~line 37-49; new code after `_env_float` ~line 75; endpoint tail ~lines 176-183)
- Test: `eval/tests/test_diarize_server_gate.py`

**Interfaces:**
- Consumes: `crop_turns_to_speech` from Task 1 (exact signature above).
- Produces: env contract used by Task 3's compose wiring — `DIARIZE_VAD_URL`, `DIARIZE_VAD_GATE`, `DIARIZE_VAD_THRESHOLD`, `DIARIZE_VAD_PAD_MS`, `DIARIZE_VAD_TIMEOUT`.

- [ ] **Step 1: Write the failing tests**

Create `eval/tests/test_diarize_server_gate.py`. Follows `eval/tests/test_clip_server.py`'s pattern: the heavy deps are stubbed in `sys.modules` before `diarize_server` is imported, and the module is skipped in the eval-run env (real pyannote would load the actual checkpoint at import).

```python
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
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("pyannote") is not None:
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

    def itertracks(self, yield_label: bool = True):
        """Yield (segment, track, label) like pyannote's Annotation."""
        for start, end, speaker in self._turns:
            yield types.SimpleNamespace(start=start, end=end), None, speaker

    def labels(self) -> list[str]:
        """Distinct labels, sorted, like pyannote's Annotation."""
        return sorted({speaker for _, _, speaker in self._turns})


_PIPELINE_TURNS = [(0.0, 4.0, "SPEAKER_00"), (10.0, 12.0, "SPEAKER_01")]


def _fake_pipeline(payload, **kwargs):
    """The stubbed pyannote pipeline: same two turns for any audio."""
    return _FakeAnnotation(list(_PIPELINE_TURNS))


# Stub the heavy imports before diarize_server is imported.
_torch_stub = types.ModuleType("torch")
_torch_stub.from_numpy = lambda array: _FakeWaveform()  # type: ignore[attr-defined]
sys.modules.setdefault("torch", _torch_stub)

_pipeline_stub = types.ModuleType("diarize_pipeline")
_pipeline_stub.DEFAULT_MODEL = "stub/diarize-model"  # type: ignore[attr-defined]
_pipeline_stub.build_pipeline = lambda **kwargs: _fake_pipeline  # type: ignore[attr-defined]
sys.modules["diarize_pipeline"] = _pipeline_stub

import diarize_server  # noqa: E402 — must come after the sys.modules stubs

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(diarize_server.app)

_UPLOAD = {"file": ("clip.wav", io.BytesIO(b"fake-bytes"), "audio/wav")}


@pytest.fixture(autouse=True)
def _decoded_audio(monkeypatch: pytest.MonkeyPatch):
    """Bypass ffmpeg: every upload decodes to a second of silence."""
    monkeypatch.setattr(diarize_server, "decode_audio", lambda data: np.zeros(16000, dtype=np.float32))


def test_no_vad_url_returns_ungated_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset DIARIZE_VAD_URL → byte-identical legacy behavior, no /vad call."""
    monkeypatch.delenv("DIARIZE_VAD_URL", raising=False)

    def _explode(*args, **kwargs):
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

    def _explode(*args, **kwargs):
        raise AssertionError("requests.post must not be called when the kill switch is off")

    monkeypatch.setattr(diarize_server.requests, "post", _explode)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    assert len(response.json()["segments"]) == 2


def test_gated_turns_are_cropped_to_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speech timeline [1,3] crops turn one and drops turn two entirely."""
    monkeypatch.setenv("DIARIZE_VAD_URL", "http://vad:8000")
    monkeypatch.delenv("DIARIZE_VAD_GATE", raising=False)
    calls: list[dict] = []

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"segments": [{"start": 1.0, "end": 3.0}], "has_speech": True, "sampling_rate": 16000}

    def _fake_post(url, files=None, data=None, timeout=None):
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

    def _boom(*args, **kwargs):
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
    calls: list[dict] = []

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"segments": [], "has_speech": False, "sampling_rate": 16000}

    def _fake_post(url, files=None, data=None, timeout=None):
        calls.append({"url": url, "data": data, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(diarize_server.requests, "post", _fake_post)
    response = client.post("/diarize", files=_UPLOAD)
    assert response.status_code == 200
    assert response.json()["segments"] == []  # no speech at all → everything gated away
    assert calls[0]["url"] == "http://vad:8000/vad"
    assert calls[0]["data"] == {"threshold": 0.3, "speech_pad_ms": 150}
    assert calls[0]["timeout"] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group eval pytest eval/tests/test_diarize_server_gate.py -v`
Expected: FAIL — `AttributeError: module 'diarize_server' has no attribute 'requests'` (or the ungated-only tests pass and the gating tests fail; either proves the feature is absent)

- [ ] **Step 3: Implement in `src/diarize_server.py`**

3a. Extend the imports block (after `import threading`, before `import torch`):

```python
from dataclasses import dataclass

import requests
```

(`requests` is already in both diarize images transitively via `huggingface_hub`; it becomes a direct import here.)

3b. Add after `_env_float` (below line 74), reusing its warn-and-fall-back convention:

```python
_FALSEY = frozenset({"off", "false", "no", "0"})


@dataclass(frozen=True)
class VadGateConfig:
    """Resolved VAD-gate settings for one request."""

    url: str
    threshold: float
    pad_ms: int
    timeout: float


def _load_vad_gate_config() -> VadGateConfig | None:
    """Resolve the VAD gate from the environment, or None when disabled.

    Gating is enabled by ``DIARIZE_VAD_URL`` (the full-stack compose sets it
    to the ``vad`` service) and vetoed by the ``DIARIZE_VAD_GATE`` kill
    switch. Read per-request so a compose-level env change only needs a
    container restart, and tests can flip it without reloading the module.

    Returns:
        The resolved config, or None when the URL is unset/blank or the
        kill switch is off.
    """
    url = (os.environ.get("DIARIZE_VAD_URL") or "").strip().rstrip("/")
    if not url:
        return None
    if (os.environ.get("DIARIZE_VAD_GATE") or "").strip().lower() in _FALSEY:
        return None
    threshold = _env_float("DIARIZE_VAD_THRESHOLD")
    pad_ms = _env_float("DIARIZE_VAD_PAD_MS")
    timeout = _env_float("DIARIZE_VAD_TIMEOUT")
    return VadGateConfig(
        url=url,
        threshold=0.4 if threshold is None else threshold,
        pad_ms=100 if pad_ms is None else int(pad_ms),
        timeout=30.0 if timeout is None else timeout,
    )


def _fetch_speech_timeline(
    audio_bytes: bytes, filename: str, config: VadGateConfig
) -> list[tuple[float, float]] | None:
    """POST the original upload to the vad service; None on any failure.

    Fail-open by design: a degraded gate must not take diarization down, so
    every failure mode (connection error, non-200, timeout, malformed body)
    logs one warning and returns None — the caller then skips gating.

    Args:
        audio_bytes: The raw uploaded media, forwarded as-is (the vad
            service decodes via ffmpeg itself, so no re-encode is needed).
        filename: Original upload filename, forwarded for the multipart part.
        config: The resolved gate settings.

    Returns:
        ``(start, end)`` speech intervals in seconds, or None on failure.
    """
    try:
        response = requests.post(
            f"{config.url}/vad",
            files={"file": (filename, audio_bytes)},
            data={"threshold": config.threshold, "speech_pad_ms": config.pad_ms},
            timeout=config.timeout,
        )
        response.raise_for_status()
        segments = response.json()["segments"]
        return [(float(segment["start"]), float(segment["end"])) for segment in segments]
    except Exception as exc:
        _log.warning("VAD gate unavailable (%s); returning ungated turns.", exc)
        return None
```

3c. Import the crop helper (with the other first-party imports, after
`from diarize_audio import ...`):

```python
from diarize_gate import crop_turns_to_speech
```

3d. Replace the endpoint tail (current lines 176-183, the `segments = [...]` /
`return DiarizeResponse(...)` block) with:

```python
    segments = [
        DiarizeSegment(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    speakers = [str(label) for label in annotation.labels()]

    # Optional VAD gate: crop turns to the Silero speech timeline, dropping
    # the music/noise the diarizer over-detects as speech. Fail-open — an
    # unavailable /vad leaves the turns ungated.
    gate = _load_vad_gate_config()
    if gate is not None:
        speech = _fetch_speech_timeline(audio_bytes, file.filename or "audio", gate)
        if speech is not None:
            cropped = crop_turns_to_speech(
                [(segment.start, segment.end, segment.speaker) for segment in segments],
                speech,
            )
            segments = [
                DiarizeSegment(start=start, end=end, speaker=speaker) for start, end, speaker in cropped
            ]
            speakers = sorted({segment.speaker for segment in segments})

    return DiarizeResponse(segments=segments, speakers=speakers)
```

3e. Extend the module docstring's `POST /diarize` paragraph (lines 10-18) with one sentence at the end:

```
        When ``DIARIZE_VAD_URL`` is set (the full-stack default), turns are
        cropped to the Silero ``/vad`` speech timeline before the response
        is built (fail-open; ``DIARIZE_VAD_GATE=off`` disables).
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `uv run --group eval pytest eval/tests/test_diarize_server_gate.py -v`
Expected: 5 passed

Run: `uv run --group eval pytest eval/tests/ -v`
Expected: all pass (no regressions; `test_diarize_server_gate.py` and `test_clip_server.py` may be skipped in an eval-run env — run them in the torch-free env)

- [ ] **Step 5: Commit**

```bash
git add src/diarize_server.py eval/tests/test_diarize_server_gate.py
git commit -m "feat(diarize): VAD-gate /diarize turns via the vad service (fail-open)"
```

---

### Task 3: Compose + `.env.example` wiring (full-stack default on)

**Files:**
- Modify: `docker/compose.yaml` (diarize service `environment:` block, ~line 322)
- Modify: `.env.example` (diarize knobs block, ~line 168)

**Interfaces:**
- Consumes: the `DIARIZE_VAD_*` env contract from Task 2.

- [ ] **Step 1: Wire the env into the full-stack diarize service**

In `docker/compose.yaml`, extend the `diarize` service `environment:` block (after `DIARIZE_DEVICE`):

```yaml
      # VAD gating: crop /diarize turns to the vad service's Silero speech
      # timeline (drops music/noise over-detected as speech; measured −35%
      # false alarm). On by default in the full stack; DIARIZE_VAD_GATE=off
      # disables, and the gate fails open if vad is unreachable. Request-time
      # call — no depends_on needed (router traffic starts after vad is
      # healthy anyway).
      DIARIZE_VAD_URL: ${DIARIZE_VAD_URL:-http://vad:8000}
      DIARIZE_VAD_GATE: ${DIARIZE_VAD_GATE:-true}
      DIARIZE_VAD_THRESHOLD: ${DIARIZE_VAD_THRESHOLD:-0.4}
      DIARIZE_VAD_PAD_MS: ${DIARIZE_VAD_PAD_MS:-100}
      DIARIZE_VAD_TIMEOUT: ${DIARIZE_VAD_TIMEOUT:-30}
```

- [ ] **Step 2: Document the knobs in `.env.example`**

After the `DIARIZE_SEG_MIN_DURATION_OFF` line (~168), add:

```bash
#
# VAD gating (full stack: ON by default — compose points DIARIZE_VAD_URL at
# the vad service). Crops /diarize turns to the Silero speech timeline,
# dropping music/noise over-detected as speech (measured −35% false alarm /
# −12.5% DER; eval/reports/2026-07-11-false-alarm-vad-gating.md). Fail-open:
# an unreachable vad leaves turns ungated. In diarize-only the URL is unset →
# gating off; set it to http://vad-only:8000 when co-deployed with vad-only.
# DIARIZE_VAD_GATE=off                  # kill switch (URL stays set)
# DIARIZE_VAD_URL=http://vad:8000       # unset → gating disabled
# DIARIZE_VAD_THRESHOLD=0.4             # Silero threshold (stock 0.5 over-cuts speech)
# DIARIZE_VAD_PAD_MS=100                # padding forwarded as speech_pad_ms
# DIARIZE_VAD_TIMEOUT=30                # /vad request timeout, seconds
```

- [ ] **Step 3: Validate the compose file parses**

Run: `docker compose --env-file .env -f docker/compose.yaml config --quiet && echo OK`
Expected: `OK` (no output before it)

- [ ] **Step 4: Commit**

```bash
git add docker/compose.yaml .env.example
git commit -m "feat(compose): enable diarize VAD gating by default in the full stack"
```

---

### Task 4: Docs + verification gate

**Files:**
- Modify: `README.md` (Diarization backend section — after the endpoint block)
- Modify: `CLAUDE.md` (Diarization backend paragraph)

**Interfaces:**
- Consumes: everything above; documentation only.

- [ ] **Step 1: README — add a "VAD gating" paragraph to the Diarization backend section**

After the `/diarize` endpoint block and before the model-identity paragraph, insert:

```markdown
**VAD gating (full stack: on by default).** Before responding, the service
crops the pipeline's turns to the Silero speech timeline fetched from the
stack's `vad` service (`DIARIZE_VAD_URL`, set to `http://vad:8000` by the
full-stack compose), dropping the music/noise the diarizer over-detects as
speech — measured −35% false alarm / −12.5% DER at the tuned
`DIARIZE_VAD_THRESHOLD=0.4` / `DIARIZE_VAD_PAD_MS=100`
(`eval/reports/2026-07-11-false-alarm-vad-gating.md`). Fail-open: an
unreachable `vad` logs a warning and returns ungated turns; the response
shape never changes. `DIARIZE_VAD_GATE=off` disables it. In `diarize-only`
the URL is unset (gating off) — co-deploy `vad-only` and set
`DIARIZE_VAD_URL=http://vad-only:8000` to gate there too. Consumers that
gate client-side (Nextext's `NEXTEXT_DIARIZE_VAD_GATE`) should disable
their gate once this is live — double-gating is harmless but wasteful.
```

- [ ] **Step 2: CLAUDE.md — extend the "Diarization backend (full stack)" section**

After the endpoint block's closing paragraph (the one ending "so the service returns raw turns only."), append one sentence to that paragraph:

```markdown
When `DIARIZE_VAD_URL` is set (the full-stack compose default,
`http://vad:8000`), the server VAD-gates its output first — turns are
cropped to the Silero `/vad` speech timeline (fail-open;
`DIARIZE_VAD_GATE=off` disables; tuned `DIARIZE_VAD_THRESHOLD=0.4` /
`DIARIZE_VAD_PAD_MS=100`), so music/noise false alarms are dropped
server-side for every consumer.
```

Also update the CLAUDE.md sentence listing the only Python sources (the "The only Python sources are …" list near the top) to include `src/diarize_gate.py` among the diarize helpers.

- [ ] **Step 3: Full verification**

```bash
uv run --group eval pytest eval/tests/ -v      # all green
make verify                                     # ruff + pyrefly green (torch-free venv: plain `uv sync` first if pyannote is installed locally)
```

Expected: all tests pass; pre-commit passes. (`make verify` mirrors CI only in the light venv — if `uv sync --group eval --group eval-run` was run locally, `uv sync` first, verify, then re-sync the eval groups.)

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document diarize backend VAD gating"
```

---

## Out of scope (from the spec)

- Per-request gate override fields on `/diarize`.
- Any change to the `vad` service or its contract.
- Nextext's `NEXTEXT_DIARIZE_VAD_GATE=off` flip (cross-repo follow-up).
- Real-clip validation of the backend gate (optional follow-up, mirrors the Fb validation flow).
