"""Speaker-diarization server (pyannote/speaker-diarization-3.1).

Mirrors the pyannote pipeline Nextext runs in-process
(``nextext.core.transcription``) over HTTP so Nextext can drop its local
pyannote runtime and reach diarization from the shared inference stack,
the same way docint reaches GLiNER, rerank, and CLIP.

Endpoints:

    POST /diarize
        Body multipart with ``file=<audio bytes>`` (any container ffmpeg
        can decode; resampled to 16 kHz mono server-side) plus optional
        integer form fields ``num_speakers`` (exact count) or
        ``min_speakers``/``max_speakers`` (bounds). ``num_speakers`` is
        mutually exclusive with the bounds — combining them returns 400.
        Returns ``{"segments": [{"start": <sec>, "end": <sec>,
        "speaker": "SPEAKER_00"}, ...], "speakers": ["SPEAKER_00", ...]}``
        with times in absolute seconds, segments in chronological order.
        When ``DIARIZE_VAD_URL`` is set (the full-stack default), turns are
        cropped to the Silero ``/vad`` speech timeline before the response
        is built (fail-open; ``DIARIZE_VAD_GATE=off`` disables).

    GET /health
        Liveness probe; returns ``{"status": "ok", "model": ...,
        "device": ...}``.

Model identity is fixed at container startup via ``DIARIZE_MODEL``
(default ``pyannote/speaker-diarization-3.1``). Loaded once at module
import so the first request is hot. The checkpoints are gated on the
Hugging Face Hub — see .env.example for the one-time download procedure.

Decoding never goes through torchaudio: uploaded bytes are piped through
``ffmpeg`` to 16 kHz mono float32 PCM (the same normalization
``whisper.load_audio`` performs) and handed to the pipeline as a
pre-decoded ``{"waveform", "sample_rate"}`` dict. Speaker-to-ASR-segment
alignment stays client-side: consumers assign speakers to their
transcription segments by maximum overlap.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from diarize_audio import SAMPLE_RATE, decode_audio
from diarize_gate import crop_turns_to_speech
from diarize_pipeline import DEFAULT_MODEL, build_pipeline

MODEL_ID = os.environ.get("DIARIZE_MODEL", DEFAULT_MODEL)  # shared default → /health can't drift from the loaded model
DEVICE = os.environ.get("DIARIZE_DEVICE", "cuda")

_log = logging.getLogger("diarize")


def _env_float(name: str) -> float | None:
    """Read an optional float clustering override from the environment.

    Args:
        name: Environment variable name.

    Returns:
        The parsed float, or None when unset/blank/unparseable. An unparseable
        value warns and falls back so a typo cannot crash server startup.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        _log.warning("Ignoring %s=%r: not a float; using the pipeline default.", name, raw)
        return None


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
        kill switch is off. A non-finite (``nan``/``inf``) override is
        treated the same as unset — it falls back to the default rather
        than raising.
    """
    url = (os.environ.get("DIARIZE_VAD_URL") or "").strip().rstrip("/")
    if not url:
        return None
    if (os.environ.get("DIARIZE_VAD_GATE") or "").strip().lower() in _FALSEY:
        return None
    threshold = _env_float("DIARIZE_VAD_THRESHOLD")
    pad_ms = _env_float("DIARIZE_VAD_PAD_MS")
    timeout = _env_float("DIARIZE_VAD_TIMEOUT")
    if threshold is not None and not math.isfinite(threshold):
        _log.warning("Ignoring DIARIZE_VAD_THRESHOLD=%r: not finite; using the default.", threshold)
        threshold = None
    if pad_ms is not None and not math.isfinite(pad_ms):
        _log.warning("Ignoring DIARIZE_VAD_PAD_MS=%r: not finite; using the default.", pad_ms)
        pad_ms = None
    if timeout is not None and not math.isfinite(timeout):
        _log.warning("Ignoring DIARIZE_VAD_TIMEOUT=%r: not finite; using the default.", timeout)
        timeout = None
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


app = FastAPI(title="vllm-service diarize", version="1.0")

# Optional clustering-hyperparameter overrides. All unset → stock model defaults
# (byte-identical to before). DIARIZE_FB is community-1's speaker-granularity knob
# (lower → more speakers); see src/diarize_pipeline.py::_resolve_param_overrides.
pipeline = build_pipeline(
    clustering_threshold=_env_float("DIARIZE_CLUSTERING_THRESHOLD"),
    segmentation_min_duration_off=_env_float("DIARIZE_SEG_MIN_DURATION_OFF"),
    fa=_env_float("DIARIZE_FA"),
    fb=_env_float("DIARIZE_FB"),
)

# Diarization runs for seconds-to-minutes on the device; serialize requests
# so concurrent uploads queue instead of contending for GPU memory.
_pipeline_lock = threading.Lock()


class DiarizeSegment(BaseModel):
    """One speaker turn, in absolute seconds from the start of the audio."""

    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    """Chronological speaker turns plus the distinct speaker labels."""

    segments: list[DiarizeSegment]
    speakers: list[str]


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency marker
    num_speakers: int | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
) -> DiarizeResponse:
    """Run speaker diarization on an uploaded media file.

    ``num_speakers`` (exact count) is mutually exclusive with the
    ``min_speakers``/``max_speakers`` bounds — pyannote would silently
    ignore the bounds when both are given, so combining them is rejected
    with 400 instead. Declared ``def`` (not ``async``) on purpose:
    FastAPI then runs it on the threadpool, so a minutes-long pipeline
    call cannot starve ``/health`` on the event loop.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        num_speakers: Exact number of speakers, if known.
        min_speakers: Lower bound on the number of speakers.
        max_speakers: Upper bound on the number of speakers.

    Returns:
        DiarizeResponse: Chronological speaker turns and distinct labels.
    """
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise HTTPException(
            status_code=400,
            detail="num_speakers is mutually exclusive with min_speakers/max_speakers",
        )
    for name, value in (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None and value < 1:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 1")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise HTTPException(status_code=400, detail="min_speakers must be <= max_speakers")

    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio payload")
    try:
        audio = decode_audio(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"failed to decode audio: {exc}") from exc

    waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel=1, time)
    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    try:
        with _pipeline_lock:
            result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"diarization failed: {exc}") from exc

    # pyannote.audio 4.x returns a DiarizeOutput wrapper; 3.x returns the Annotation
    # directly. `.speaker_diarization` is the standard, overlap-allowing Annotation
    # (matches 3.x semantics — consumers align speakers by overlap).
    annotation: Any = getattr(result, "speaker_diarization", result)

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
            segments = [DiarizeSegment(start=start, end=end, speaker=speaker) for start, end, speaker in cropped]
            speakers = sorted({segment.speaker for segment in segments})

    return DiarizeResponse(segments=segments, speakers=speakers)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}
