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
import os
import threading
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from diarize_audio import SAMPLE_RATE, decode_audio
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
    return DiarizeResponse(
        segments=segments,
        speakers=[str(label) for label in annotation.labels()],
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}
