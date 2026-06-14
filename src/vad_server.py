"""Voice-activity-detection server (Silero VAD).

Wraps the Silero VAD model in a tiny FastAPI app and exposes ``POST /vad`` so
consumer apps (Nextext) can drop their in-process ``torch.hub`` Silero load and
reach voice-activity detection from the shared inference stack — the same way
they already reach diarization, NER, rerank, and CLIP.

Endpoints:

    POST /vad
        Body multipart with ``file=<audio bytes>`` (any container ffmpeg can
        decode; resampled to 16 kHz mono server-side) plus optional float/int
        form fields ``threshold``, ``min_speech_duration_ms``,
        ``min_silence_duration_ms``, ``speech_pad_ms``, and
        ``max_speech_duration_s`` (each defaulting to Silero's own default when
        omitted). Returns ``{"segments": [{"start": <sec>, "end": <sec>}, ...],
        "has_speech": <bool>, "sampling_rate": 16000}`` with times in absolute
        seconds, in chronological order. Like ``/diarize``, the service returns
        raw turns — consumers reduce them (e.g. to a speech/no-speech gate)
        client-side.

    GET /health
        Liveness probe; returns ``{"status": "ok", "model": ..., "device": ...}``.

Model identity is fixed at container startup via ``VAD_MODEL`` (default
``silero_vad``, informational only — the ``silero-vad`` package bundles its
weights, so there is exactly one model and nothing is downloaded at runtime;
this shape is airgap-clean out of the box, unlike the gated diarize weights).
Set ``VAD_USE_ONNX=true`` to run the bundled ONNX graph instead of the Torch JIT
model. Loaded once at module import so the first request is hot.

Decoding goes through ffmpeg (same helper as ``diarize_server.py``): uploaded
bytes are piped to 16 kHz mono float32 PCM — the sample rate Silero requires.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16_000

MODEL_ID = os.environ.get("VAD_MODEL", "silero_vad")
DEVICE = os.environ.get("VAD_DEVICE", "cpu")
USE_ONNX = os.environ.get("VAD_USE_ONNX", "false").lower() == "true"

app = FastAPI(title="vllm-service vad", version="1.0")

model = load_silero_vad(onnx=USE_ONNX)

# Silero is fast, but the model is not safe to call concurrently; serialize so
# uploads queue instead of racing (mirrors diarize_server).
_model_lock = threading.Lock()


class VadSegment(BaseModel):
    """One detected speech turn, in absolute seconds from the start of the audio."""

    start: float
    end: float


class VadResponse(BaseModel):
    """Chronological speech turns plus a convenience speech/no-speech flag."""

    segments: list[VadSegment]
    has_speech: bool
    sampling_rate: int


def _decode_audio(data: bytes) -> np.ndarray:
    """Decode arbitrary media bytes to 16 kHz mono float32 PCM via ffmpeg.

    Mirrors ``diarize_server._decode_audio``: the same s16le -> float32 /
    32768 normalization ``whisper.load_audio`` performs, spooled through a
    temp file because MP4-family containers with a trailing moov atom cannot
    be demuxed from a non-seekable stdin pipe.

    Args:
        data: Raw bytes of any container/codec ffmpeg can decode.

    Returns:
        A 1-D float32 array of samples at ``SAMPLE_RATE``.

    Raises:
        ValueError: If ffmpeg fails or the payload holds no audio samples.
    """
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i",
                tmp.name,
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                "pipe:1",
            ],
            capture_output=True,
        )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"ffmpeg could not decode payload: {tail}")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("decoded audio contains no samples")
    return audio


@app.post("/vad", response_model=VadResponse)
def vad(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    min_speech_duration_ms: int | None = Form(default=None),
    min_silence_duration_ms: int | None = Form(default=None),
    speech_pad_ms: int | None = Form(default=None),
    max_speech_duration_s: float | None = Form(default=None),
) -> VadResponse:
    """Detect speech turns in an uploaded media file with Silero VAD.

    Declared ``def`` (not ``async``) on purpose: FastAPI then runs it on the
    threadpool, so a long decode cannot starve ``/health`` on the event loop.
    Every tuning knob is optional — omitted ones fall back to Silero's
    defaults.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        threshold: Speech-probability cutoff (Silero default ~0.5).
        min_speech_duration_ms: Drop speech chunks shorter than this.
        min_silence_duration_ms: Minimum silence to split two turns.
        speech_pad_ms: Padding added to each side of a detected turn.
        max_speech_duration_s: Force-split turns longer than this.

    Returns:
        VadResponse: Chronological speech turns plus a ``has_speech`` flag.
    """
    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio payload")
    try:
        audio = _decode_audio(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"failed to decode audio: {exc}") from exc

    opts: dict[str, float | int] = {}
    if threshold is not None:
        opts["threshold"] = threshold
    if min_speech_duration_ms is not None:
        opts["min_speech_duration_ms"] = min_speech_duration_ms
    if min_silence_duration_ms is not None:
        opts["min_silence_duration_ms"] = min_silence_duration_ms
    if speech_pad_ms is not None:
        opts["speech_pad_ms"] = speech_pad_ms
    if max_speech_duration_s is not None:
        opts["max_speech_duration_s"] = max_speech_duration_s

    waveform = torch.from_numpy(audio)
    try:
        with _model_lock:
            timestamps = get_speech_timestamps(
                waveform,
                model,
                sampling_rate=SAMPLE_RATE,
                return_seconds=True,
                **opts,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vad failed: {exc}") from exc

    segments = [VadSegment(start=float(ts["start"]), end=float(ts["end"])) for ts in timestamps]
    return VadResponse(segments=segments, has_speech=bool(segments), sampling_rate=SAMPLE_RATE)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}
