"""ASR (Whisper) server for the asr-only deployment shape.

Wraps openai-whisper in a tiny FastAPI app and exposes the OpenAI-compatible
``POST /v1/audio/transcriptions`` (and ``/v1/audio/translations``) routes — the
same contract the full-stack vLLM ``asr`` service serves — so consumers
(Nextext's ``ExternalWhisperTranscriber``) target either backend by changing
only the base URL.

This is the CPU counterpart to the full-stack ``asr`` service: that one runs
Whisper on vLLM (CUDA-only); this one runs the reference openai-whisper decoder
(the same one Nextext loads in-process), so a non-CUDA / Ollama host can offer
ASR alongside the other ``-only`` shapes. It mirrors how ``rerank_server.py``
reimplements the cross-encoder forward pass rather than running vLLM.

Endpoints:

    POST /v1/audio/transcriptions
        OpenAI-compatible multipart form: ``file=<audio bytes>`` (any container
        ffmpeg can decode; resampled to 16 kHz mono server-side) plus optional
        ``model`` (accepted but ignored — the server always uses the model it
        loaded at boot, like ``rerank_server.py``), ``language``, ``prompt``,
        ``temperature``, and ``response_format`` (``json`` [default],
        ``verbose_json``, or ``text``). ``verbose_json`` returns per-segment
        ``no_speech_prob`` and the detected ``language`` — the fields Nextext
        filters on.

    POST /v1/audio/translations
        Same contract, but translates into English (Whisper ``task=translate``).

    GET /health
        Liveness probe; returns ``{"status": "ok", "model": ..., "device": ...}``.

Model identity is fixed at container startup via ``WHISPER_MODEL`` (default
``openai/whisper-large-v3``), mapped to the openai-whisper checkpoint name by
stripping the Hugging Face ``openai/whisper-`` prefix (so
``openai/whisper-large-v3`` loads the ``large-v3`` weights). Loaded once at
module import so the first request is hot. Weights download to a subdirectory of
the shared huggingface-cache volume (``ASR_DOWNLOAD_ROOT``); openai-whisper
fetches from its own CDN rather than the HF Hub, so prime the cache once on a
networked host (the weights are public — no gated access).

Decoding goes through ffmpeg (same as ``diarize_server.py``): uploaded bytes are
piped to 16 kHz mono float32 PCM and handed to ``model.transcribe`` as a
pre-decoded waveform, so any container ffmpeg can read is accepted.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from typing import Any

import numpy as np
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

SAMPLE_RATE = 16_000

MODEL_ID = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")
DEVICE = os.environ.get("ASR_DEVICE", "cpu")
# Subdirectory of the mounted huggingface-cache volume (mounted at
# /root/.cache/huggingface/hub) so downloaded Whisper weights persist there.
DOWNLOAD_ROOT = os.environ.get("ASR_DOWNLOAD_ROOT", "/root/.cache/huggingface/hub/whisper")


def _whisper_name(model_id: str) -> str:
    """Map a Hugging Face Whisper id to an openai-whisper checkpoint name.

    openai-whisper's ``load_model`` takes short names (``large-v3``,
    ``base``, ...) or a local path, not Hugging Face ids. Strip the
    ``<org>/`` prefix and a leading ``whisper-`` so the full-stack
    ``WHISPER_MODEL=openai/whisper-large-v3`` resolves to ``large-v3``. An
    ``ASR_WHISPER_NAME`` override wins outright (custom checkpoints or
    on-disk paths).

    Args:
        model_id: The configured ``WHISPER_MODEL`` value.

    Returns:
        The openai-whisper checkpoint name (or path) to load.
    """
    override = os.environ.get("ASR_WHISPER_NAME", "").strip()
    if override:
        return override
    return model_id.rsplit("/", 1)[-1].removeprefix("whisper-")


app = FastAPI(title="vllm-service asr-only", version="1.0")

os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
_model = whisper.load_model(_whisper_name(MODEL_ID), device=DEVICE, download_root=DOWNLOAD_ROOT)

# Whisper decoding runs for seconds-to-minutes and the model is not safe to call
# concurrently; serialize requests so uploads queue instead of racing.
_model_lock = threading.Lock()


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


def _transcribe(
    file: UploadFile,
    task: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    response_format: str,
) -> Any:
    """Decode an upload and run Whisper, shaping the OpenAI response.

    Declared on the ``def`` (threadpool) endpoints below so a minutes-long
    decode never starves ``/health`` on the event loop; the module-level
    lock serializes the non-reentrant model.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        task: ``transcribe`` (verbatim) or ``translate`` (to English).
        language: Source language hint; ``None`` lets Whisper auto-detect.
        prompt: Optional decoding prompt (``initial_prompt``).
        temperature: Optional sampling temperature; ``None`` uses Whisper's
            built-in fallback schedule.
        response_format: ``json`` (default), ``verbose_json``, or ``text``.

    Returns:
        A dict (serialized as JSON) for ``json`` / ``verbose_json``, or a
        ``PlainTextResponse`` for ``text``.
    """
    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio payload")
    try:
        audio = _decode_audio(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"failed to decode audio: {exc}") from exc

    options: dict[str, Any] = {"task": task, "fp16": DEVICE != "cpu"}
    if language:
        options["language"] = language
    if prompt:
        options["initial_prompt"] = prompt
    if temperature is not None:
        options["temperature"] = temperature
    try:
        with _model_lock:
            result = _model.transcribe(audio, **options)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc

    text = str(result.get("text", "")).strip()
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        segments = [
            {
                "id": int(seg.get("id", i)),
                "seek": int(seg.get("seek", 0)),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")),
                "tokens": list(seg.get("tokens", [])),
                "temperature": float(seg.get("temperature", 0.0)),
                "avg_logprob": float(seg.get("avg_logprob", 0.0)),
                "compression_ratio": float(seg.get("compression_ratio", 0.0)),
                "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
            }
            for i, seg in enumerate(result.get("segments", []))
        ]
        return {
            "task": task,
            "language": result.get("language"),
            "duration": len(audio) / SAMPLE_RATE,
            "text": text,
            "segments": segments,
        }
    return {"text": text}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency marker
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    response_format: str = Form(default="json"),
) -> Any:
    """Transcribe an uploaded media file (Whisper ``task=transcribe``).

    ``model`` is accepted for OpenAI-client compatibility but ignored — the
    server always uses the model it loaded at boot.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        model: Ignored; present for OpenAI-client compatibility.
        language: Source language hint; ``None`` auto-detects.
        prompt: Optional decoding prompt.
        temperature: Optional sampling temperature.
        response_format: ``json`` (default), ``verbose_json``, or ``text``.

    Returns:
        The transcription in the requested ``response_format``.
    """
    return _transcribe(file, "transcribe", language, prompt, temperature, response_format)


@app.post("/v1/audio/translations")
def translations(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency marker
    model: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    response_format: str = Form(default="json"),
) -> Any:
    """Translate an uploaded media file into English (Whisper ``task=translate``).

    OpenAI's translation endpoint takes no ``language`` field — Whisper always
    emits English here. ``model`` is accepted but ignored.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        model: Ignored; present for OpenAI-client compatibility.
        prompt: Optional decoding prompt.
        temperature: Optional sampling temperature.
        response_format: ``json`` (default), ``verbose_json``, or ``text``.

    Returns:
        The English translation in the requested ``response_format``.
    """
    return _transcribe(file, "translate", None, prompt, temperature, response_format)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}
