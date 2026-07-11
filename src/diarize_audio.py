"""Decode arbitrary media bytes to 16 kHz mono float32 PCM via ffmpeg.

Extracted from ``diarize_server`` so the eval harness decodes audio through the
exact same path the server uses (eval == production).
"""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np

SAMPLE_RATE = 16_000


def decode_audio(data: bytes) -> np.ndarray:
    """Decode media bytes to a 1-D float32 array at ``SAMPLE_RATE``.

    Applies the same s16le -> float32 / 32768 normalization that
    ``whisper.load_audio`` performs. The bytes are spooled to a temp file
    because MP4-family containers with a trailing moov atom cannot be demuxed
    from a non-seekable stdin pipe.

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
