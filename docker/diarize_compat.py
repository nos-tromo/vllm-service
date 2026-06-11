"""torchaudio compatibility shims for pyannote.audio 3.x.

pyannote.audio 3.3.x references a few ``torchaudio`` symbols at import time
that torchaudio 2.9+ removed when it moved file decoding to torchcodec:

* ``torchaudio.AudioMetaData`` — used as a return annotation in
  ``pyannote/audio/core/io.py`` (evaluated at module import, so its mere
  absence is a hard ImportError).
* ``torchaudio.list_audio_backends`` / ``torchaudio.info`` — used for
  backend probing when constructing the ``Audio`` helper.

The diarize server never decodes through torchaudio — audio arrives as a
pre-decoded ``{"waveform", "sample_rate"}`` dict (ffmpeg does the
decoding) — so restoring these names with inert stubs lets the pyannote
import succeed without affecting any code path the server exercises.

Import this module BEFORE importing ``pyannote.audio``. Importing it is a
no-op on torchaudio versions that still ship the symbols (each stub is
guarded by ``hasattr``), so it is safe across the whole 2.x line.
"""

from __future__ import annotations

import torchaudio


class _AudioMetaData:
    """Inert stand-in for the removed ``torchaudio.AudioMetaData`` type.

    Only ever referenced as a type annotation in pyannote's I/O module,
    which the waveform-dict path does not call, so it never needs fields.
    """


def _unavailable(*_args: object, **_kwargs: object) -> object:
    """Raise for torchaudio file-I/O entry points the server never uses."""
    raise RuntimeError(
        "torchaudio file I/O is unavailable in this image; the diarize "
        "server decodes audio via ffmpeg and feeds pyannote a pre-decoded "
        "waveform dict."
    )


if not hasattr(torchaudio, "AudioMetaData"):
    torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "info"):
    torchaudio.info = _unavailable
