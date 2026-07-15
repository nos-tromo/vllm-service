"""Compatibility shims for running pyannote.audio 3.x on this image.

Two independent mismatches between pyannote.audio 3.3.x and the pinned PyTorch
base image are patched here; import this module BEFORE importing
``pyannote.audio``.

1. Missing torchaudio symbols. pyannote.audio 3.3.x references a few
   ``torchaudio`` symbols at import time that torchaudio 2.9+ removed when it
   moved file decoding to torchcodec:

   * ``torchaudio.AudioMetaData`` — used as a return annotation in
     ``pyannote/audio/core/io.py`` (evaluated at module import, so its mere
     absence is a hard ImportError).
   * ``torchaudio.list_audio_backends`` / ``torchaudio.info`` — used for
     backend probing when constructing the ``Audio`` helper.

   The diarize server never decodes through torchaudio — audio arrives as a
   pre-decoded ``{"waveform", "sample_rate"}`` dict (ffmpeg does the decoding)
   — so restoring these names with inert stubs lets the pyannote import succeed
   without affecting any code path the server exercises. Each stub is guarded
   by ``hasattr``, so this stays a no-op on torchaudio versions that still ship
   the symbols.

2. torch.load weights_only default. PyTorch 2.6 flipped ``torch.load``'s
   ``weights_only`` default to True, and pyannote loads its checkpoints through
   Lightning's ``pl_load`` -> ``torch.load`` without overriding it. The trusted
   gated pyannote checkpoints pickle a few non-tensor globals the safe
   unpickler rejects, so ``TRUSTED_CHECKPOINT_GLOBALS`` (below) allowlists
   exactly those. That keeps loading on the ``weights_only=True`` path rather
   than disabling it wholesale, which would re-enable arbitrary-object
   unpickling for every ``torch.load`` in the process.

The torchaudio stubs (1) must be applied before ``pyannote.audio`` is imported;
the allowlist (2) imports pyannote task types and so runs after them, lower in
this module.
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


# setattr (not plain assignment): these attributes are absent from installed
# torchaudio's own surface — that's the point of the shim — so direct
# assignment fails pyrefly wherever torchaudio is actually installed (the
# dev eval-run venv; CI lints with torchaudio as Any and sees no attributes
# either way).
if not hasattr(torchaudio, "AudioMetaData"):
    setattr(torchaudio, "AudioMetaData", _AudioMetaData)  # noqa: B010

if not hasattr(torchaudio, "list_audio_backends"):
    setattr(torchaudio, "list_audio_backends", lambda: ["soundfile"])  # noqa: B010

if not hasattr(torchaudio, "info"):
    setattr(torchaudio, "info", _unavailable)  # noqa: B010


# (2) torch.load weights_only allowlist — see the module docstring. Imported
# here, AFTER the torchaudio stubs above, so importing pyannote's task types
# succeeds. Exposed as a tuple so the build-time smoke test
# (Dockerfile.diarize.cuda) can round-trip these exact globals through the
# weights_only loader without the gated weights present. A base-image bump that
# renames one of these symbols breaks this import (caught at build); one that
# makes a checkpoint pickle a NEW global makes the loader raise naming it at
# runtime, to be appended here.
import torch  # noqa: E402
from pyannote.audio.core.task import Problem, Resolution, Specifications  # noqa: E402
from torch.torch_version import TorchVersion  # noqa: E402

TRUSTED_CHECKPOINT_GLOBALS = (TorchVersion, Specifications, Problem, Resolution)
torch.serialization.add_safe_globals(list(TRUSTED_CHECKPOINT_GLOBALS))
