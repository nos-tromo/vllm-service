"""Run one diarization config over a set of audio files → hypothesis RTTM.

``run_diarization`` takes an already-built pipeline (any callable returning a
pyannote ``Annotation``) so it is testable without torch; ``main`` wires the real
``build_pipeline`` and a ``pyannote.database`` protocol.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

# src/ is not a package; import the shared server helpers from it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from diarize_audio import SAMPLE_RATE, decode_audio
from eval.configs import DiarizeConfig


def run_diarization(pipeline: Any, files: list[tuple[str, str]], out_dir: str, config: DiarizeConfig) -> list[str]:
    """Diarize each file and write one hypothesis RTTM per recording.

    Args:
        pipeline: Callable ``(payload, **bounds) -> Annotation`` (real or fake).
        files: ``(uri, audio_path)`` pairs to process.
        out_dir: Directory to write ``<uri>.rttm`` into (created if absent).
        config: The configuration; its ``pipeline_kwargs`` are passed through.

    Returns:
        The list of written RTTM paths, in input order.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for uri, audio_path in files:
        with open(audio_path, "rb") as handle:
            audio = decode_audio(handle.read())
        waveform = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        annotation = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": uri}, **config.pipeline_kwargs)
        out_path = os.path.join(out_dir, f"{uri}.rttm")
        with open(out_path, "w", encoding="utf-8") as rttm:
            annotation.write_rttm(rttm)
        written.append(out_path)
    return written


def main() -> None:
    """CLI: build the pipeline for a config and run it over a database protocol.

    Loads ``data/database.yml`` (see ``eval/prepare_data.py``), resolves the
    requested ``<Name>.SpeakerDiarization.Benchmark`` protocol's ``test`` files,
    builds the pipeline via ``build_pipeline`` and writes hypotheses. See
    ``eval/README.md`` for arguments.
    """
    raise NotImplementedError("Wire argparse + pyannote.database in Task 8's runbook; see eval/README.md.")
