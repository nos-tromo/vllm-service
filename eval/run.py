"""Run one diarization config over a set of audio files → hypothesis RTTM.

``run_diarization`` takes an already-built pipeline (any callable returning a
pyannote ``Annotation``) so it is testable without torch; ``main`` wires the real
``build_pipeline`` and a ``pyannote.database`` protocol.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
from pyannote.database import FileFinder, registry

# src/ is not a package; import the shared server helpers from it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from diarize_audio import SAMPLE_RATE, decode_audio
from eval.configs import DiarizeConfig


def _to_waveform(audio: np.ndarray) -> Any:
    """Convert 1-D float32 PCM to the ``(channel, time)`` tensor pyannote needs.

    pyannote's pipeline calls ``Tensor.unfold`` on the waveform, so it must be a
    torch tensor, not a numpy array. torch is imported lazily here — never at
    module load — so importing this module (and the fake-pipeline unit tests,
    which monkeypatch this function) stays torch-free. Mirrors the ``/diarize``
    server's ``torch.from_numpy(audio).unsqueeze(0)`` exactly (eval == production).

    Args:
        audio: 1-D float32 PCM samples at ``SAMPLE_RATE`` from ``decode_audio``.

    Returns:
        A ``(1, time)`` float32 torch tensor.
    """
    import torch

    return torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)


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
        waveform = _to_waveform(audio)
        result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": uri}, **config.pipeline_kwargs)
        # pyannote.audio 4.x returns a DiarizeOutput wrapper; 3.x (and the fake
        # test pipeline) return the Annotation directly. `.speaker_diarization`
        # is the standard overlap-allowing Annotation, matching 3.x's output.
        annotation: Any = getattr(result, "speaker_diarization", result)
        out_path = os.path.join(out_dir, f"{uri}.rttm")
        with open(out_path, "w", encoding="utf-8") as rttm:
            annotation.write_rttm(rttm)
        written.append(out_path)
    return written


def main() -> None:
    """CLI: build the pipeline for a config and run it over a database protocol.

    Loads the ``pyannote.database`` registry written by ``eval/prepare_data.py``,
    resolves the requested ``<Name>.SpeakerDiarization.Benchmark`` protocol's
    ``test`` files (audio paths resolved via ``FileFinder``), builds the
    pipeline via ``build_pipeline`` for the given config, and writes one
    hypothesis RTTM per file into ``--out-dir``. This only writes hypotheses —
    it does not score them; see ``eval/sweep.py`` (even for a single config)
    for a scored report. See ``eval/README.md`` for the full runbook.
    """
    parser = argparse.ArgumentParser(description="Run a diarization config over a pyannote.database protocol.")
    parser.add_argument("--database", default="data/database.yml", help="Path to the pyannote.database database.yml.")
    parser.add_argument(
        "--protocol",
        default="VoxConverse.SpeakerDiarization.Benchmark",
        help="Full protocol name, e.g. VoxConverse.SpeakerDiarization.Benchmark.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write hypothesis RTTM files into.")
    parser.add_argument("--label", default="baseline", help="Config label, used as the report row key.")
    parser.add_argument("--model", default=None, help="pyannote pipeline id; default: DIARIZE_MODEL env or 3.1.")
    parser.add_argument("--device", default=None, help="Torch device; default: DIARIZE_DEVICE env or cuda.")
    parser.add_argument("--clustering-threshold", type=float, default=None, help="Clustering-threshold override.")
    parser.add_argument("--min-speakers", type=int, default=None, help="Lower bound on the speaker count.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Upper bound on the speaker count.")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact speaker count, if known.")
    args = parser.parse_args()

    registry.load_database(args.database)
    protocol = registry.get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
    files = [(str(f["uri"]), str(f["audio"])) for f in protocol.test()]

    config = DiarizeConfig(
        label=args.label,
        model_id=args.model,
        device=args.device,
        clustering_threshold=args.clustering_threshold,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    from diarize_pipeline import build_pipeline  # lazy: torch only on a real run

    pipeline = build_pipeline(
        model_id=config.model_id,
        device=config.device,
        clustering_threshold=config.clustering_threshold,
        segmentation_min_duration_off=config.segmentation_min_duration_off,
    )
    written = run_diarization(pipeline, files, args.out_dir, config)
    print(f"wrote {len(written)} hypothesis RTTM file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
