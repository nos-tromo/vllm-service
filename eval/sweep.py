"""Compare diarization configurations by DER.

``summarize_sweep`` renders a ranked table from ``(config, RunReport)`` pairs
and stays pure/torch-free — it is unit-tested without pyannote.audio or data.
``run_sweep`` is the orchestration layer: it loads a protocol once via
``pyannote.database``, then for each config builds a pipeline, runs it, scores
it, and logs progress, so a sweep never silently drops a config from the
final table. ``main`` wires ``run_sweep`` to argparse; see ``eval/README.md``
for the runbook.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pyannote.core import Annotation, Timeline
from pyannote.database import FileFinder, registry
from pyannote.database.util import load_rttm

from eval.configs import DiarizeConfig
from eval.score import RunReport, score_run

logger = logging.getLogger(__name__)


def summarize_sweep(results: list[tuple[DiarizeConfig, RunReport]]) -> str:
    """Render a DER-ranked comparison table over the swept configurations.

    Args:
        results: One ``(config, report)`` pair per configuration evaluated.

    Returns:
        A Markdown table sorted ascending by overall DER (best first), one row
        per configuration.
    """
    ranked = sorted(results, key=lambda pair: pair[1].overall_der)
    header = "| config | DER | conf | FA | miss | count MAE |\n|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {cfg.label} | {rep.overall_der:.3f} | {rep.overall_confusion:.1f} | "
        f"{rep.overall_false_alarm:.1f} | {rep.overall_missed_detection:.1f} | {rep.speaker_count_mae:.2f} |\n"
        for cfg, rep in ranked
    )
    return header + body


def _hypothesis_annotation(rttm_path: str, uri: str) -> Annotation:
    """Load a single-file hypothesis RTTM back into an Annotation.

    ``run_diarization`` writes one RTTM per uri; this reads it back for
    scoring. Falls back to an empty Annotation when the file has no speaker
    turns (e.g. a diarization run that found no speech) rather than raising,
    so one degenerate file scores as a total miss instead of aborting the sweep.

    Args:
        rttm_path: Path to the hypothesis RTTM written by ``run_diarization``.
        uri: The recording id this hypothesis belongs to, used for the
            fallback empty Annotation's identity.

    Returns:
        The hypothesis as a pyannote ``Annotation``.
    """
    loaded = load_rttm(rttm_path)
    return next(iter(loaded.values()), Annotation(uri=uri))


def run_sweep(
    database: str,
    protocol_name: str,
    configs: list[DiarizeConfig],
    out_root: str,
    *,
    collar: float = 0.25,
) -> str:
    """Run and score every config over a protocol, then rank them by DER.

    Loads the protocol once via ``pyannote.database`` (shared across every
    config). For each config: builds a fresh pipeline, runs it over every
    ``test`` file into ``<out_root>/<label>/``, reloads the written hypothesis
    RTTMs, scores them against the protocol's references/UEM, and logs the
    config's start and outcome — so a config that errors or scores
    unexpectedly never silently vanishes from the final table.

    ``build_pipeline``/``run_diarization`` are imported lazily so importing
    this module (e.g. for the pure ``summarize_sweep`` unit tests) never pulls
    in torch.

    Args:
        database: Path to the ``pyannote.database`` ``database.yml`` (see
            ``eval/prepare_data.py``).
        protocol_name: Full protocol name, e.g.
            ``VoxConverse.SpeakerDiarization.Benchmark``.
        configs: The configurations to evaluate, in order.
        out_root: Root directory; each config's hypotheses are written under
            a ``<label>`` subdirectory.
        collar: Forgiveness collar in seconds, forwarded to ``score_run``.

    Returns:
        The Markdown comparison table from ``summarize_sweep``, ranked by
        ascending DER (best first).
    """
    from diarize_pipeline import build_pipeline  # lazy: torch only on a real run
    from eval.run import run_diarization  # lazy: keeps this module's own import torch-free

    registry.load_database(database)
    protocol = registry.get_protocol(protocol_name, preprocessors={"audio": FileFinder()})
    protocol_files = list(protocol.test())
    files = [(str(f["uri"]), str(f["audio"])) for f in protocol_files]
    references: dict[str, tuple[Annotation, Timeline | None]] = {
        str(f["uri"]): (f["annotation"], f["annotated"]) for f in protocol_files
    }
    logger.info("sweep: loaded protocol %r with %d file(s)", protocol_name, len(files))

    results: list[tuple[DiarizeConfig, RunReport]] = []
    for index, config in enumerate(configs, start=1):
        logger.info("sweep: [%d/%d] running config %r", index, len(configs), config.label)
        pipeline = build_pipeline(
            model_id=config.model_id,
            device=config.device,
            clustering_threshold=config.clustering_threshold,
            segmentation_min_duration_off=config.segmentation_min_duration_off,
        )
        out_dir = os.path.join(out_root, config.label)
        hyp_paths = run_diarization(pipeline, files, out_dir, config)

        items: list[tuple[str, Annotation, Annotation, Timeline | None]] = []
        for (uri, _audio_path), hyp_path in zip(files, hyp_paths, strict=True):
            reference, uem = references[uri]
            items.append((uri, reference, _hypothesis_annotation(hyp_path, uri), uem))
        report = score_run(items, collar=collar)

        logger.info(
            "sweep: [%d/%d] config %r scored overall DER=%.3f (%d file(s))",
            index,
            len(configs),
            config.label,
            report.overall_der,
            len(report.files),
        )
        results.append((config, report))

    return summarize_sweep(results)


def main() -> None:
    """CLI: run and score a grid of configs (given as JSON) over a protocol.

    ``--configs`` points at a JSON file holding a list of objects; each
    becomes one ``DiarizeConfig`` (``label`` is required, every other field is
    optional and defaults to None — the same fields ``DiarizeConfig`` accepts).
    Prints the ranked Markdown comparison table; ``--report`` additionally
    writes it to a file. See ``eval/README.md`` for the full runbook and an
    example configs file.
    """
    parser = argparse.ArgumentParser(description="Run and score a grid of diarization configs over a protocol.")
    parser.add_argument("--database", default="data/database.yml", help="Path to the pyannote.database database.yml.")
    parser.add_argument(
        "--protocol",
        default="VoxConverse.SpeakerDiarization.Benchmark",
        help="Full protocol name, e.g. VoxConverse.SpeakerDiarization.Benchmark.",
    )
    parser.add_argument("--out-root", required=True, help="Root directory; each config writes to <out-root>/<label>/.")
    parser.add_argument(
        "--configs", required=True, help="Path to a JSON file listing the configs to sweep (see eval/README.md)."
    )
    parser.add_argument("--collar", type=float, default=0.25, help="Forgiveness collar in seconds (default: 0.25).")
    parser.add_argument("--report", default=None, help="Optional path to also write the Markdown table to.")
    args = parser.parse_args()

    with open(args.configs, encoding="utf-8") as handle:
        raw_configs = json.load(handle)
    configs = [DiarizeConfig(**entry) for entry in raw_configs]

    table = run_sweep(args.database, args.protocol, configs, args.out_root, collar=args.collar)
    print(table)
    if args.report:
        Path(args.report).write_text(table, encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
