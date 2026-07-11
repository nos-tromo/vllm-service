"""Compare diarization configurations by DER.

``summarize_sweep`` renders a ranked table from ``(config, RunReport)`` pairs.
The actual run/score orchestration is driven from ``main`` (Task 8 runbook) so
that the pure table rendering stays unit-testable without torch or data.
"""

from __future__ import annotations

from eval.configs import DiarizeConfig
from eval.score import RunReport


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
