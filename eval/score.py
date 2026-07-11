"""Score diarization hypotheses against references with pyannote.metrics.

DER is accumulated across files by a single metric instance (Σ error / Σ
reference), never by averaging per-file rates. Speaker-count error is computed
directly from the annotations' distinct labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyannote.core import Annotation, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

# Detailed-component keys emitted by DiarizationErrorRate(detailed=True).
_CONFUSION = "confusion"
_MISS = "missed detection"
_FALSE_ALARM = "false alarm"
_TOTAL = "total"
_DER = "diarization error rate"


@dataclass(frozen=True)
class FileScore:
    """Per-file diarization scores (seconds for the duration components)."""

    uri: str
    der: float
    confusion: float
    false_alarm: float
    missed_detection: float
    total: float
    ref_speakers: int
    hyp_speakers: int
    speaker_count_error: int


@dataclass(frozen=True)
class RunReport:
    """Aggregate scores for a run over a set of files."""

    files: list[FileScore]
    overall_der: float
    overall_confusion: float
    overall_false_alarm: float
    overall_missed_detection: float
    speaker_count_mae: float
    speaker_count_bias: float

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Return one dict per file plus an OVERALL row for CSV export.

        Returns:
            A list of flat dictionaries suitable for ``csv.DictWriter``.
        """
        rows: list[dict[str, Any]] = [
            {
                "uri": f.uri,
                "der": round(f.der, 4),
                "confusion": round(f.confusion, 2),
                "false_alarm": round(f.false_alarm, 2),
                "missed_detection": round(f.missed_detection, 2),
                "ref_speakers": f.ref_speakers,
                "hyp_speakers": f.hyp_speakers,
                "speaker_count_error": f.speaker_count_error,
            }
            for f in self.files
        ]
        rows.append(
            {
                "uri": "OVERALL",
                "der": round(self.overall_der, 4),
                "confusion": round(self.overall_confusion, 2),
                "false_alarm": round(self.overall_false_alarm, 2),
                "missed_detection": round(self.overall_missed_detection, 2),
                "ref_speakers": "",
                "hyp_speakers": "",
                "speaker_count_error": round(self.speaker_count_mae, 3),
            }
        )
        return rows

    def to_markdown(self) -> str:
        """Render the report as a Markdown table.

        Returns:
            A Markdown string with a per-file row per file and an OVERALL row.
        """
        header = "| uri | DER | conf | FA | miss | ref# | hyp# | count_err |\n|---|---|---|---|---|---|---|---|\n"
        body = "".join(
            f"| {f.uri} | {f.der:.3f} | {f.confusion:.1f} | {f.false_alarm:.1f} | "
            f"{f.missed_detection:.1f} | {f.ref_speakers} | {f.hyp_speakers} | {f.speaker_count_error} |\n"
            for f in self.files
        )
        overall = (
            f"| **OVERALL** | **{self.overall_der:.3f}** | {self.overall_confusion:.1f} | "
            f"{self.overall_false_alarm:.1f} | {self.overall_missed_detection:.1f} | | | "
            f"MAE {self.speaker_count_mae:.2f} / bias {self.speaker_count_bias:+.2f} |\n"
        )
        return header + body + overall


def _rate(numerator: float, denominator: float) -> float:
    """Safe ratio; 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def _acc(metric: DiarizationErrorRate, key: str) -> float:
    """Read one accumulated component from a running ``DiarizationErrorRate``.

    ``DiarizationErrorRate.__getitem__`` is typed to return ``float |
    Dict[str, float]`` because the same accessor is shared with metrics whose
    components nest sub-dicts; the scalar components this module reads never
    do, so this narrows the union for pyrefly and at runtime.

    Args:
        metric: The metric instance accumulated across every scored file.
        key: One of the detailed-component keys (e.g. ``_CONFUSION``).

    Returns:
        The accumulated component value as a float.
    """
    value = metric[key]
    assert not isinstance(value, dict)
    return float(value)


def score_run(
    items: list[tuple[str, Annotation, Annotation, Timeline | None]],
    *,
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> RunReport:
    """Score reference/hypothesis pairs and aggregate DER + speaker-count error.

    Args:
        items: ``(uri, reference, hypothesis, uem)`` tuples; ``uem`` may be None.
        collar: Forgiveness collar in seconds around reference boundaries.
        skip_overlap: When True, exclude overlapped-speech regions from scoring.

    Returns:
        A RunReport with per-file scores and correctly-accumulated overall rates.
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    files: list[FileScore] = []
    count_errors: list[int] = []
    count_signed: list[int] = []
    for uri, reference, hypothesis, uem in items:
        components = metric(reference, hypothesis, uem=uem, detailed=True)
        assert isinstance(components, dict)
        total = float(components[_TOTAL])
        confusion = float(components[_CONFUSION])
        false_alarm = float(components[_FALSE_ALARM])
        missed = float(components[_MISS])
        ref_n = len(reference.labels())
        hyp_n = len(hypothesis.labels())
        files.append(
            FileScore(
                uri=uri,
                der=float(components[_DER]),
                confusion=confusion,
                false_alarm=false_alarm,
                missed_detection=missed,
                total=total,
                ref_speakers=ref_n,
                hyp_speakers=hyp_n,
                speaker_count_error=abs(ref_n - hyp_n),
            )
        )
        count_errors.append(abs(ref_n - hyp_n))
        count_signed.append(hyp_n - ref_n)

    return RunReport(
        files=files,
        overall_der=abs(metric),
        overall_confusion=_acc(metric, _CONFUSION),
        overall_false_alarm=_acc(metric, _FALSE_ALARM),
        overall_missed_detection=_acc(metric, _MISS),
        speaker_count_mae=_rate(sum(count_errors), len(count_errors)),
        speaker_count_bias=_rate(sum(count_signed), len(count_signed)),
    )
