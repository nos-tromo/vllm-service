"""Segment-level speaker accuracy + turn-boundary F1 for Nextext transcripts.

This scores the pipeline's end-to-end output (timestamped, speaker-labelled
segments) against a corrected copy of the same transcript — the two share one
segmentation, so scoring is an exact per-segment comparison. It is deliberately
*not* DER: DER's forgiveness collar (0.25 s) blurs exactly the fast speaker
changes this metric is built to measure.

Two complementary numbers:

* **Speaker accuracy** — the fraction of segments (and of duration) whose speaker
  is correct, after an optimal relabelling of the hypothesis's arbitrary speaker
  ids onto the truth's (Hungarian assignment). Each figure is maximised under its
  own mapping — segment accuracy over matched segment count, duration accuracy
  over matched seconds — so neither depends on how the other's ties break. Answers
  "did we attribute this line to the right person?"
* **Turn-boundary (speaker-change) F1** — over every adjacent-segment boundary,
  precision/recall/F1 of "the speaker changes here". Relabelling-independent —
  a change is simply "adjacent labels differ". Answers "did we detect the turn?"

An unlabelled hypothesis segment (``hyp_speaker == ""``) is treated as
"no speaker assigned": it is correct only where the truth is also unlabelled,
and it is never counted as a distinct speaker.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.transcript_label import LabeledSegment, read_labeled


@dataclass(frozen=True)
class TranscriptScore:
    """Segment-level scores for one labelled transcript.

    Attributes:
        n_segments: Number of scored segments.
        total_duration: Summed segment duration in seconds.
        speaker_accuracy_seg: Fraction of segments correctly attributed (each
            segment weighted equally).
        speaker_accuracy_dur: Fraction of *duration* correctly attributed (long
            segments weighted more) — the DER-comparable figure.
        correct_segments: Count of correctly-attributed segments (raw, for
            exact multi-clip pooling).
        correct_duration: Correctly-attributed duration in seconds (raw).
        change_precision: Of the speaker changes the pipeline predicted, the
            fraction that are real.
        change_recall: Of the real speaker changes, the fraction the pipeline found.
        change_f1: Harmonic mean of ``change_precision`` and ``change_recall``.
        ref_speakers: Distinct non-empty truth speaker labels.
        hyp_speakers: Distinct non-empty hypothesis speaker labels.
        speaker_count_error: ``abs(hyp_speakers - ref_speakers)``.
        n_ref_changes: Real speaker-change boundaries.
        n_hyp_changes: Predicted speaker-change boundaries.
        n_correct_changes: Boundaries where a change is both real and predicted.
        label_mapping: The duration-optimal hypothesis-label -> truth-label
            relabelling (the canonical "who is this speaker" identity, over
            non-empty labels only). Segment accuracy is maximised under a
            separate count-optimal mapping, so this may differ from the exact
            assignment behind ``speaker_accuracy_seg``.
    """

    n_segments: int
    total_duration: float
    speaker_accuracy_seg: float
    speaker_accuracy_dur: float
    correct_segments: int
    correct_duration: float
    change_precision: float
    change_recall: float
    change_f1: float
    ref_speakers: int
    hyp_speakers: int
    speaker_count_error: int
    n_ref_changes: int
    n_hyp_changes: int
    n_correct_changes: int
    label_mapping: dict[str, str]


def _optimal_mapping(segments: list[LabeledSegment], weights: list[float]) -> dict[str, str]:
    """Find the hyp->true label relabelling that maximises matched ``weights``.

    Speaker ids are arbitrary, so the hypothesis's labels are mapped onto the
    truth's before comparison. A contingency table over the segments where both
    sides are labelled — each cell the summed ``weights`` of its segments — is
    solved as a linear assignment (Hungarian). Pass per-segment durations to
    maximise matched duration, or all-ones to maximise matched segment count.
    Empty labels are excluded from the mapping — they are handled separately as
    "unlabelled".

    Args:
        segments: The transcript segments.
        weights: Per-segment non-negative weight, aligned to ``segments``.

    Returns:
        A mapping from non-empty hypothesis labels to non-empty truth labels.
        Empty when either side has no non-empty labels.
    """
    hyp_labels = sorted({s.hyp_speaker for s in segments if s.hyp_speaker})
    true_labels = sorted({s.true_speaker for s in segments if s.true_speaker})
    if not hyp_labels or not true_labels:
        return {}

    hyp_index = {label: i for i, label in enumerate(hyp_labels)}
    true_index = {label: j for j, label in enumerate(true_labels)}
    weight = np.zeros((len(hyp_labels), len(true_labels)), dtype=float)
    for seg, unit in zip(segments, weights, strict=True):
        if seg.hyp_speaker and seg.true_speaker:
            weight[hyp_index[seg.hyp_speaker], true_index[seg.true_speaker]] += unit

    # linear_sum_assignment minimises cost, so negate to maximise matched weight.
    row_ind, col_ind = linear_sum_assignment(-weight)
    return {hyp_labels[r]: true_labels[c] for r, c in zip(row_ind.tolist(), col_ind.tolist(), strict=True)}


def _is_correct(seg: LabeledSegment, mapping: dict[str, str]) -> bool:
    """Whether a segment's hypothesis speaker matches the truth under ``mapping``.

    An unlabelled hypothesis (``hyp_speaker == ""``) is correct only where the
    truth is also unlabelled; a hypothesis label absent from ``mapping`` (an
    unmapped surplus speaker) never matches.

    Args:
        seg: The segment to test.
        mapping: A hyp->true label relabelling from :func:`_optimal_mapping`.

    Returns:
        ``True`` when the (relabelled) hypothesis speaker equals the truth.
    """
    if not seg.hyp_speaker:
        return not seg.true_speaker
    return mapping.get(seg.hyp_speaker) == seg.true_speaker


def _change_boundaries(labels: list[str]) -> set[int]:
    """Return the indices ``i`` where ``labels[i] != labels[i + 1]`` (a turn)."""
    return {i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]}


def score_transcript(segments: list[LabeledSegment]) -> TranscriptScore:
    """Score one labelled transcript for speaker accuracy and turn-boundary F1.

    Args:
        segments: The labelled segments (each carrying ``hyp_speaker`` and the
            corrected ``true_speaker``), in transcript order.

    Returns:
        The computed :class:`TranscriptScore`.

    Raises:
        ValueError: If ``segments`` is empty (nothing to score).
    """
    if not segments:
        raise ValueError("cannot score an empty transcript")

    n = len(segments)
    durations = [max(0.0, seg.end - seg.start) for seg in segments]
    total_duration = sum(durations)

    # Each accuracy is maximised under its OWN optimal relabelling: segment
    # accuracy under a count-weighted mapping, duration accuracy under a
    # duration-weighted one. Sharing a single (duration) mapping would make the
    # segment figure depend on how duration ties are broken and can understate it.
    seg_mapping = _optimal_mapping(segments, [1.0] * n)
    dur_mapping = _optimal_mapping(segments, durations)

    correct_segments = sum(1 for seg in segments if _is_correct(seg, seg_mapping))
    correct_duration = sum(dur for seg, dur in zip(segments, durations, strict=True) if _is_correct(seg, dur_mapping))

    ref_changes = _change_boundaries([seg.true_speaker for seg in segments])
    hyp_changes = _change_boundaries([seg.hyp_speaker for seg in segments])
    true_positive = len(ref_changes & hyp_changes)
    false_positive = len(hyp_changes - ref_changes)
    false_negative = len(ref_changes - hyp_changes)

    # No predictions -> vacuously precise; no real changes -> vacuously complete.
    precision = true_positive / (true_positive + false_positive) if hyp_changes else 1.0
    recall = true_positive / (true_positive + false_negative) if ref_changes else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ref_speakers = len({seg.true_speaker for seg in segments if seg.true_speaker})
    hyp_speakers = len({seg.hyp_speaker for seg in segments if seg.hyp_speaker})

    return TranscriptScore(
        n_segments=n,
        total_duration=total_duration,
        speaker_accuracy_seg=correct_segments / n,
        # No duration signal (all zero-length) -> defer to the segment figure
        # rather than report a misleading 0.0.
        speaker_accuracy_dur=(correct_duration / total_duration if total_duration else correct_segments / n),
        correct_segments=correct_segments,
        correct_duration=correct_duration,
        change_precision=precision,
        change_recall=recall,
        change_f1=f1,
        ref_speakers=ref_speakers,
        hyp_speakers=hyp_speakers,
        speaker_count_error=abs(ref_speakers - hyp_speakers),
        n_ref_changes=len(ref_changes),
        n_hyp_changes=len(hyp_changes),
        n_correct_changes=true_positive,
        label_mapping=dur_mapping,
    )


@dataclass(frozen=True)
class TranscriptReport:
    """Per-clip scores plus a micro-averaged OVERALL across a set of clips.

    OVERALL pools raw counts (segments, seconds, change boundaries) rather than
    averaging per-clip rates, so a long clip weighs more than a short one and a
    clip with few turns cannot dominate the boundary F1.

    Attributes:
        files: ``(clip_name, score)`` pairs in input order.
        overall_accuracy_seg: Segments correct / segments total, pooled.
        overall_accuracy_dur: Duration correct / duration total, pooled.
        overall_change_precision: Pooled turn-boundary precision.
        overall_change_recall: Pooled turn-boundary recall.
        overall_change_f1: Pooled turn-boundary F1.
    """

    files: list[tuple[str, TranscriptScore]]
    overall_accuracy_seg: float
    overall_accuracy_dur: float
    overall_change_precision: float
    overall_change_recall: float
    overall_change_f1: float

    @classmethod
    def from_scores(cls, named_scores: list[tuple[str, TranscriptScore]]) -> TranscriptReport:
        """Pool per-clip scores into a report with a micro-averaged OVERALL.

        Args:
            named_scores: ``(clip_name, score)`` pairs.

        Returns:
            The assembled :class:`TranscriptReport`.

        Raises:
            ValueError: If ``named_scores`` is empty (nothing to pool).
        """
        if not named_scores:
            raise ValueError("cannot build a report from zero clips")
        total_seg = sum(s.n_segments for _, s in named_scores)
        correct_seg = sum(s.correct_segments for _, s in named_scores)
        total_dur = sum(s.total_duration for _, s in named_scores)
        correct_dur = sum(s.correct_duration for _, s in named_scores)
        true_positive = sum(s.n_correct_changes for _, s in named_scores)
        false_positive = sum(s.n_hyp_changes - s.n_correct_changes for _, s in named_scores)
        false_negative = sum(s.n_ref_changes - s.n_correct_changes for _, s in named_scores)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return cls(
            files=list(named_scores),
            overall_accuracy_seg=correct_seg / total_seg if total_seg else 0.0,
            overall_accuracy_dur=correct_dur / total_dur if total_dur else 0.0,
            overall_change_precision=precision,
            overall_change_recall=recall,
            overall_change_f1=f1,
        )

    def to_markdown(self) -> str:
        """Render the report as a ranked Markdown table with an OVERALL row.

        Returns:
            A Markdown table string (one row per clip, then OVERALL).
        """
        header = (
            "| clip | seg_acc | dur_acc | turn_P | turn_R | turn_F1 | ref# | hyp# | segs |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        body = "".join(
            f"| {uri} | {s.speaker_accuracy_seg:.3f} | {s.speaker_accuracy_dur:.3f} | "
            f"{s.change_precision:.3f} | {s.change_recall:.3f} | {s.change_f1:.3f} | "
            f"{s.ref_speakers} | {s.hyp_speakers} | {s.n_segments} |\n"
            for uri, s in self.files
        )
        overall = (
            f"| **OVERALL** | **{self.overall_accuracy_seg:.3f}** | **{self.overall_accuracy_dur:.3f}** | "
            f"{self.overall_change_precision:.3f} | {self.overall_change_recall:.3f} | "
            f"**{self.overall_change_f1:.3f}** | | | |\n"
        )
        return header + body + overall

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Return one flat dict per clip plus an OVERALL row for CSV export.

        Returns:
            A list of dictionaries suitable for ``csv.DictWriter``.
        """
        rows: list[dict[str, Any]] = [
            {
                "clip": uri,
                "seg_acc": round(s.speaker_accuracy_seg, 4),
                "dur_acc": round(s.speaker_accuracy_dur, 4),
                "turn_precision": round(s.change_precision, 4),
                "turn_recall": round(s.change_recall, 4),
                "turn_f1": round(s.change_f1, 4),
                "ref_speakers": s.ref_speakers,
                "hyp_speakers": s.hyp_speakers,
                "n_segments": s.n_segments,
            }
            for uri, s in self.files
        ]
        rows.append(
            {
                "clip": "OVERALL",
                "seg_acc": round(self.overall_accuracy_seg, 4),
                "dur_acc": round(self.overall_accuracy_dur, 4),
                "turn_precision": round(self.overall_change_precision, 4),
                "turn_recall": round(self.overall_change_recall, 4),
                "turn_f1": round(self.overall_change_f1, 4),
                "ref_speakers": "",
                "hyp_speakers": "",
                "n_segments": "",
            }
        )
        return rows


def score_files(paths: list[Path]) -> TranscriptReport:
    """Read labelled template TSVs and score them into one report.

    Args:
        paths: Labelled template TSVs (each the output of
            :mod:`eval.make_transcript_template` after correction).

    Returns:
        A :class:`TranscriptReport` with a per-clip row for each path (named by
        the file stem) and a pooled OVERALL.
    """
    named: list[tuple[str, TranscriptScore]] = []
    for path in paths:
        named.append((path.stem, score_transcript(read_labeled(path))))
    return TranscriptReport.from_scores(named)


def _main() -> None:
    """CLI: score one or more labelled template TSVs and print the table."""
    parser = argparse.ArgumentParser(
        description="Score corrected diarization transcripts: segment speaker "
        "accuracy + turn-boundary F1 (no forgiveness collar).",
    )
    parser.add_argument("labeled", nargs="+", type=Path, help="labelled template TSV(s)")
    parser.add_argument("--report", type=Path, default=None, help="also write the Markdown table here")
    parser.add_argument("--csv", type=Path, default=None, help="also write per-clip rows as CSV here")
    args = parser.parse_args()

    report = score_files(list(args.labeled))
    markdown = report.to_markdown()
    print(markdown)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown, encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        rows = report.to_csv_rows()
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    _main()
