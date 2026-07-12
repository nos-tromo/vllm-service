"""The labelling template's data model + on-disk TSV I/O.

Unlike the RTTM-based DER harness, this metric scores the *end-to-end Nextext
transcript* — the timestamped, speaker-labelled segments an operator actually
reads. Ground truth is made by correcting those labels in place, so a labelled
file is self-contained: each row carries both the pipeline's guess
(``hyp_speaker``) and the corrected truth (``true_speaker``) for one span of
transcript. Sharing a single segmentation is what makes segment-level scoring
exact — there is no hypothesis-vs-reference time alignment to get wrong.

The on-disk format is a tab-separated file (transcript text routinely contains
commas, never tabs) with a header row::

    start	end	hyp_speaker	true_speaker	text
    0.000	3.400	Speaker 1	Speaker 1	If there's one secret ...
    3.400	5.300	Speaker 1	Speaker 2	Why would he stop there?

Lines beginning ``#`` are comments (the template writes an instruction banner).
An operator corrects the ``true_speaker`` column only — it is pre-filled with
``hyp_speaker`` so only the rows the pipeline got wrong need editing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_COLUMNS = ("start", "end", "hyp_speaker", "true_speaker", "text")

_TEMPLATE_BANNER = (
    "# Diarization labelling template. Correct the `true_speaker` column ONLY.\n"
    "# It is pre-filled with the pipeline's guess (`hyp_speaker`); change a row's\n"
    "# `true_speaker` only where the pipeline mislabelled the speaker. Leave\n"
    "# `hyp_speaker`, times, and text untouched. Tab-separated; one segment per row.\n"
)


@dataclass(frozen=True)
class LabeledSegment:
    """One transcript segment with both the guessed and the corrected speaker.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        hyp_speaker: The speaker label the pipeline assigned. May be ``""`` when
            the pipeline left the segment unlabelled (no overlapping turn).
        true_speaker: The corrected ground-truth speaker label.
        text: The transcript text for the segment. Carried for human context so
            the labeller can hear/read who is speaking; not used by the metric.
    """

    start: float
    end: float
    hyp_speaker: str
    true_speaker: str
    text: str


def _clean(value: str) -> str:
    """Strip characters that would corrupt a one-row-per-segment TSV.

    Tabs become spaces (they are the column delimiter) and any embedded newline
    becomes a space (a segment must stay on one line).

    Args:
        value: The raw field value.

    Returns:
        The value with tabs and newlines flattened to single spaces.
    """
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_template(segments: list[LabeledSegment], path: Path) -> None:
    """Write segments as a labelling template TSV (truth pre-filled to the guess).

    Each row's ``true_speaker`` is written as-is from the segment; template
    generation (:mod:`eval.make_transcript_template`) sets it equal to
    ``hyp_speaker`` so an operator only edits the wrong rows.

    Args:
        segments: The transcript segments to write.
        path: Destination file path. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_TEMPLATE_BANNER)
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for seg in segments:
            writer.writerow(
                [
                    f"{seg.start:.3f}",
                    f"{seg.end:.3f}",
                    _clean(seg.hyp_speaker),
                    _clean(seg.true_speaker),
                    _clean(seg.text),
                ]
            )


def read_labeled(path: Path) -> list[LabeledSegment]:
    """Read a labelled template TSV back into segments.

    Comment lines (starting ``#``) and blank lines are skipped, and the single
    header row is detected and dropped. Each remaining row must have the five
    columns ``start, end, hyp_speaker, true_speaker, text``.

    Args:
        path: Path to a labelled template TSV.

    Returns:
        The parsed segments in file order.

    Raises:
        ValueError: If a data row does not have exactly five columns, or its
            ``start``/``end`` fields are not numbers.
    """
    segments: list[LabeledSegment] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(
            (line for line in handle if line.strip() and not line.lstrip().startswith("#")),
            delimiter="\t",
        )
        for row in rows:
            if list(row) == list(_COLUMNS):  # header row
                continue
            if len(row) != len(_COLUMNS):
                raise ValueError(f"expected {len(_COLUMNS)} columns, got {len(row)}: {row!r}")
            start_s, end_s, hyp, true, text = row
            try:
                start = float(start_s)
                end = float(end_s)
            except ValueError as exc:
                raise ValueError(f"non-numeric start/end in row {row!r}") from exc
            segments.append(LabeledSegment(start=start, end=end, hyp_speaker=hyp, true_speaker=true, text=text))
    return segments
