"""Turn a Nextext ``transcript.csv`` into a labelling template TSV.

Workflow: run a clip through Nextext (max speakers > 1), download its
``transcript.csv`` artifact, and run this to produce a ``*.label.tsv`` whose
``true_speaker`` column is pre-filled with the pipeline's guess. An operator
then corrects only the mislabelled rows; :mod:`eval.transcript_metric` scores
the corrected file.

Nextext's ``transcript.csv`` has columns ``start,end[,speaker],text`` — the
``speaker`` column is present only when the clip was diarized into ≥2 speakers.
Times are ``str(timedelta)`` rounded to whole seconds (e.g. ``0:00:33``); this
reader also accepts plain float seconds so a hand-made CSV works too.

CLI::

    uv run --group eval python -m eval.make_transcript_template \
        path/to/transcript.csv -o path/to/clip.label.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eval.transcript_label import LabeledSegment, write_template

_START = "start"
_END = "end"
_SPEAKER = "speaker"
_TEXT = "text"


def parse_timestamp(value: str) -> float:
    """Parse a transcript timestamp into float seconds.

    Accepts either a plain float-seconds string (``"12.5"``) or a
    ``str(timedelta)`` rendering (``"0:00:33"``, ``"1:02:03"``,
    ``"1 day, 0:00:00"``) as written by Nextext.

    Args:
        value: The timestamp field from a transcript CSV.

    Returns:
        The time in seconds.

    Raises:
        ValueError: If the value is neither a float nor a ``H:MM:SS`` string.
    """
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass

    days = 0.0
    if "day" in text:  # "1 day, 0:00:00" / "2 days, 1:00:00"
        day_part, text = text.split(",", 1)
        days = float(day_part.strip().split()[0])
        text = text.strip()

    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    else:
        raise ValueError(f"unrecognized timestamp: {value!r}")
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def transcript_csv_to_segments(csv_path: Path) -> list[LabeledSegment]:
    """Read a Nextext ``transcript.csv`` into pre-filled labelling segments.

    The ``speaker`` column becomes both ``hyp_speaker`` and ``true_speaker``
    (the truth is pre-filled to the guess so only wrong rows need editing). When
    the CSV has no ``speaker`` column (an undiarized single-speaker transcript),
    both are the empty string.

    Args:
        csv_path: Path to a Nextext ``transcript.csv``.

    Returns:
        Segments in file order, ready for :func:`eval.transcript_label.write_template`.

    Raises:
        ValueError: If the CSV lacks any of the required ``start``/``end``/``text``
            columns.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in (_START, _END, _TEXT) if column not in fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {', '.join(missing)}")
        has_speaker = _SPEAKER in fieldnames

        segments: list[LabeledSegment] = []
        for row in reader:
            speaker = (row.get(_SPEAKER) or "").strip() if has_speaker else ""
            segments.append(
                LabeledSegment(
                    start=parse_timestamp(row[_START]),
                    end=parse_timestamp(row[_END]),
                    hyp_speaker=speaker,
                    true_speaker=speaker,
                    text=(row.get(_TEXT) or "").strip(),
                )
            )
    return segments


def _main() -> None:
    """CLI: transcript.csv -> labelling template TSV."""
    parser = argparse.ArgumentParser(
        description="Convert a Nextext transcript.csv into a diarization labelling template.",
    )
    parser.add_argument("transcript_csv", type=Path, help="Nextext transcript.csv to convert")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output template TSV (default: <transcript_csv stem>.label.tsv)",
    )
    args = parser.parse_args()

    out_path = args.out or args.transcript_csv.with_suffix(".label.tsv")
    segments = transcript_csv_to_segments(args.transcript_csv)
    write_template(segments, out_path)
    print(f"Wrote {len(segments)} segments to {out_path}")
    print("Correct the `true_speaker` column, then score with:")
    print(f"  uv run --group eval python -m eval.transcript_metric {out_path}")


if __name__ == "__main__":
    _main()
