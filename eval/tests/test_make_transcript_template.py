"""Tests for turning a Nextext transcript.csv into a labelling template."""

from pathlib import Path

import pytest

from eval.make_transcript_template import parse_timestamp, transcript_csv_to_segments


def test_parse_timestamp_handles_nextext_timedelta_strings() -> None:
    """Nextext writes start/end as str(timedelta), rounded to whole seconds."""
    assert parse_timestamp("0:00:33") == 33.0
    assert parse_timestamp("0:01:05") == 65.0
    assert parse_timestamp("1:02:03") == 3723.0


def test_parse_timestamp_also_accepts_float_seconds() -> None:
    """A CSV that already stores float seconds is accepted too."""
    assert parse_timestamp("12.5") == 12.5
    assert parse_timestamp("0") == 0.0


def test_generates_prefilled_segments_from_diarized_csv(tmp_path: Path) -> None:
    """A diarized transcript.csv becomes segments with hyp == true (pre-filled)."""
    csv_path = tmp_path / "transcript.csv"
    csv_path.write_text(
        "start,end,speaker,text\n"
        "0:00:00,0:00:03,Speaker 1,if there's one secret\n"
        "0:00:03,0:00:05,Speaker 2,why would he stop there?\n",
        encoding="utf-8",
    )
    segments = transcript_csv_to_segments(csv_path)
    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 3.0
    assert segments[0].hyp_speaker == "Speaker 1"
    assert segments[0].true_speaker == "Speaker 1"  # pre-filled to the guess
    assert segments[1].hyp_speaker == "Speaker 2"
    assert segments[1].text == "why would he stop there?"


def test_undiarized_csv_has_no_speaker_column_so_hyp_is_blank(tmp_path: Path) -> None:
    """Nextext omits the speaker column for <2 speakers; hyp becomes unlabelled."""
    csv_path = tmp_path / "transcript.csv"
    csv_path.write_text(
        "start,end,text\n0:00:00,0:00:03,hello world\n",
        encoding="utf-8",
    )
    segments = transcript_csv_to_segments(csv_path)
    assert segments[0].hyp_speaker == ""
    assert segments[0].true_speaker == ""


def test_missing_required_column_raises(tmp_path: Path) -> None:
    """A CSV without a text column cannot be a transcript."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("start,end,speaker\n0:00:00,0:00:03,Speaker 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        transcript_csv_to_segments(csv_path)
