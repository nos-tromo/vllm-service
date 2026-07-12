"""Round-trip + parsing tests for the labelling-template TSV I/O."""

from pathlib import Path

import pytest

from eval.transcript_label import LabeledSegment, read_labeled, write_template


def _seg(start: float, end: float, hyp: str, true: str, text: str = "hello") -> LabeledSegment:
    return LabeledSegment(start=start, end=end, hyp_speaker=hyp, true_speaker=true, text=text)


def test_round_trip_preserves_corrected_truth_distinct_from_guess(tmp_path: Path) -> None:
    """A corrected true_speaker that differs from hyp_speaker survives the round trip."""
    path = tmp_path / "clip.tsv"
    segments = [
        _seg(0.0, 3.4, "Speaker 1", "Speaker 1", "if there's one secret"),
        _seg(3.4, 5.3, "Speaker 1", "Speaker 2", "why would he stop there?"),
    ]
    write_template(segments, path)
    restored = read_labeled(path)
    assert restored == segments
    # The whole point: hyp and true are stored independently.
    assert restored[1].hyp_speaker == "Speaker 1"
    assert restored[1].true_speaker == "Speaker 2"


def test_template_writes_banner_and_header(tmp_path: Path) -> None:
    """The written file carries the instruction banner and a column header."""
    path = tmp_path / "clip.tsv"
    write_template([_seg(0.0, 1.0, "A", "A")], path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#")
    assert "true_speaker` column ONLY" in "\n".join(lines)
    header = next(line for line in lines if not line.startswith("#"))
    assert header.split("\t") == ["start", "end", "hyp_speaker", "true_speaker", "text"]


def test_reads_hand_written_file_with_comments_and_blank_lines(tmp_path: Path) -> None:
    """Comment lines, the header, and blank lines are ignored when reading."""
    path = tmp_path / "clip.tsv"
    path.write_text(
        "# a comment\n"
        "start\tend\thyp_speaker\ttrue_speaker\ttext\n"
        "\n"
        "0.0\t1.0\tSpeaker 1\tSpeaker 2\thello there\n"
        "# trailing note\n",
        encoding="utf-8",
    )
    segments = read_labeled(path)
    assert segments == [_seg(0.0, 1.0, "Speaker 1", "Speaker 2", "hello there")]


def test_text_with_tabs_and_newlines_is_flattened(tmp_path: Path) -> None:
    """Delimiter/newline characters in text can't corrupt the one-row-per-segment layout."""
    path = tmp_path / "clip.tsv"
    write_template([_seg(0.0, 1.0, "A", "A", "line one\tcol\nline two")], path)
    restored = read_labeled(path)
    assert len(restored) == 1
    assert "\t" not in restored[0].text
    assert "\n" not in restored[0].text


def test_wrong_column_count_raises(tmp_path: Path) -> None:
    """A truncated data row is a hard error, not a silent skip."""
    path = tmp_path / "bad.tsv"
    path.write_text("0.0\t1.0\tSpeaker 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_labeled(path)


def test_non_numeric_time_raises(tmp_path: Path) -> None:
    """A non-numeric start/end is a hard error."""
    path = tmp_path / "bad.tsv"
    path.write_text("start_oops\t1.0\tSpeaker 1\tSpeaker 1\thi\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_labeled(path)
