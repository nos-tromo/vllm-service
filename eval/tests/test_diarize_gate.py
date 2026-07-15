"""Tests for the pure VAD-gating interval logic (torch-free)."""

import sys
from pathlib import Path

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diarize_gate import crop_turns_to_speech


def test_turn_inside_speech_is_unchanged() -> None:
    """A turn fully covered by one speech interval passes through as-is."""
    turns = [(1.0, 2.0, "SPEAKER_00")]
    speech = [(0.5, 3.0)]
    assert crop_turns_to_speech(turns, speech) == [(1.0, 2.0, "SPEAKER_00")]


def test_turn_outside_speech_is_dropped() -> None:
    """A turn with no speech overlap (music/noise false alarm) is removed."""
    turns = [(5.0, 7.0, "SPEAKER_00")]
    speech = [(0.0, 4.0)]
    assert crop_turns_to_speech(turns, speech) == []


def test_straddling_turn_is_trimmed() -> None:
    """A turn overhanging the speech edge is cropped to the overlap."""
    turns = [(1.0, 5.0, "SPEAKER_00")]
    speech = [(2.0, 8.0)]
    assert crop_turns_to_speech(turns, speech) == [(2.0, 5.0, "SPEAKER_00")]


def test_turn_spanning_a_gap_splits_in_two() -> None:
    """A turn bridging two speech intervals yields one sub-turn per interval."""
    turns = [(0.0, 10.0, "SPEAKER_00")]
    speech = [(1.0, 3.0), (6.0, 8.0)]
    assert crop_turns_to_speech(turns, speech) == [
        (1.0, 3.0, "SPEAKER_00"),
        (6.0, 8.0, "SPEAKER_00"),
    ]


def test_empty_speech_drops_everything() -> None:
    """No detected speech (pure music upload) → no turns survive."""
    assert crop_turns_to_speech([(0.0, 5.0, "SPEAKER_00")], []) == []


def test_empty_turns_is_noop() -> None:
    """No turns in → no turns out, regardless of speech."""
    assert crop_turns_to_speech([], [(0.0, 5.0)]) == []


def test_zero_length_overlap_is_dropped() -> None:
    """A turn merely touching a speech boundary produces no zero-length turn."""
    turns = [(3.0, 5.0, "SPEAKER_00")]
    speech = [(0.0, 3.0)]
    assert crop_turns_to_speech(turns, speech) == []


def test_unsorted_input_yields_chronological_output() -> None:
    """Output is chronological even when turns and speech arrive unsorted."""
    turns = [(6.0, 7.0, "SPEAKER_01"), (1.0, 2.0, "SPEAKER_00")]
    speech = [(5.0, 8.0), (0.0, 3.0)]
    assert crop_turns_to_speech(turns, speech) == [
        (1.0, 2.0, "SPEAKER_00"),
        (6.0, 7.0, "SPEAKER_01"),
    ]
