"""Pure interval logic for VAD-gating diarization turns.

The diarize server (``src/diarize_server.py``) optionally crops the pyannote
pipeline's speaker turns to a Silero VAD speech timeline fetched from the
stack's ``vad`` service, dropping the music/noise the diarizer over-detects
as speech (measured −35% false alarm / −12.5% DER at threshold 0.4 /
pad 100 ms — see eval/reports/2026-07-11-false-alarm-vad-gating.md). This
module holds only the interval intersection: no torch, no HTTP, no env, so
the torch-free eval test group can exercise it directly.
"""

from __future__ import annotations


def crop_turns_to_speech(
    turns: list[tuple[float, float, str]],
    speech: list[tuple[float, float]],
) -> list[tuple[float, float, str]]:
    """Intersect speaker turns with a speech timeline.

    Each turn is cropped to the speech intervals it overlaps: a turn fully
    inside speech is unchanged, one overhanging an edge is trimmed, one
    bridging a silence gap splits into one sub-turn per speech interval,
    and one with no overlap is dropped. Zero-length results are dropped.

    Args:
        turns: ``(start, end, speaker)`` turns in seconds; any order.
        speech: ``(start, end)`` speech intervals in seconds; any order.

    Returns:
        Cropped ``(start, end, speaker)`` turns in chronological order.
    """
    cropped: list[tuple[float, float, str]] = []
    for start, end, speaker in turns:
        for speech_start, speech_end in speech:
            overlap_start = max(start, speech_start)
            overlap_end = min(end, speech_end)
            if overlap_end > overlap_start:
                cropped.append((overlap_start, overlap_end, speaker))
    cropped.sort(key=lambda turn: (turn[0], turn[1], turn[2]))
    return cropped
