"""Hand-computed cases for segment-level speaker accuracy + turn-boundary F1.

Every segment carries both the pipeline's guess (``hyp_speaker``) and the
corrected truth (``true_speaker``); the two share one segmentation, so scoring
is an exact per-segment comparison — no time alignment. Speaker labels are
arbitrary ids, so accuracy is computed after an optimal hyp->true relabelling;
change-F1 is relabelling-independent (a change is just "adjacent labels differ").
"""

import math

import pytest

from eval.transcript_label import LabeledSegment
from eval.transcript_metric import TranscriptReport, score_transcript


def _seg(start: float, end: float, hyp: str, true: str, text: str = "x") -> LabeledSegment:
    """Build a LabeledSegment from (start, end, hyp_speaker, true_speaker)."""
    return LabeledSegment(start=start, end=end, hyp_speaker=hyp, true_speaker=true, text=text)


def test_perfect_match_survives_relabelling() -> None:
    """Hyp uses different speaker ids but a consistent 1-1 mapping to truth.

    hyp {1,1,2} vs true {2,2,1}: the optimal remap 1->2, 2->1 makes every
    segment correct, and the single turn sits at the same boundary in both.
    """
    segs = [_seg(0, 1, "1", "2"), _seg(1, 2, "1", "2"), _seg(2, 3, "2", "1")]
    s = score_transcript(segs)
    assert s.speaker_accuracy_seg == 1.0
    assert s.speaker_accuracy_dur == 1.0
    assert s.change_precision == 1.0
    assert s.change_recall == 1.0
    assert s.change_f1 == 1.0
    assert s.ref_speakers == 2
    assert s.hyp_speakers == 2
    assert s.speaker_count_error == 0
    assert s.n_segments == 3


def test_merged_speaker_misses_every_turn() -> None:
    """Hyp collapses A,B,A into one speaker: 2/3 correct, both turns missed."""
    segs = [_seg(0, 1, "1", "A"), _seg(1, 2, "1", "B"), _seg(2, 3, "1", "A")]
    s = score_transcript(segs)
    assert math.isclose(s.speaker_accuracy_seg, 2 / 3)
    assert s.n_ref_changes == 2
    assert s.n_hyp_changes == 0
    assert s.change_recall == 0.0
    assert s.change_f1 == 0.0
    assert s.speaker_count_error == 1  # hyp 1 speaker vs ref 2


def test_hallucinated_turn_is_a_false_positive() -> None:
    """Hyp splits one true speaker across a boundary: a false change, no real one."""
    segs = [_seg(0, 1, "1", "A"), _seg(1, 2, "2", "A")]
    s = score_transcript(segs)
    assert s.n_ref_changes == 0
    assert s.n_hyp_changes == 1
    assert s.change_precision == 0.0
    assert s.change_recall == 1.0  # no real turns to miss -> vacuously perfect
    assert s.change_f1 == 0.0
    assert s.speaker_accuracy_seg == 0.5


def test_duration_weighting_differs_from_segment_count() -> None:
    """A long correct segment + a short wrong one: dur-accuracy >> seg-accuracy."""
    segs = [_seg(0, 10, "1", "A"), _seg(10, 11, "1", "B")]
    s = score_transcript(segs)
    assert s.speaker_accuracy_seg == 0.5  # 1 of 2 segments
    assert math.isclose(s.speaker_accuracy_dur, 10 / 11)  # 10 of 11 seconds


def test_single_speaker_no_turns_is_perfect() -> None:
    """One speaker throughout, correctly labelled: no turns, none invented."""
    segs = [_seg(0, 1, "1", "A"), _seg(1, 2, "1", "A")]
    s = score_transcript(segs)
    assert s.speaker_accuracy_seg == 1.0
    assert s.n_ref_changes == 0
    assert s.n_hyp_changes == 0
    assert s.change_precision == 1.0  # nothing predicted, nothing wrong
    assert s.change_recall == 1.0
    assert s.change_f1 == 1.0


def test_empty_hyp_label_is_unlabelled_not_a_speaker() -> None:
    """A segment the pipeline left unlabelled ("") is wrong, and not counted as a speaker."""
    segs = [_seg(0, 1, "A", "A"), _seg(1, 2, "", "A")]
    s = score_transcript(segs)
    assert s.speaker_accuracy_seg == 0.5  # the unlabelled segment is wrong
    assert s.hyp_speakers == 1  # "" does not count as a distinct speaker
    assert s.ref_speakers == 1


def test_label_mapping_is_reported() -> None:
    """The optimal hyp->true relabelling is exposed for inspection."""
    segs = [_seg(0, 1, "1", "Alice"), _seg(1, 2, "2", "Bob")]
    s = score_transcript(segs)
    assert s.label_mapping["1"] == "Alice"
    assert s.label_mapping["2"] == "Bob"


def test_empty_transcript_is_a_caller_error() -> None:
    """Scoring nothing is a mistake, not a silent zero."""
    with pytest.raises(ValueError):
        score_transcript([])


def test_seg_accuracy_is_invariant_to_speaker_label_spelling() -> None:
    """seg_acc uses a count-optimal mapping, so renaming a true speaker can't change it.

    hyp "1" merges one 3 s true-A segment and three 1 s true-B segments. Duration
    ties (A=3 s, B=3 s) but counts do not (A=1, B=3). A duration-optimal mapping
    would break the tie on label spelling and flip seg_acc between 0.25 and 0.75;
    a count-optimal mapping always identifies "1" as the 3-segment speaker.
    """
    base = [_seg(0, 3, "1", "A"), _seg(3, 4, "1", "B"), _seg(4, 5, "1", "B"), _seg(5, 6, "1", "B")]
    renamed = [_seg(0, 3, "1", "Z"), _seg(3, 4, "1", "B"), _seg(4, 5, "1", "B"), _seg(5, 6, "1", "B")]
    assert score_transcript(base).speaker_accuracy_seg == 0.75
    assert score_transcript(renamed).speaker_accuracy_seg == 0.75
    # duration accuracy stays at the duration optimum (3 s of 6 s) either way.
    assert score_transcript(base).speaker_accuracy_dur == 0.5


def test_zero_duration_transcript_falls_back_to_segment_accuracy() -> None:
    """All-zero-length segments carry no duration signal; dur_acc defers to seg_acc."""
    segs = [_seg(1.0, 1.0, "1", "A"), _seg(2.0, 2.0, "1", "A")]
    s = score_transcript(segs)
    assert s.total_duration == 0.0
    assert s.speaker_accuracy_seg == 1.0
    assert s.speaker_accuracy_dur == 1.0  # not a misleading 0.0


def test_report_over_no_clips_is_a_caller_error() -> None:
    """Pooling zero clips is a mistake, matching score_transcript's empty guard."""
    with pytest.raises(ValueError):
        TranscriptReport.from_scores([])


def test_unmapped_hypothesis_label_scores_wrong() -> None:
    """With more hyp speakers than true, the surplus hyp label stays unmapped (wrong)."""
    segs = [_seg(0, 1, "1", "A"), _seg(1, 2, "2", "A"), _seg(2, 3, "3", "A")]
    s = score_transcript(segs)
    assert s.hyp_speakers == 3
    assert s.ref_speakers == 1
    assert math.isclose(s.speaker_accuracy_seg, 1 / 3)  # only one hyp label maps to A


def test_score_exposes_raw_counts_for_pooling() -> None:
    """Accuracy is also carried as raw counts so multi-clip pooling stays exact."""
    segs = [_seg(0, 1, "1", "A"), _seg(1, 2, "1", "B"), _seg(2, 3, "1", "A")]
    s = score_transcript(segs)
    assert s.correct_segments == 2
    assert s.correct_duration == 2.0


def test_report_pools_across_clips_by_count_not_mean() -> None:
    """OVERALL is micro-averaged (pool segments/boundaries), not a mean of rates."""
    clip_a = score_transcript([_seg(0, 1, "1", "A"), _seg(1, 2, "1", "B"), _seg(2, 3, "1", "A")])
    clip_b = score_transcript([_seg(0, 1, "1", "A"), _seg(1, 2, "2", "B")])
    report = TranscriptReport.from_scores([("clip_a", clip_a), ("clip_b", clip_b)])
    # segment accuracy: (2 + 2) correct of (3 + 2) segments = 0.8
    assert math.isclose(report.overall_accuracy_seg, 0.8)
    assert math.isclose(report.overall_accuracy_dur, 0.8)
    # changes pooled: TP=1, FP=0, FN=2 -> P=1.0, R=1/3, F1=0.5
    assert math.isclose(report.overall_change_precision, 1.0)
    assert math.isclose(report.overall_change_recall, 1 / 3)
    assert math.isclose(report.overall_change_f1, 0.5)


def test_report_markdown_has_per_clip_rows_and_overall() -> None:
    """The rendered table names each clip and carries an OVERALL summary line."""
    clip_a = score_transcript([_seg(0, 1, "1", "A"), _seg(1, 2, "2", "B")])
    report = TranscriptReport.from_scores([("clip_a", clip_a)])
    md = report.to_markdown()
    assert "clip_a" in md
    assert "OVERALL" in md
    assert "acc" in md.lower()
