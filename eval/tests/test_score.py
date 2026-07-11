"""Tests for DER scoring (hand-computed cases, collar=0)."""

from pyannote.core import Annotation, Segment, Timeline

from eval.score import score_run


def _ann(spans: list[tuple[float, float, str]]) -> Annotation:
    """Build an Annotation from (start, end, speaker) triples."""
    ann = Annotation()
    for start, end, spk in spans:
        ann[Segment(start, end)] = spk
    return ann


def test_missed_detection_half() -> None:
    """Ref speaks 0-10, hyp only 0-5 → 5s miss over 10s total → DER 0.5."""
    ref = _ann([(0.0, 10.0, "A")])
    hyp = _ann([(0.0, 5.0, "A")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    assert report.files[0].missed_detection == 5.0
    assert report.overall_der == 0.5


def test_confusion_is_permutation_invariant() -> None:
    """Two ref speakers, one hyp speaker: best mapping leaves 5s confusion → DER 0.5."""
    ref = _ann([(0.0, 5.0, "A"), (5.0, 10.0, "B")])
    hyp = _ann([(0.0, 10.0, "X")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    assert report.overall_der == 0.5
    assert report.files[0].confusion == 5.0


def test_speaker_count_error_and_bias() -> None:
    """Ref has 2 speakers, hyp has 1 → count error 1, bias -1 (under-count)."""
    ref = _ann([(0.0, 5.0, "A"), (5.0, 10.0, "B")])
    hyp = _ann([(0.0, 10.0, "X")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    assert report.files[0].ref_speakers == 2
    assert report.files[0].hyp_speakers == 1
    assert report.files[0].speaker_count_error == 1
    assert report.speaker_count_mae == 1.0
    assert report.speaker_count_bias == -1.0


def test_der_is_one_when_reference_is_empty_but_hypothesis_speaks() -> None:
    """Empty reference + a hallucinated hypothesis segment → der 1.0, not 0.0.

    With zero reference speech, ``confusion + false_alarm + missed`` over a
    ``total`` of 0 must report the worst case (1.0), matching pyannote's own
    per-file "diarization error rate" convention — not the old hand-rolled
    ``_rate`` division, which returned 0.0 (perfect) for a 0 denominator.
    """
    ref = Annotation()
    hyp = _ann([(0.0, 5.0, "A")])
    uem = Timeline(segments=[Segment(0.0, 5.0)])
    report = score_run([("f1", ref, hyp, uem)], collar=0.0)
    assert report.files[0].der == 1.0


def test_markdown_has_overall_row() -> None:
    """The rendered report includes an OVERALL summary line."""
    ref = _ann([(0.0, 10.0, "A")])
    hyp = _ann([(0.0, 10.0, "A")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    md = report.to_markdown()
    assert "OVERALL" in md
    assert "f1" in md
