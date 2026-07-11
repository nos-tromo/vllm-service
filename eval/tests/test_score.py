"""Tests for DER scoring (hand-computed cases, collar=0)."""

from pyannote.core import Annotation, Segment

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


def test_markdown_has_overall_row() -> None:
    """The rendered report includes an OVERALL summary line."""
    ref = _ann([(0.0, 10.0, "A")])
    hyp = _ann([(0.0, 10.0, "A")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    md = report.to_markdown()
    assert "OVERALL" in md
    assert "f1" in md
