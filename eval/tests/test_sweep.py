"""Tests for the sweep comparison-table rendering."""

from eval.configs import DiarizeConfig
from eval.score import RunReport
from eval.sweep import summarize_sweep


def _report(der: float) -> RunReport:
    """A minimal RunReport carrying just an overall DER.

    Args:
        der: The overall diarization error rate.

    Returns:
        A RunReport with the given DER and zero other metrics.
    """
    return RunReport(
        files=[],
        overall_der=der,
        overall_confusion=0.0,
        overall_false_alarm=0.0,
        overall_missed_detection=0.0,
        speaker_count_mae=0.0,
        speaker_count_bias=0.0,
    )


def test_table_sorted_ascending_by_der() -> None:
    """The best (lowest-DER) config appears first."""
    results = [
        (DiarizeConfig(label="baseline"), _report(0.40)),
        (DiarizeConfig(label="community1"), _report(0.22)),
    ]
    table = summarize_sweep(results)
    assert table.index("community1") < table.index("baseline")
    assert "0.220" in table and "0.400" in table
