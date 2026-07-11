"""Tests for the pure hyperparameter-override resolver (torch-free)."""

import sys
from pathlib import Path

import pytest

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diarize_pipeline import _resolve_param_overrides


def test_no_overrides_returns_none() -> None:
    """No override requested → None, so the caller preserves pretrained defaults."""
    defaults = {"clustering": {"threshold": 0.7}, "segmentation": {"min_duration_off": 0.0}}
    assert _resolve_param_overrides(defaults, clustering_threshold=None, segmentation_min_duration_off=None) is None


def test_threshold_override_is_merged_without_mutating_defaults() -> None:
    """Overriding the threshold changes only that key and leaves defaults intact."""
    defaults = {"clustering": {"threshold": 0.7, "method": "centroid"}, "segmentation": {"min_duration_off": 0.0}}
    merged = _resolve_param_overrides(defaults, clustering_threshold=0.55, segmentation_min_duration_off=None)
    assert merged is not None
    assert merged["clustering"]["threshold"] == 0.55
    assert merged["clustering"]["method"] == "centroid"
    assert merged["segmentation"]["min_duration_off"] == 0.0
    assert defaults["clustering"]["threshold"] == 0.7  # unmutated


def test_missing_knob_raises() -> None:
    """Requesting a knob the loaded pipeline lacks is a loud error, not a no-op."""
    defaults = {"segmentation": {"min_duration_off": 0.0}}
    with pytest.raises(ValueError, match=r"clustering\.threshold"):
        _resolve_param_overrides(defaults, clustering_threshold=0.5, segmentation_min_duration_off=None)
