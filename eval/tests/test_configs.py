"""Tests for the DiarizeConfig value object."""

from eval.configs import DiarizeConfig


def test_defaults_are_all_none_and_label_required() -> None:
    """A bare config carries only its label; every knob defaults to None."""
    cfg = DiarizeConfig(label="baseline")
    assert cfg.label == "baseline"
    assert cfg.model_id is None
    assert cfg.clustering_threshold is None
    assert cfg.pipeline_kwargs == {}


def test_pipeline_kwargs_include_only_set_speaker_bounds() -> None:
    """pipeline_kwargs surfaces only the speaker bounds that were set."""
    cfg = DiarizeConfig(label="floor2", min_speakers=2)
    assert cfg.pipeline_kwargs == {"min_speakers": 2}


def test_as_dict_round_trips_all_fields() -> None:
    """as_dict is flat and includes every field for report traceability."""
    cfg = DiarizeConfig(label="c1", model_id="pyannote/speaker-diarization-community-1", clustering_threshold=0.6)
    d = cfg.as_dict()
    assert d["label"] == "c1"
    assert d["model_id"] == "pyannote/speaker-diarization-community-1"
    assert d["clustering_threshold"] == 0.6
    assert d["num_speakers"] is None
