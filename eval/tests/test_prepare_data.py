"""Tests for the pyannote.database registry generation."""

import yaml

from eval.prepare_data import ProtocolPaths, build_database_yml


def test_database_yml_registers_each_protocol() -> None:
    """Each ProtocolPaths becomes a resolvable Databases + Protocols entry."""
    protocols = [
        ProtocolPaths(
            name="VoxConverse",
            audio_dir="/data/voxconverse/audio",
            rttm="/data/voxconverse/dev.rttm",
            uem="/data/voxconverse/dev.uem",
        ),
        ProtocolPaths(name="AMI", audio_dir="/data/ami/audio", rttm="/data/ami/test.rttm", uem="/data/ami/test.uem"),
    ]
    text = build_database_yml(protocols)
    parsed = yaml.safe_load(text)
    assert set(parsed["Protocols"]) == {"VoxConverse", "AMI"}
    vox = parsed["Protocols"]["VoxConverse"]["SpeakerDiarization"]["Benchmark"]
    assert vox["test"]["annotation"] == "/data/voxconverse/dev.rttm"
    assert vox["test"]["annotated"] == "/data/voxconverse/dev.uem"
    assert "/data/voxconverse/audio" in parsed["Databases"]["VoxConverse"][0]
