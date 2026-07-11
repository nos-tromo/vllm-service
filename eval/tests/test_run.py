"""Tests for run_diarization with an injected fake pipeline (torch-free)."""

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pyannote.core import Annotation, Segment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.configs import DiarizeConfig
from eval.run import run_diarization


class _FakePipeline:
    """Records call kwargs and returns a fixed two-speaker annotation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, payload: dict[str, object], **kwargs: object) -> Annotation:
        self.calls.append(kwargs)
        ann = Annotation(uri=cast(str | None, payload.get("uri")))
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"
        ann[Segment(1.0, 2.0)] = "SPEAKER_01"
        return ann


def test_writes_one_rttm_per_file_and_passes_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each file yields an RTTM; config speaker-bounds reach the pipeline call."""
    monkeypatch.setattr("eval.run.decode_audio", lambda data: np.zeros(16000, dtype=np.float32))  # bypass ffmpeg
    monkeypatch.setattr(
        "eval.run._to_waveform", lambda audio: audio
    )  # skip the lazy torch import (fake ignores waveform)
    (tmp_path / "a.wav").write_bytes(b"x")
    fake = _FakePipeline()
    out = tmp_path / "hyp"
    written = run_diarization(
        fake, [("a", str(tmp_path / "a.wav"))], str(out), DiarizeConfig(label="floor2", min_speakers=2)
    )
    assert written == [str(out / "a.rttm")]
    assert (out / "a.rttm").read_text().count("SPEAKER_") == 2
    assert fake.calls[0] == {"min_speakers": 2}
