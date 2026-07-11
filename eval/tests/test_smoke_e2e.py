"""Synthetic end-to-end smoke test for the eval harness glue (torch-free).

Exercises prepare_data -> pyannote.database registry load -> run_diarization
(fake pipeline) -> score_run -> summarize_sweep against two tiny synthetic
recordings, without pyannote.audio/torch, a GPU, network, an ffmpeg binary, or
any downloaded corpus. This is the harness's only test that proves the full
wiring end to end -- including that ``eval.prepare_data``'s generated
``database.yml`` is actually loadable by the installed ``pyannote.database``
(6.1.1), not just YAML-shape-correct -- rather than each stage's pure logic in
isolation. The real ``build_pipeline`` path (an actual
pyannote/speaker-diarization-3.1 model, the ``eval-run`` group, a GPU) is
exercised only by a real run against downloaded corpora (see
``eval/README.md``), not here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database import FileFinder, registry
from pyannote.database.util import load_rttm

# src/ is not a package; make its modules importable for the unit test
# (mirrors eval/tests/test_run.py, eval/tests/test_diarize_pipeline.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import diarize_pipeline
from eval.configs import DiarizeConfig
from eval.prepare_data import ProtocolPaths, write_database_yml
from eval.run import run_diarization
from eval.score import score_run
from eval.sweep import run_sweep, summarize_sweep

_SAMPLE_RATE = 16_000
_DURATION_S = 2.0


class _FakePipeline:
    """Deterministic stand-in for a pyannote Pipeline -- no torch, no model.

    Returns a canned two-speaker Annotation per file, matching the reference
    layout written by ``_write_rttm``, so scoring the result is a
    hand-verifiable perfect match (DER 0.0).
    """

    def __call__(self, payload: dict[str, object], **kwargs: object) -> Annotation:
        """Return a fixed two-speaker annotation regardless of input.

        Args:
            payload: The ``{"waveform", "sample_rate", "uri"}`` dict passed by
                ``run_diarization``; only ``uri`` is used, for the returned
                Annotation's identity.
            kwargs: Speaker-bound kwargs from ``DiarizeConfig.pipeline_kwargs``;
                unused -- the canned output is independent of the bounds.

        Returns:
            A two-speaker Annotation: 0-1s SPEAKER_00, 1-2s SPEAKER_01.
        """
        ann = Annotation(uri=str(payload.get("uri")))
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"
        ann[Segment(1.0, 2.0)] = "SPEAKER_01"
        return ann


def _write_wav(path: Path, seconds: float = _DURATION_S) -> None:
    """Write a tiny silent 16 kHz mono WAV fixture.

    Args:
        path: Destination path for the WAV file.
        seconds: Duration to synthesize.
    """
    sf.write(str(path), np.zeros(int(_SAMPLE_RATE * seconds), dtype=np.float32), _SAMPLE_RATE)


def _write_rttm(path: Path, uris: list[str]) -> None:
    """Write a two-speaker reference RTTM covering ``uris``.

    Each recording gets the same two-speaker layout (0-1s SPEAKER_00, 1-2s
    SPEAKER_01) that ``_FakePipeline`` returns as its hypothesis, so scoring
    is an easy, hand-verifiable perfect match (DER 0.0).

    Args:
        path: Destination RTTM path.
        uris: Recording ids to emit reference turns for.
    """
    lines = [
        f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for uri in uris
        for start, dur, speaker in ((0.0, 1.0, "SPEAKER_00"), (1.0, 1.0, "SPEAKER_01"))
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_uem(path: Path, uris: list[str], seconds: float = _DURATION_S) -> None:
    """Write a UEM covering the full duration of each recording.

    Args:
        path: Destination UEM path.
        uris: Recording ids to emit a scored region for.
        seconds: End time of the scored region, in seconds.
    """
    lines = [f"{uri} 1 0.000 {seconds:.3f}" for uri in uris]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_corpus(tmp_path: Path, database_name: str, uris: list[str]) -> Path:
    """Build a tiny two-file synthetic corpus and register it via write_database_yml.

    Args:
        tmp_path: Directory to build the corpus under (a pytest ``tmp_path``).
        database_name: Name to register the corpus/protocol under; kept
            distinct per test so the shared ``pyannote.database`` registry
            singleton never sees two tests register the same name.
        uris: Recording ids to synthesize audio and references for.

    Returns:
        The path to the written ``database.yml``.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for uri in uris:
        _write_wav(audio_dir / f"{uri}.wav")

    rttm_path = tmp_path / "test.rttm"
    uem_path = tmp_path / "test.uem"
    _write_rttm(rttm_path, uris)
    _write_uem(uem_path, uris)

    database_yml = tmp_path / "database.yml"
    write_database_yml(
        [ProtocolPaths(name=database_name, audio_dir=str(audio_dir), rttm=str(rttm_path), uem=str(uem_path))],
        str(database_yml),
    )
    return database_yml


def test_prepare_load_run_score_sweep_glue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: synthetic corpus -> database.yml -> run -> score -> sweep table.

    Builds a tiny two-file synthetic protocol, writes it through
    ``write_database_yml``, loads it back via the real ``pyannote.database``
    registry (proving the generated YAML plus its derived URI list are
    actually loadable, not just YAML-shape-correct), diarizes both files with
    a fake pipeline, scores the perfect-match hypotheses, and renders the
    sweep table. No torch, no model download, no network, no GPU -- and, like
    ``eval/tests/test_run.py``, no dependency on an ``ffmpeg`` binary being on
    ``PATH`` (``decode_audio`` is bypassed; the fake pipeline ignores the
    waveform content anyway, so only the *files existing* on disk matters).
    """
    uris = ["f1", "f2"]
    database_name = "SmokeGlue"
    database_yml = _build_corpus(tmp_path, database_name, uris)
    protocol_name = f"{database_name}.SpeakerDiarization.Benchmark"
    monkeypatch.setattr("eval.run.decode_audio", lambda data: np.zeros(_SAMPLE_RATE, dtype=np.float32))
    monkeypatch.setattr("eval.run._to_waveform", lambda audio: audio)  # skip the lazy torch import

    registry.load_database(str(database_yml))
    protocol = registry.get_protocol(protocol_name, preprocessors={"audio": FileFinder()})
    protocol_files = list(protocol.test())
    assert {f["uri"] for f in protocol_files} == set(uris)

    files = [(str(f["uri"]), str(f["audio"])) for f in protocol_files]
    references: dict[str, tuple[Annotation, Timeline | None]] = {
        str(f["uri"]): (f["annotation"], f["annotated"]) for f in protocol_files
    }

    out_dir = tmp_path / "hyp"
    config = DiarizeConfig(label="smoke-fake")
    written = run_diarization(_FakePipeline(), files, str(out_dir), config)
    assert len(written) == 2
    assert all(Path(p).is_file() for p in written)
    assert {Path(p).name for p in written} == {"f1.rttm", "f2.rttm"}

    items: list[tuple[str, Annotation, Annotation, Timeline | None]] = []
    for uri, _audio_path in files:
        reference, uem = references[uri]
        hypothesis_by_uri = load_rttm(str(out_dir / f"{uri}.rttm"))
        hypothesis = next(iter(hypothesis_by_uri.values()), Annotation(uri=uri))
        items.append((uri, reference, hypothesis, uem))

    report = score_run(items, collar=0.0)
    assert len(report.files) == 2
    assert report.overall_der == 0.0
    assert report.speaker_count_mae == 0.0
    assert report.speaker_count_bias == 0.0
    for file_score in report.files:
        assert file_score.ref_speakers == 2
        assert file_score.hyp_speakers == 2
        assert file_score.speaker_count_error == 0
        assert file_score.der == 0.0

    table = summarize_sweep([(config, report)])
    assert "smoke-fake" in table
    assert "0.000" in table


def test_run_sweep_orchestrates_the_same_glue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """run_sweep drives prepare->load->run->score->sweep itself, and logs the config.

    Same shape of synthetic corpus as the glue test above, but exercised
    through ``eval.sweep.run_sweep`` directly (the orchestration function)
    rather than by hand-calling each stage -- the only thing faked is the
    torch-heavy ``build_pipeline`` call (real models need a real run; see
    ``eval/README.md``). Also asserts a log record names the config label, so
    a sweep can never silently drop a config from its output without a trace.
    """
    uris = ["f1", "f2"]
    database_name = "SmokeSweep"
    database_yml = _build_corpus(tmp_path, database_name, uris)
    protocol_name = f"{database_name}.SpeakerDiarization.Benchmark"

    monkeypatch.setattr(diarize_pipeline, "build_pipeline", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr("eval.run.decode_audio", lambda data: np.zeros(_SAMPLE_RATE, dtype=np.float32))
    monkeypatch.setattr("eval.run._to_waveform", lambda audio: audio)  # skip the lazy torch import

    config = DiarizeConfig(label="smoke-sweep-fake")
    with caplog.at_level(logging.INFO, logger="eval.sweep"):
        table = run_sweep(str(database_yml), protocol_name, [config], str(tmp_path / "hyp"))

    assert "smoke-sweep-fake" in table
    assert "0.000" in table
    assert any("smoke-sweep-fake" in record.getMessage() for record in caplog.records)
