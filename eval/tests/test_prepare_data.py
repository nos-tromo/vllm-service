"""Tests for the pyannote.database registry generation."""

from pathlib import Path

import yaml

from eval.prepare_data import ProtocolPaths, build_database_yml, write_database_yml


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
    assert vox["scope"] == "file"
    assert vox["test"]["annotation"] == "/data/voxconverse/dev.rttm"
    assert vox["test"]["annotated"] == "/data/voxconverse/dev.uem"
    assert vox["test"]["uri"] == "/data/voxconverse/dev.rttm.uris.lst"
    assert "/data/voxconverse/audio" in parsed["Databases"]["VoxConverse"][0]


def test_write_database_yml_derives_sorted_uri_list_from_rttm(tmp_path: Path) -> None:
    """write_database_yml derives each protocol's uri list from its RTTM.

    pyannote.database's custom-protocol loader requires an explicit ``uri``
    list file (see ``_uris_lst_path``'s docstring) -- this pins that
    ``write_database_yml`` actually produces one, sorted and deduplicated,
    from the RTTM's own recording ids.
    """
    rttm_path = tmp_path / "dev.rttm"
    rttm_path.write_text(
        "SPEAKER b 1 0.000 1.000 <NA> <NA> spk1 <NA> <NA>\n"
        "SPEAKER a 1 0.000 1.000 <NA> <NA> spk1 <NA> <NA>\n"
        "SPEAKER a 1 1.000 1.000 <NA> <NA> spk2 <NA> <NA>\n",
        encoding="utf-8",
    )
    uem_path = tmp_path / "dev.uem"
    uem_path.write_text("a 1 0.000 2.000\nb 1 0.000 1.000\n", encoding="utf-8")
    out_path = tmp_path / "database.yml"

    write_database_yml(
        [ProtocolPaths(name="Vox", audio_dir=str(tmp_path / "audio"), rttm=str(rttm_path), uem=str(uem_path))],
        str(out_path),
    )

    assert out_path.exists()
    lst_path = Path(f"{rttm_path}.uris.lst")
    assert lst_path.read_text(encoding="utf-8").splitlines() == ["a", "b"]
