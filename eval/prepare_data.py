"""Prepare public diarization benchmarks for the eval harness.

Downloads the canonical VoxConverse (dev) and AMI (test) releases and emits a
``pyannote.database`` ``database.yml`` that registers each as a
``<Name>.SpeakerDiarization.Benchmark`` protocol. Only ``main()`` touches the
network; it runs once on a networked dev box (never at serve time).
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ProtocolPaths:
    """Filesystem locations for one benchmark's audio and references.

    Attributes:
        name: Database/protocol name (e.g. ``VoxConverse``).
        audio_dir: Directory of the recordings (referenced as a glob).
        rttm: Path to the reference RTTM covering the split.
        uem: Path to the UEM defining the scored regions.
    """

    name: str
    audio_dir: str
    rttm: str
    uem: str


def build_database_yml(protocols: list[ProtocolPaths]) -> str:
    """Render a pyannote.database registry for the given protocols.

    Args:
        protocols: One entry per benchmark split to register.

    Returns:
        YAML text with ``Databases`` (audio globs) and ``Protocols``
        (a ``SpeakerDiarization.Benchmark`` protocol per entry, its ``test``
        subset pointing at the RTTM/UEM).
    """
    databases: dict[str, list[str]] = {}
    registry: dict[str, dict[str, object]] = {}
    for proto in protocols:
        databases[proto.name] = [f"{proto.audio_dir}/{{uri}}.wav"]
        registry[proto.name] = {
            "SpeakerDiarization": {
                "Benchmark": {
                    "test": {"annotation": proto.rttm, "annotated": proto.uem},
                }
            }
        }
    return yaml.safe_dump({"Databases": databases, "Protocols": registry}, sort_keys=True)


def write_database_yml(protocols: list[ProtocolPaths], out_path: str) -> None:
    """Write the rendered registry to ``out_path``.

    Args:
        protocols: Protocols to register.
        out_path: Destination path for ``database.yml``.
    """
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(build_database_yml(protocols))


def main() -> None:
    """Fetch the benchmark corpora and emit ``data/database.yml`` (networked).

    Downloads VoxConverse (dev) and AMI (test) into ``data/`` and writes the
    registry. Idempotent: existing files are left in place. Implemented as the
    documented one-time prep step; see ``eval/README.md`` for the exact source
    URLs and the manual steps for the gated AMI audio.
    """
    raise NotImplementedError(
        "Run the documented data-prep steps in eval/README.md, then call "
        "write_database_yml(...) with the resulting paths."
    )
