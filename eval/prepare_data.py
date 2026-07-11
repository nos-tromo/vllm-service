"""Prepare public diarization benchmarks for the eval harness.

Registers already-downloaded VoxConverse (dev) / AMI (test) corpora as a
``pyannote.database`` ``database.yml`` with a ``<Name>.SpeakerDiarization.Benchmark``
protocol per corpus. Only ``main()`` touches the filesystem beyond the
registry file itself (reading each RTTM to derive its URI list); nothing
here downloads anything — corpus acquisition is a manual, documented step
(see ``eval/README.md``). It runs once on a networked dev box, never at
serve time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyannote.database.util import load_rttm


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


def _uris_lst_path(rttm_path: str) -> str:
    """Return the deterministic sibling path for a protocol's URI list.

    pyannote.database's custom-protocol loader requires an explicit ``uri``
    list file (one recording id per line) alongside ``annotation``/
    ``annotated`` in each subset block — unlike those two, it never derives
    the file list from the RTTM's own contents (confirmed against the
    installed pyannote.database 6.1.1: a subset with only ``annotation``/
    ``annotated`` and no ``uri``/``uris`` entry raises ``ValueError: Missing
    mandatory 'uri' entry``). ``write_database_yml`` derives that list from
    the RTTM itself and writes it here, so it always matches what the RTTM
    actually contains.

    Args:
        rttm_path: Path to the protocol's reference RTTM.

    Returns:
        The path the derived URI list is written to / referenced from.
    """
    return f"{rttm_path}.uris.lst"


def build_database_yml(protocols: list[ProtocolPaths]) -> str:
    """Render a pyannote.database registry for the given protocols.

    Args:
        protocols: One entry per benchmark split to register.

    Returns:
        YAML text with ``Databases`` (audio globs) and ``Protocols`` (a
        file-scoped ``SpeakerDiarization.Benchmark`` protocol per entry,
        whose ``test`` subset points at the RTTM/UEM and at the derived URI
        list from ``_uris_lst_path``).
    """
    databases: dict[str, list[str]] = {}
    registry: dict[str, dict[str, object]] = {}
    for proto in protocols:
        databases[proto.name] = [f"{proto.audio_dir}/{{uri}}.wav"]
        registry[proto.name] = {
            "SpeakerDiarization": {
                "Benchmark": {
                    # VoxConverse/AMI speaker labels (e.g. SPEAKER_00) are only
                    # meaningful within a single recording. Declaring the scope
                    # explicitly also silences pyannote.database's "protocol
                    # does not define the scope of speaker labels" warning.
                    "scope": "file",
                    "test": {
                        "uri": _uris_lst_path(proto.rttm),
                        "annotation": proto.rttm,
                        "annotated": proto.uem,
                    },
                }
            }
        }
    return yaml.safe_dump({"Databases": databases, "Protocols": registry}, sort_keys=True)


def write_database_yml(protocols: list[ProtocolPaths], out_path: str) -> None:
    """Write the rendered registry (plus its derived URI lists) to disk.

    Alongside ``out_path``, writes one ``<rttm>.uris.lst`` file per protocol
    (see ``_uris_lst_path``) listing every recording id found in that
    protocol's RTTM, sorted — the ``uri`` entry ``pyannote.database`` requires
    to enumerate a custom protocol's files.

    Args:
        protocols: Protocols to register.
        out_path: Destination path for ``database.yml``.
    """
    for proto in protocols:
        uris = sorted(load_rttm(proto.rttm).keys())
        Path(_uris_lst_path(proto.rttm)).write_text("\n".join(uris) + "\n", encoding="utf-8")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(build_database_yml(protocols))


def _require_all_or_none(parser: argparse.ArgumentParser, flag_group: str, values: tuple[str | None, ...]) -> bool:
    """Validate that a protocol's flag trio was given together or not at all.

    Args:
        parser: The parser to raise a usage error through (exits the process).
        flag_group: Human-readable flag names, for the error message.
        values: The trio's raw argument values (audio dir, RTTM, UEM).

    Returns:
        True if every value in ``values`` was given (so the protocol should
        be registered); False if none were given (so it should be skipped).

    Raises:
        SystemExit: Via ``parser.error`` if only some of ``values`` were given.
    """
    given = [value is not None for value in values]
    if any(given) and not all(given):
        parser.error(f"{flag_group} must be given together")
    return all(given)


def main() -> None:
    """CLI: register already-downloaded corpora as a pyannote.database registry.

    Takes filesystem paths to corpora that have already been downloaded and
    normalized per ``eval/README.md`` — this does not fetch anything itself.
    Each protocol's trio of flags (audio dir, RTTM, UEM) must be given
    together or not at all; at least one protocol must be given. Registering
    both VoxConverse and AMI requires passing both trios in the same
    invocation, since each run overwrites ``--out`` rather than merging into
    an existing registry.
    """
    parser = argparse.ArgumentParser(
        description="Register already-downloaded diarization corpora as a pyannote.database registry.",
    )
    parser.add_argument("--voxconverse-audio-dir", default=None, help="Directory of VoxConverse dev *.wav files.")
    parser.add_argument("--voxconverse-rttm", default=None, help="Path to the VoxConverse dev reference RTTM.")
    parser.add_argument("--voxconverse-uem", default=None, help="Path to the VoxConverse dev UEM.")
    parser.add_argument("--ami-audio-dir", default=None, help="Directory of AMI test *.wav files.")
    parser.add_argument("--ami-rttm", default=None, help="Path to the AMI test reference RTTM.")
    parser.add_argument("--ami-uem", default=None, help="Path to the AMI test UEM.")
    parser.add_argument("--out", default="data/database.yml", help="Destination path for the generated database.yml.")
    args = parser.parse_args()

    protocols: list[ProtocolPaths] = []
    if _require_all_or_none(
        parser,
        "--voxconverse-audio-dir/--voxconverse-rttm/--voxconverse-uem",
        (args.voxconverse_audio_dir, args.voxconverse_rttm, args.voxconverse_uem),
    ):
        protocols.append(
            ProtocolPaths(
                name="VoxConverse",
                audio_dir=args.voxconverse_audio_dir,
                rttm=args.voxconverse_rttm,
                uem=args.voxconverse_uem,
            )
        )
    if _require_all_or_none(
        parser, "--ami-audio-dir/--ami-rttm/--ami-uem", (args.ami_audio_dir, args.ami_rttm, args.ami_uem)
    ):
        protocols.append(ProtocolPaths(name="AMI", audio_dir=args.ami_audio_dir, rttm=args.ami_rttm, uem=args.ami_uem))

    if not protocols:
        parser.error("at least one protocol's flag trio must be given (VoxConverse and/or AMI)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_database_yml(protocols, str(out_path))
    print(f"wrote {out_path} registering: {', '.join(p.name for p in protocols)}")


if __name__ == "__main__":
    main()
