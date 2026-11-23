"""Static checks on how the two diarization images install their dependencies.

``pyannote.audio`` used to be installed unpinned (``>=4,<5``), so a rebuild picked up 4.1, which removed
``Audio(mono=...)`` and broke the build smoke test. The images now install a lock (``docker/diarize/``) and pin the
torch stack in the Dockerfile. These tests read the Dockerfiles and the lock as text, so they need neither Docker nor
the heavy dependencies; the build itself is the real check.
"""

import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
CPU = DOCKER / "Dockerfile.diarize.cpu"
CUDA = DOCKER / "Dockerfile.diarize.cuda"
DOCKERFILES = pytest.mark.parametrize("dockerfile", [CPU, CUDA], ids=lambda path: path.name)

PYANNOTE = "4.0.7"
TORCH = "2.11.0"
TORCH_STACK = ("torch", "torchaudio", "torchcodec")


def _instructions(dockerfile: Path) -> list[str]:
    """Return a Dockerfile's instructions, comments dropped and continuation lines joined.

    Args:
        dockerfile: The Dockerfile to read.

    Returns:
        One whitespace-normalised string per instruction.
    """
    lines = [line for line in dockerfile.read_text().splitlines() if not line.lstrip().startswith("#")]
    return [" ".join(part.split()) for part in "\n".join(lines).replace("\\\n", " ").split("\n") if part.strip()]


def _installs(dockerfile: Path, marker: str) -> list[str]:
    """Return the commands of a Dockerfile that contain ``marker``.

    A ``RUN`` that chains commands with ``&&`` counts as one command per link, so a check on one
    ``pip install`` does not read the options of its neighbour.

    Args:
        dockerfile: The Dockerfile to read.
        marker: The text to look for.

    Returns:
        The matching commands, each whitespace-normalised.
    """
    commands = [command.strip() for line in _instructions(dockerfile) for command in line.split("&&")]
    return [command for command in commands if marker in command]


def _runtime_instructions(dockerfile: Path) -> list[str]:
    """Return the instructions of a Dockerfile's last stage, the one that ships.

    Args:
        dockerfile: The Dockerfile to read.

    Returns:
        The instructions after its last ``FROM``, ``FROM`` included.
    """
    instructions = _instructions(dockerfile)
    last = max(index for index, line in enumerate(instructions) if line.startswith("FROM "))
    return instructions[last:]


def _env(dockerfile: Path) -> dict[str, str]:
    """Return the variables the last stage of a Dockerfile sets with ``ENV``.

    Args:
        dockerfile: The Dockerfile to read.

    Returns:
        The variable names and their values.
    """
    variables: dict[str, str] = {}
    for line in _runtime_instructions(dockerfile):
        if line.startswith("ENV "):
            for token in shlex.split(line.removeprefix("ENV ")):
                name, _, value = token.partition("=")
                variables[name] = value
    return variables


def _lock() -> dict[str, Any]:
    """Return the parsed ``docker/diarize/uv.lock``.

    Returns:
        The lock file's TOML as a dict.
    """
    return tomllib.loads((DOCKER / "diarize" / "uv.lock").read_text())


@DOCKERFILES
def test_no_dockerfile_passes_mono_to_audio(dockerfile: Path) -> None:
    """``Audio(mono=...)`` was removed in pyannote.audio 4.1, so the smoke test must not pass it."""
    assert "mono=" not in dockerfile.read_text()


@DOCKERFILES
def test_the_smoke_test_builds_audio_with_a_sample_rate_only(dockerfile: Path) -> None:
    """The one call that works on 4.0.x and 4.1.x is ``Audio(sample_rate=16000)``."""
    assert "Audio(sample_rate=16000);" in dockerfile.read_text()


@DOCKERFILES
def test_the_image_installs_the_lock_with_hashes_and_no_dependency_resolution(dockerfile: Path) -> None:
    """The closure comes from the exported lock, hash-checked, without resolving anything at build time."""
    (export,) = _installs(dockerfile, "uv export")
    assert "--locked" in export
    assert "--format requirements-txt" in export
    (install,) = _installs(dockerfile, "--require-hashes")
    assert "--no-deps" in install
    assert re.search(r"-r /(build|tmp)/requirements\.txt$", install)
    (copy,) = _installs(dockerfile, "docker/diarize/uv.lock")
    assert copy.startswith("COPY docker/diarize/pyproject.toml docker/diarize/uv.lock ")


@DOCKERFILES
def test_no_dockerfile_installs_pyannote_outside_the_lock(dockerfile: Path) -> None:
    """A direct ``pip``/``uv pip install pyannote...`` would bypass the lock."""
    assert not re.search(r"pyannote", " ".join(_installs(dockerfile, "install")), flags=re.IGNORECASE)


def test_the_lock_pins_pyannote_audio() -> None:
    """The lock holds the pyannote.audio release the eval reports were measured on."""
    versions = {package["name"]: package["version"] for package in _lock()["package"]}
    assert versions["pyannote-audio"] == PYANNOTE


def test_the_project_pins_pyannote_audio_exactly() -> None:
    """The lock only holds if its project file pins the release, not a range."""
    project = tomllib.loads((DOCKER / "diarize" / "pyproject.toml").read_text())
    assert f"pyannote.audio=={PYANNOTE}" in project["project"]["dependencies"]


def test_the_lock_leaves_the_torch_stack_to_the_dockerfiles() -> None:
    """The Dockerfiles pin torch, torchaudio and torchcodec; the lock must not also carry (or hash) them."""
    names = {package["name"] for package in _lock()["package"]}
    assert not names & set(TORCH_STACK)
    assert not [name for name in names if name.startswith("nvidia-") or name == "triton"]
    project = tomllib.loads((DOCKER / "diarize" / "pyproject.toml").read_text())
    assert set(project["tool"]["uv"]["exclude-dependencies"]) == set(TORCH_STACK)


def test_the_lock_covers_the_python_of_both_images() -> None:
    """The CPU image runs the uv base's Python 3.11 and the CUDA image the PyTorch base's 3.12."""
    assert _lock()["requires-python"] == ">=3.11, <3.13"


def test_the_cpu_image_pins_the_torch_stack() -> None:
    """The CPU image pins torch and torchaudio to the CUDA base's torch, and a torchcodec version."""
    (install,) = _installs(CPU, "--index-url https://download.pytorch.org/whl/cpu")
    assert f"torch=={TORCH} " in install + " "
    assert f"torchaudio=={TORCH} " in install + " "
    assert re.search(r"\storchcodec==\d+\.\d+\.\d+(\s|$)", install)


def test_the_cuda_image_pins_torchaudio_and_torchcodec_on_the_base_torch() -> None:
    """The CUDA image keeps the base's torch, pins the rest of the stack and names the torch it expects."""
    assert f"FROM docker.io/pytorch/pytorch:{TORCH}-" in CUDA.read_text()
    (torchaudio,) = _installs(CUDA, "torchaudio==")
    assert f"'torchaudio=={TORCH}'" in torchaudio
    assert "https://download.pytorch.org/whl/cu128" in torchaudio
    (torchcodec,) = _installs(CUDA, "torchcodec==")
    assert re.search(r"'torchcodec==\d+\.\d+\.\d+'", torchcodec)
    assert "https://download.pytorch.org/whl/cpu" in torchcodec


@DOCKERFILES
def test_the_image_turns_pyannote_telemetry_off(dockerfile: Path) -> None:
    """pyannote.audio 4.x sends OpenTelemetry usage spans unless told not to; nothing leaves an airgapped stack."""
    assert _env(dockerfile).get("PYANNOTE_METRICS_ENABLED") == "false"


@DOCKERFILES
def test_the_telemetry_is_off_before_the_smoke_test_imports_pyannote(dockerfile: Path) -> None:
    """The variable is set ahead of the smoke test, so the build imports pyannote as the container will."""
    instructions = _runtime_instructions(dockerfile)
    off = next(i for i, line in enumerate(instructions) if line == "ENV PYANNOTE_METRICS_ENABLED=false")
    smoke = next(i for i, line in enumerate(instructions) if "diarize build smoke: OK" in line)
    assert off < smoke


@DOCKERFILES
def test_the_caches_of_a_read_only_container_are_under_tmp(dockerfile: Path) -> None:
    """The root filesystem is read-only and only ``/tmp`` is writable, so matplotlib and fontconfig cache there."""
    variables = _env(dockerfile)
    assert variables.get("MPLCONFIGDIR", "").startswith("/tmp/")
    assert variables.get("XDG_CACHE_HOME", "").startswith("/tmp/")


@DOCKERFILES
def test_the_image_checks_its_dependencies_after_the_lock_install(dockerfile: Path) -> None:
    """A locked package that needs another torch than the pinned one fails the build, not the first request."""
    commands = [command for line in _runtime_instructions(dockerfile) for command in line.split("&&")]
    commands = [command.strip() for command in commands]
    install = next(i for i, command in enumerate(commands) if "--require-hashes" in command)
    checks = [i for i, command in enumerate(commands) if re.search(r"\bpip check\b", command)]
    assert checks
    assert min(checks) > install


def test_the_cpu_image_keeps_the_working_directory_of_its_base() -> None:
    """The build files are used from ``/build`` without leaving it as the runtime working directory."""
    assert not [line for line in _runtime_instructions(CPU) if line.startswith("WORKDIR ")]
