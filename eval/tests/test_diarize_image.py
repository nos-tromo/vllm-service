"""Static checks on how the two diarization images install their dependencies.

``pyannote.audio`` used to be installed unpinned (``>=4,<5``), so a rebuild picked up 4.1, which removed
``Audio(mono=...)`` and broke the build smoke test. The images now install a lock (``docker/diarize/``) and pin the
torch stack in the Dockerfile. These tests read the Dockerfiles and the lock as text, so they need neither Docker nor
the heavy dependencies; the build itself is the real check.
"""

import re
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
