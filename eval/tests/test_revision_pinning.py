"""Every vLLM launch loads the revision the router records.

The router records ``model_info.revision`` from a ``<MODEL>_REVISION``
variable (``docker/litellm.config.yaml``); the backends must load by that
same variable, or the audit log's ``ai.revision`` could name weights other
than the ones served. Each ``docker/compose.yaml`` command runs under ``sh``
with a stub ``vllm`` that prints its arguments. Unset or empty, no revision
is passed and vLLM keeps its default (the cache's ``refs/main``).
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "docker" / "compose.yaml"
LITELLM_CONFIG = REPO / "docker" / "litellm.config.yaml"

#: A distinct fake commit per variable, so a launch wired to another model's
#: variable fails instead of passing by coincidence.
SHAS: dict[str, str] = {
    "TEXT_MODEL_REVISION": "1" * 40,
    "EMBED_MODEL_REVISION": "2" * 40,
    "RERANK_MODEL_REVISION": "3" * 40,
    "WHISPER_MODEL_REVISION": "4" * 40,
}

#: embed-sparse serves EMBED_MODEL through a pass-through route, so it has no
#: model_info of its own; it loads the same revision as embed.
PASS_THROUGH_LAUNCHES = {"embed-sparse": "EMBED_MODEL_REVISION"}


def _compose_services() -> dict[str, dict[str, object]]:
    """Return the services of the full-stack compose file."""
    services: dict[str, dict[str, object]] = yaml.safe_load(COMPOSE.read_text())["services"]
    return services


def _router_revision_vars() -> dict[str, str]:
    """Map each router deployment id to the variable its ``model_info.revision`` reads."""
    config = yaml.safe_load(LITELLM_CONFIG.read_text())
    return {
        str(entry["model_info"]["id"]): str(entry["model_info"]["revision"]).removeprefix("os.environ/")
        for entry in config["model_list"]
    }


def _vllm_launches() -> dict[str, str]:
    """Map every vLLM service to the revision variable it must load by."""
    return {**_router_revision_vars(), **PASS_THROUGH_LAUNCHES}


def _launch_argv(service: str, env: dict[str, str], tmp_path: Path) -> list[str]:
    """Run a service's compose command with a stub ``vllm``; return the stub's argv.

    Args:
        service: The compose service whose ``command`` to run.
        env: Variables the container would see (beyond ``PATH``).
        tmp_path: Where the stub ``vllm`` is written.

    Returns:
        The arguments the command passed to ``vllm``, one per item.
    """
    command = _compose_services()[service]["command"]
    assert isinstance(command, list)
    # Compose turns '$$' into a literal '$' before the container's shell sees it.
    script = str(command[0]).replace("$$", "$")
    stub = tmp_path / "vllm"
    stub.write_text('#!/bin/sh\nfor arg in "$@"; do printf "%s\\n" "$arg"; done\n')
    stub.chmod(0o755)
    shell = shutil.which("dash") or "/bin/sh"
    result = subprocess.run(
        [shell, "-c", script],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def _flag_values(argv: list[str], flag: str) -> list[str]:
    """Return the value after each occurrence of ``flag`` in ``argv``."""
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]


def test_every_vllm_launch_is_pinned() -> None:
    """Each compose service that runs ``vllm serve`` has a revision variable here.

    A vLLM backend added without one would serve whatever the cache resolves
    to while this suite stayed green.
    """
    vllm_services = set()
    for name, service in _compose_services().items():
        command = service.get("command")
        if isinstance(command, list) and any("vllm serve" in str(part) for part in command):
            vllm_services.add(name)
    assert vllm_services == set(_vllm_launches())


@pytest.mark.parametrize("others", ["unset", "set"])
@pytest.mark.parametrize("service", sorted(_vllm_launches()))
def test_vllm_launch_loads_the_recorded_revision(service: str, others: str, tmp_path: Path) -> None:
    """The launch passes its own variable as ``--revision`` and ``--tokenizer-revision``.

    With only its own variable set, a launch keyed to another one would pass
    no flag; with every variable set, one reading another's value would pass
    the wrong commit.
    """
    variable = _vllm_launches()[service]
    env = dict(SHAS) if others == "set" else {variable: SHAS[variable]}
    argv = _launch_argv(service, env, tmp_path)
    assert argv[:1] == ["serve"]
    assert _flag_values(argv, "--revision") == [SHAS[variable]]
    assert _flag_values(argv, "--tokenizer-revision") == [SHAS[variable]]


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
@pytest.mark.parametrize("service", sorted(_vllm_launches()))
def test_vllm_launch_without_revision_passes_no_flag(service: str, value: str | None, tmp_path: Path) -> None:
    """Unset or empty, the launch is today's: no revision flag, whatever the other models pin."""
    variable = _vllm_launches()[service]
    env = {name: sha for name, sha in SHAS.items() if name != variable}
    if value is not None:
        env[variable] = value
    argv = _launch_argv(service, env, tmp_path)
    assert argv[:1] == ["serve"]
    assert "--revision" not in argv
    assert "--tokenizer-revision" not in argv
