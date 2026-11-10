"""How the faces service sits in the compose files.

Read from ``docker/compose.yaml`` and ``docker/compose.faces-only.yaml``; no
container is started.
"""

from pathlib import Path

import yaml

DOCKER = Path(__file__).resolve().parents[2] / "docker"


def _services(name: str) -> dict[str, dict[str, object]]:
    """Return the services of one compose file.

    Args:
        name: The file under ``docker/``.

    Returns:
        Each service's definition by name.
    """
    services: dict[str, dict[str, object]] = yaml.safe_load((DOCKER / name).read_text())["services"]
    return services


def test_router_waits_only_for_faces_to_start() -> None:
    """A faces backend without its model files must not keep inference down."""
    router = _services("compose.yaml")["router"]
    depends_on = router["depends_on"]
    assert isinstance(depends_on, dict)
    assert depends_on["faces"] == {"condition": "service_started"}


def test_faces_keeps_its_own_healthcheck() -> None:
    """The service still reports whether it serves, in both shapes."""
    assert "healthcheck" in _services("compose.yaml")["faces"]
    assert "healthcheck" in _services("compose.faces-only.yaml")["faces-only"]


def test_faces_revisions_come_from_the_env_file_only() -> None:
    """The revision variables are not interpolated into ``environment:``.

    Interpolation reads the operator's shell first, so the router (which reads
    the env_file) and the backend could get different commits.
    """
    for name, service in (("compose.yaml", "faces"), ("compose.faces-only.yaml", "faces-only")):
        environment = _services(name)[service]["environment"]
        assert isinstance(environment, dict)
        assert not [variable for variable in environment if str(variable).endswith("_REVISION")]
        assert _services(name)[service]["env_file"]
