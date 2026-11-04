"""Every model launch loads the revision the deployment recorded.

The router records ``model_info.revision`` from a ``<MODEL>_REVISION``
variable (``docker/litellm.config.yaml``); the backends must load by that
same variable, or the audit log's ``ai.revision`` could name weights other
than the ones served. The vLLM launches are checked by running each
``docker/compose.yaml`` command under ``sh`` with a stub ``vllm`` that prints
its arguments; the first-party servers by importing each one over stubbed
loaders that record what they were asked to load. Unset or empty, no revision
is passed and the loaders keep their default (the cache's ``refs/main``).

The heavy ML deps are not installed in this env (see pyproject: the eval group
is torch-free by design), so they are stubbed in ``sys.modules`` for each test
and restored by ``monkeypatch`` afterwards. If the real ``transformers`` is
importable (the eval-run env), importing a server would load an actual
checkpoint, so the server tests are skipped there.
"""

import importlib.util
import shutil
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
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

needs_torch_free_env = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None,
    reason="server import tests need the torch-free env (real transformers would load the checkpoint)",
)


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


class _FakeLoaded:
    """What a stubbed ``from_pretrained`` returns: enough for each server's import."""

    bos_token_id = 0
    eos_token_id = 2

    class _Config:
        projection_dim = 4
        hidden_size = 4

    config = _Config()

    def train(self, mode: bool = True) -> "_FakeLoaded":
        """No-op train/eval toggle."""
        return self

    def to(self, device: str) -> "_FakeLoaded":
        """No-op device move."""
        return self


def _recording_loader(calls: list[tuple[str, dict[str, object]]], name: str) -> type:
    """Return a class whose ``from_pretrained`` records its kwargs under ``name``."""

    class _Loader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeLoaded:
            calls.append((name, dict(kwargs)))
            return _FakeLoaded()

    return _Loader


def _server_stubs(calls: list[tuple[str, dict[str, object]]]) -> dict[str, types.ModuleType]:
    """Stub torch, transformers, huggingface_hub and PIL for the server imports."""
    torch_stub = types.ModuleType("torch")
    torch_stub.float32 = "float32"  # type: ignore[attr-defined]
    torch_stub.float16 = "float16"  # type: ignore[attr-defined]
    torch_stub.no_grad = nullcontext  # type: ignore[attr-defined]
    torch_stub.load = lambda *_a, **_k: {}  # type: ignore[attr-defined]

    class _Linear:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def load_state_dict(self, *_a: object, **_k: object) -> None:
            return None

        def train(self, mode: bool = True) -> "_Linear":
            return self

    nn_stub = types.ModuleType("torch.nn")
    nn_stub.Linear = _Linear  # type: ignore[attr-defined]
    torch_stub.nn = nn_stub  # type: ignore[attr-defined]

    transformers_stub = types.ModuleType("transformers")
    for loader in ("AutoProcessor", "CLIPModel", "AutoTokenizer", "AutoModel", "AutoModelForSequenceClassification"):
        setattr(transformers_stub, loader, _recording_loader(calls, loader))

    def _hf_hub_download(**kwargs: object) -> str:
        calls.append(("hf_hub_download", {k: v for k, v in kwargs.items() if k not in ("repo_id", "filename")}))
        return "sparse_linear.pt"

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]

    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_stub.Image = pil_image_stub  # type: ignore[attr-defined]

    return {
        "torch": torch_stub,
        "torch.nn": nn_stub,
        "transformers": transformers_stub,
        "huggingface_hub": hub_stub,
        "PIL": pil_stub,
        "PIL.Image": pil_image_stub,
    }


def _import_fresh(module: str, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Execute ``src/<module>.py`` as a new module, so its import-time loading runs again."""
    spec = importlib.util.spec_from_file_location(f"_revision_pinning_{module}", SRC / f"{module}.py")
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    return loaded


SERVERS = [
    ("clip_server", "CLIP_MODEL_REVISION", ["AutoProcessor", "CLIPModel"]),
    ("rerank_server", "RERANK_MODEL_REVISION", ["AutoTokenizer", "AutoModelForSequenceClassification"]),
    ("embed_server", "EMBED_MODEL_REVISION", ["AutoTokenizer", "AutoModel", "hf_hub_download"]),
]


@needs_torch_free_env
@pytest.mark.parametrize(("module", "variable", "loaders"), SERVERS, ids=[s[0] for s in SERVERS])
def test_server_loads_the_recorded_revision(
    module: str, variable: str, loaders: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every file the server loads comes from the commit its variable names."""
    calls: list[tuple[str, dict[str, object]]] = []
    for name, stub in _server_stubs(calls).items():
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setenv(variable, "5" * 40)
    _import_fresh(module, monkeypatch)
    assert [name for name, _ in calls] == loaders
    assert all(kwargs.get("revision") == "5" * 40 for _, kwargs in calls)


@needs_torch_free_env
@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
@pytest.mark.parametrize(("module", "variable", "loaders"), SERVERS, ids=[s[0] for s in SERVERS])
def test_server_without_revision_passes_none(
    module: str, variable: str, loaders: list[str], value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset or empty, the loaders get no revision argument, whatever the other servers pin."""
    calls: list[tuple[str, dict[str, object]]] = []
    for name, stub in _server_stubs(calls).items():
        monkeypatch.setitem(sys.modules, name, stub)
    for other in {v for _, v, _ in SERVERS} | set(SHAS):
        monkeypatch.setenv(other, "7" * 40)
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, value)
    _import_fresh(module, monkeypatch)
    assert [name for name, _ in calls] == loaders
    assert all("revision" not in kwargs for _, kwargs in calls)


class _FakePipeline:
    """The loaded pyannote pipeline: no parameters to override, any device."""

    def parameters(self, instantiated: bool = False) -> dict[str, object]:
        """Return an empty parameter tree."""
        return {}

    def to(self, device: object) -> "_FakePipeline":
        """No-op device move."""
        return self


def _pyannote_stubs(calls: list[tuple[str, dict[str, object]]], *, legacy: bool) -> dict[str, types.ModuleType]:
    """Stub pyannote.audio (4.x signature, or 3.x with ``legacy``), torch and the compat shim."""

    class _Pipeline:
        @staticmethod
        def from_pretrained(checkpoint: str, revision: str | None = None, token: str | None = None) -> _FakePipeline:
            calls.append((checkpoint, {"revision": revision, "token": token}))
            return _FakePipeline()

    class _LegacyPipeline:
        @staticmethod
        def from_pretrained(checkpoint_path: str, use_auth_token: str | None = None) -> _FakePipeline:
            calls.append((checkpoint_path, {"use_auth_token": use_auth_token}))
            return _FakePipeline()

    audio_stub = types.ModuleType("pyannote.audio")
    audio_stub.Pipeline = _LegacyPipeline if legacy else _Pipeline  # type: ignore[attr-defined]
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda name: name  # type: ignore[attr-defined]
    return {"pyannote.audio": audio_stub, "torch": torch_stub, "diarize_compat": types.ModuleType("diarize_compat")}


def _build_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, legacy: bool = False, model_id: str | None = None
) -> list[tuple[str, dict[str, object]]]:
    """Run ``build_pipeline`` over the pyannote stubs; return the loader calls."""
    calls: list[tuple[str, dict[str, object]]] = []
    for name, stub in _pyannote_stubs(calls, legacy=legacy).items():
        monkeypatch.setitem(sys.modules, name, stub)
    for variable in ("DIARIZE_MODEL", "DIARIZE_DEVICE", "HF_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    diarize_pipeline = _import_fresh("diarize_pipeline", monkeypatch)
    diarize_pipeline.build_pipeline(model_id=model_id, device="cpu")
    return calls


def test_pipeline_loads_the_recorded_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """With pyannote 4.x, DIARIZE_MODEL (here its default) loads at DIARIZE_MODEL_REVISION."""
    monkeypatch.setenv("DIARIZE_MODEL_REVISION", "6" * 40)
    calls = _build_pipeline(monkeypatch)
    assert calls == [("pyannote/speaker-diarization-community-1", {"revision": "6" * 40, "token": None})]


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_pipeline_without_revision_passes_none(value: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset or empty, pyannote gets its own default revision (None), as before."""
    if value is None:
        monkeypatch.delenv("DIARIZE_MODEL_REVISION", raising=False)
    else:
        monkeypatch.setenv("DIARIZE_MODEL_REVISION", value)
    calls = _build_pipeline(monkeypatch)
    assert calls == [("pyannote/speaker-diarization-community-1", {"revision": None, "token": None})]


def test_pipeline_revision_does_not_apply_to_a_named_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """DIARIZE_MODEL_REVISION is DIARIZE_MODEL's commit, not that of a model_id the caller names."""
    monkeypatch.setenv("DIARIZE_MODEL_REVISION", "6" * 40)
    calls = _build_pipeline(monkeypatch, model_id="example-org/example-pipeline")
    assert calls == [("example-org/example-pipeline", {"revision": None, "token": None})]


@pytest.mark.parametrize(
    ("value", "checkpoint"),
    [
        ("6" * 40, f"pyannote/speaker-diarization-community-1@{'6' * 40}"),
        (None, "pyannote/speaker-diarization-community-1"),
    ],
    ids=["pinned", "unset"],
)
def test_legacy_pipeline_reads_the_revision_from_the_id(
    value: str | None, checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under pyannote 3.x, which has no revision kwarg, the revision rides on the id."""
    if value is None:
        monkeypatch.delenv("DIARIZE_MODEL_REVISION", raising=False)
    else:
        monkeypatch.setenv("DIARIZE_MODEL_REVISION", value)
    calls = _build_pipeline(monkeypatch, legacy=True)
    assert calls == [(checkpoint, {"use_auth_token": None})]
