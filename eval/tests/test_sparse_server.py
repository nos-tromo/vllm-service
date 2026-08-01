"""Route-level tests for the sparse-only server.

The heavy ML deps (torch, transformers) are not installed in this env
(see pyproject: the eval group is torch-free by design), so they are
stubbed in ``sys.modules`` before ``sparse_server`` is imported; the
tests then monkeypatch the tokenize/inference seams. If the real
``transformers`` is importable (the eval-run env), importing
``sparse_server`` would try to load the actual checkpoint, so the module
is skipped there.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

if importlib.util.find_spec("transformers") is not None:
    pytest.skip(
        "sparse_server unit tests need the torch-free env (real transformers would load the checkpoint at import)",
        allow_module_level=True,
    )


class _FakeTokenizer:
    """Minimal stand-in for transformers.AutoTokenizer output."""

    @staticmethod
    def from_pretrained(*_args: object, **_kwargs: object) -> "_FakeTokenizer":
        """Return the fake instance regardless of arguments."""
        return _FakeTokenizer()

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        """Return one id per whitespace token, wrapped in BOS/EOS."""
        return [0, *[100 + len(w) for w in text.split()], 2]


class _FakeModel:
    """Minimal stand-in for transformers.AutoModel."""

    class _Config:
        hidden_size = 1024

    config = _Config()

    @staticmethod
    def from_pretrained(*_args: object, **_kwargs: object) -> "_FakeModel":
        """Return the fake instance regardless of arguments."""
        return _FakeModel()

    def train(self, mode: bool = True) -> "_FakeModel":
        """No-op train/eval toggle."""
        return self


def _install_stubs() -> list[str]:
    """Stub torch + transformers + huggingface_hub in sys.modules.

    Returns:
        The names this call actually inserted, so they can be removed
        again after import. Leaving them in ``sys.modules`` would poison
        every test file that runs later — see the same dance in
        ``test_clip_server.py``.
    """
    torch_stub = types.ModuleType("torch")
    torch_stub.float32 = "float32"  # type: ignore[attr-defined]

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            return False

    torch_stub.no_grad = _NoGrad  # type: ignore[attr-defined]
    torch_stub.load = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    nn_stub = types.ModuleType("torch.nn")

    class _Linear:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def load_state_dict(self, *_a: object, **_k: object) -> None:
            return None

        def train(self, mode: bool = True) -> "_Linear":
            return self

    nn_stub.Linear = _Linear  # type: ignore[attr-defined]
    torch_stub.nn = nn_stub  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    transformers_stub.AutoModel = _FakeModel  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_stub

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.hf_hub_download = lambda *_a, **_k: "/nonexistent/sparse_linear.pt"  # type: ignore[attr-defined]

    stubs = {
        "torch": torch_stub,
        "torch.nn": nn_stub,
        "transformers": transformers_stub,
        "huggingface_hub": hub_stub,
    }
    return [name for name, module in stubs.items() if sys.modules.setdefault(name, module) is module]


_inserted_stubs = _install_stubs()

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    import sparse_server
finally:
    # sparse_server holds its own references to the stub modules; drop
    # them from sys.modules so other test files still get the real
    # ImportError (and the real torch where it is installed).
    for _name in _inserted_stubs:
        del sys.modules[_name]

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(sparse_server.app, raise_server_exceptions=False)


def test_health_reports_model() -> None:
    """/health returns ok plus the configured model id."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tokenize_returns_ids_under_tokens_key() -> None:
    """/tokenize response must satisfy docint's _extract_token_ids probe order.

    docint probes ``token_ids``, ``tokens``, ``prompt_token_ids`` in that
    order; ``tokens`` must therefore be a flat list of ints.
    """
    response = client.post("/tokenize", json={"model": "BAAI/bge-m3", "prompt": "alpha beta"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["tokens"], list)
    assert all(isinstance(token_id, int) for token_id in body["tokens"])
    assert body["tokens"] == [0, 105, 104, 2]


def test_tokenize_rejects_empty_prompt() -> None:
    """An empty prompt is a client error, not an empty-token response."""
    response = client.post("/tokenize", json={"model": "BAAI/bge-m3", "prompt": ""})
    assert response.status_code == 400
