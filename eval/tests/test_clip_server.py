"""Error-mapping tests for the clip server's /clip/embed_image endpoint.

A GPU-runtime failure (e.g. ``RuntimeError: cuDNN error:
CUDNN_STATUS_INTERNAL_ERROR`` under memory pressure) is a transient
server-side fault and must surface as 503 so consumers retry, while a
genuinely undecodable upload stays 422 (the client's fault, retrying is
pointless). Regression test for the incident where a cuDNN OOM was
returned as 422 and docint stopped retrying.

The heavy ML deps (torch, transformers, PIL) are not installed in this
env (see pyproject: the eval group is torch-free by design), so they are
stubbed in ``sys.modules`` before ``clip_server`` is imported; the tests
then monkeypatch the decode/inference seams to inject failures. If the
real ``transformers`` is importable (the eval-run env), importing
``clip_server`` would try to load the actual checkpoint, so the module
is skipped there.
"""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

if importlib.util.find_spec("transformers") is not None:
    pytest.skip(
        "clip_server unit tests need the torch-free env (real transformers would load the checkpoint at import)",
        allow_module_level=True,
    )


class _FakeCLIPModel:
    """Minimal stand-in for transformers.CLIPModel; forwards are patched per-test."""

    class _Config:
        projection_dim = 512

    config = _Config()

    def train(self, mode: bool = True) -> "_FakeCLIPModel":
        """No-op train/eval toggle."""
        return self

    def to(self, device: str) -> "_FakeCLIPModel":
        """No-op device move."""
        return self

    def get_image_features(self, **kwargs: object) -> object:
        """Placeholder forward; tests monkeypatch this."""
        raise NotImplementedError

    def get_text_features(self, **kwargs: object) -> object:
        """Placeholder forward; tests monkeypatch this."""
        raise NotImplementedError


def _install_ml_stubs() -> list[str]:
    """Register torch/transformers/PIL stubs so clip_server imports without the ML stack.

    Returns:
        The sys.modules keys that were inserted, so the caller can remove
        them after importing clip_server — other test modules in this
        session must keep seeing the real ImportError for these packages.
    """
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = nullcontext  # type: ignore[attr-defined]

    transformers_stub = types.ModuleType("transformers")

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str) -> object:
            def _processor(**kwargs: object) -> dict[str, object]:
                return {}

            return _processor

    class _CLIPModel:
        @staticmethod
        def from_pretrained(model_id: str) -> _FakeCLIPModel:
            return _FakeCLIPModel()

    transformers_stub.AutoProcessor = _AutoProcessor  # type: ignore[attr-defined]
    transformers_stub.CLIPModel = _CLIPModel  # type: ignore[attr-defined]

    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")

    def _open_unpatched(fp: object) -> object:
        raise AssertionError("test must monkeypatch Image.open")

    pil_image_stub.open = _open_unpatched  # type: ignore[attr-defined]
    pil_stub.Image = pil_image_stub  # type: ignore[attr-defined]

    stubs = {
        "torch": torch_stub,
        "transformers": transformers_stub,
        "PIL": pil_stub,
        "PIL.Image": pil_image_stub,
    }
    inserted = [name for name, module in stubs.items() if sys.modules.setdefault(name, module) is module]
    return inserted


_inserted_stubs = _install_ml_stubs()

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    import clip_server
finally:
    # clip_server holds its own references to the stub modules; drop them
    # from sys.modules so other test files still get the real ImportError.
    for _name in _inserted_stubs:
        del sys.modules[_name]

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(clip_server.app, raise_server_exceptions=False)


class _DecodedImage:
    """What a successful Image.open(...).convert('RGB') yields."""

    def convert(self, mode: str) -> "_DecodedImage":
        """Return self; mode conversion is irrelevant to the stub."""
        return self


def test_undecodable_image_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bytes Pillow cannot decode are the client's fault → 422."""

    def _open_fails(fp: object) -> object:
        raise ValueError("cannot identify image file")

    monkeypatch.setattr(clip_server.Image, "open", _open_fails)
    resp = client.post("/clip/embed_image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 422
    assert "decode" in resp.json()["detail"]


def test_inference_failure_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU-runtime failure after a clean decode is transient → 503, not 422."""
    monkeypatch.setattr(clip_server.Image, "open", lambda fp: _DecodedImage())

    def _cudnn_blows_up(**kwargs: object) -> object:
        raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")

    monkeypatch.setattr(clip_server.model, "get_image_features", _cudnn_blows_up)
    resp = client.post("/clip/embed_image", files={"file": ("x.jpg", b"valid image bytes", "image/jpeg")})
    assert resp.status_code == 503
    assert "CUDNN_STATUS_INTERNAL_ERROR" in resp.json()["detail"]


def test_text_inference_failure_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text tower can hit the same GPU-runtime failures → 503 there too."""

    def _cudnn_blows_up(**kwargs: object) -> object:
        raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")

    monkeypatch.setattr(clip_server.model, "get_text_features", _cudnn_blows_up)
    resp = client.post("/clip/embed_text", json={"text": "a photo of a cat"})
    assert resp.status_code == 503
    assert "CUDNN_STATUS_INTERNAL_ERROR" in resp.json()["detail"]
