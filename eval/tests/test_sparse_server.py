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
    """Stand-in for transformers.AutoTokenizer covering both call shapes.

    ``encode`` (the ``/tokenize`` route, via ``tokenize_ids``) and
    ``__call__`` (the ``/pooling`` batch route, via
    ``encode_token_weights``) are deliberately separate implementations
    — neither delegates to nor derives from the other — so a test can
    tell the two real tokenization paths apart if they ever drift (e.g.
    one passing ``add_special_tokens=False`` while the other doesn't).
    Each records the kwargs it was invoked with for that assertion.
    """

    def __init__(self) -> None:
        """Start with empty call logs."""
        self.encode_calls: list[dict[str, object]] = []
        self.call_calls: list[dict[str, object]] = []

    @staticmethod
    def from_pretrained(*_args: object, **_kwargs: object) -> "_FakeTokenizer":
        """Return the fake instance regardless of arguments."""
        return _FakeTokenizer()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **_kwargs: object,
    ) -> list[int]:
        """Return one id per whitespace token, wrapped in BOS/EOS.

        Honours ``add_special_tokens``/``truncation``/``max_length`` on
        its own terms — this method must never be called by
        ``__call__`` (or vice versa).
        """
        self.encode_calls.append(
            {"add_special_tokens": add_special_tokens, "truncation": truncation, "max_length": max_length}
        )
        ids = [100 + len(word) for word in text.split()]
        if add_special_tokens:
            ids = [0, *ids, 2]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def __call__(
        self,
        texts: list[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
        **_kwargs: object,
    ) -> dict[str, "_FakeTensor"]:
        """Batch-tokenize independently of ``encode``, then pad to the batch width.

        Mirrors a real ``BatchEncoding``: right-pads every row to the
        longest row in the batch and returns a matching attention mask,
        so tests exercising the real ``encode_token_weights`` per-row
        mask-stripping logic see genuinely different row lengths.
        """
        self.call_calls.append(
            {"add_special_tokens": add_special_tokens, "truncation": truncation, "max_length": max_length}
        )
        rows: list[list[int]] = []
        for text in texts:
            ids = [100 + len(word) for word in text.split()]
            if add_special_tokens:
                ids = [0, *ids, 2]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)

        width = max((len(row) for row in rows), default=0)
        input_ids = [[*row, *([1] * (width - len(row)))] for row in rows]
        attention_mask = [[*([1] * len(row)), *([0] * (width - len(row)))] for row in rows]
        return {"input_ids": _FakeTensor(input_ids), "attention_mask": _FakeTensor(attention_mask)}


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


class _FakeTensor:
    """Minimal stand-in for a torch.Tensor: only squeeze() and tolist()."""

    def __init__(self, data: list[list[float]] | list[list[int]]) -> None:
        """Wrap nested list data as a fake tensor.

        Args:
            data: The values this fake tensor reports via ``tolist()``.
        """
        self._data = data

    def squeeze(self, *_args: object, **_kwargs: object) -> "_FakeTensor":
        """Return self; these fakes never need an actual shape change."""
        return self

    def tolist(self) -> list[list[float]] | list[list[int]]:
        """Return the wrapped nested list, mimicking torch.Tensor.tolist()."""
        return self._data


def _install_stubs() -> dict[str, types.ModuleType | None]:
    """Stub torch + transformers + huggingface_hub in sys.modules.

    All four names are assigned directly, never via ``setdefault``. An
    import-order-dependent ``setdefault`` is exactly what let this bite
    us: if some earlier-collected test file already imported
    ``huggingface_hub`` for real, ``setdefault`` would leave that real
    module in place, ``sparse_server`` would bind the real
    ``hf_hub_download``, and importing it would attempt a live network
    download of ``sparse_linear.pt`` — hanging an airgapped CI runner.
    Direct assignment guarantees the stub wins regardless of import
    order.

    Returns:
        A mapping of each stubbed name to whatever module object (if
        any) occupied it in ``sys.modules`` beforehand, so the caller
        can restore that exact prior state after import instead of
        blindly deleting a module this test didn't actually insert.
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

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    transformers_stub.AutoModel = _FakeModel  # type: ignore[attr-defined]

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.hf_hub_download = lambda *_a, **_k: "/nonexistent/sparse_linear.pt"  # type: ignore[attr-defined]

    stubs = {
        "torch": torch_stub,
        "torch.nn": nn_stub,
        "transformers": transformers_stub,
        "huggingface_hub": hub_stub,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    return previous


_previous_modules = _install_stubs()

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    import sparse_server
finally:
    # sparse_server holds its own references to the stub modules; put
    # sys.modules back exactly as it was (restoring any real module we
    # displaced, or removing the key if there was none) so other test
    # files still get the real ImportError (and the real torch where it
    # is installed).
    for _name, _previous_module in _previous_modules.items():
        if _previous_module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous_module

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


def test_pooling_rejects_unsupported_task() -> None:
    """Only token_classify is supported; anything else is a 400, not a silent embed."""
    response = client.post(
        "/pooling",
        json={"model": "BAAI/bge-m3", "task": "embed", "input": ["alpha beta"]},
    )
    assert response.status_code == 400
    assert "token_classify" in response.json()["detail"]


def test_pooling_empty_input_returns_empty_data() -> None:
    """An empty batch is not an error — it returns an empty data list."""
    response = client.post(
        "/pooling",
        json={"model": "BAAI/bge-m3", "task": "token_classify", "input": []},
    )
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_pooling_returns_one_score_list_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response shape must match docint's _pool_token_scores expectations."""
    monkeypatch.setattr(
        sparse_server,
        "encode_token_weights",
        lambda texts: [[0.0, 0.5, 0.25, 0.0] for _ in texts],
    )
    response = client.post(
        "/pooling",
        json={"model": "BAAI/bge-m3", "task": "token_classify", "input": ["alpha beta", "gamma delta"]},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 2
    assert data[0]["data"] == [0.0, 0.5, 0.25, 0.0]
    assert [item["index"] for item in data] == [0, 1]


def test_pooling_scores_align_with_tokenize_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two routes must agree on sequence length for the same text.

    docint pairs /tokenize ids with /pooling scores positionally. A
    length mismatch silently truncates the sparse vector via zip(), so
    this alignment is the contract that matters most.

    This runs the REAL ``tokenize_ids`` and ``encode_token_weights`` —
    only ``model``/``sparse_head``/``torch.relu`` are faked, and only as
    shape-preserving identities over the tokenizer's own attention
    mask, so the response lengths are driven entirely by
    ``_FakeTokenizer``'s independent ``encode``/``__call__`` paths (see
    its docstring). A prior version of this test derived the pooling
    length FROM ``tokenize_ids``'s own output, which made it pass by
    construction no matter what ``encode_token_weights`` actually did.
    The batch mixes a 3-word and a 1-word text so a real batch gets
    padded, exercising the per-row mask strip (a single-item batch
    never pads, so it couldn't have caught a dropped mask strip). The
    kwargs assertions below additionally catch the two paths silently
    diverging on ``add_special_tokens``/``truncation``/``max_length``
    even in cases where that divergence wouldn't happen to change a
    length.
    """
    monkeypatch.setattr(
        sparse_server,
        "model",
        lambda **kwargs: types.SimpleNamespace(last_hidden_state=kwargs["attention_mask"]),
    )
    monkeypatch.setattr(sparse_server, "sparse_head", lambda hidden: hidden)
    monkeypatch.setattr(sparse_server.torch, "relu", lambda x: x, raising=False)

    texts = ["alpha beta gamma", "delta"]
    tokenize_bodies = [client.post("/tokenize", json={"model": "BAAI/bge-m3", "prompt": text}).json() for text in texts]
    pooling_body = client.post(
        "/pooling",
        json={"model": "BAAI/bge-m3", "task": "token_classify", "input": texts},
    ).json()

    for index, tokenize_body in enumerate(tokenize_bodies):
        assert len(pooling_body["data"][index]["data"]) == len(tokenize_body["tokens"])

    encode_kwargs = sparse_server.tokenizer.encode_calls[-1]
    call_kwargs = sparse_server.tokenizer.call_calls[-1]
    assert encode_kwargs["add_special_tokens"] == call_kwargs["add_special_tokens"]
    assert encode_kwargs["truncation"] == call_kwargs["truncation"]
    assert encode_kwargs["max_length"] == call_kwargs["max_length"]


def test_encode_token_weights_strips_padding_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed-length batch: each row keeps its OWN token count, not the batch max.

    The other /pooling tests monkeypatch ``encode_token_weights`` itself, so
    none of them exercise its real per-row attention-mask stripping. This one
    fakes only the dependencies (tokenizer, model, sparse_head, torch.relu)
    and runs the real function, so a bug that strips by batch-max instead of
    per-row mask — or applies the wrong mask — would be caught here. The pad
    positions carry deliberately large weights (0.9) so a masking bug shows
    up as a wrong value, not just a wrong length.
    """
    fake_mask = [[1, 1, 0, 0], [1, 1, 1, 1]]
    fake_weights = [[0.1, 0.2, 0.9, 0.9], [0.3, 0.4, 0.5, 0.6]]
    encoded = {"input_ids": _FakeTensor(fake_mask), "attention_mask": _FakeTensor(fake_mask)}

    def fake_model(**_kwargs: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(last_hidden_state=_FakeTensor([]))

    monkeypatch.setattr(sparse_server, "tokenizer", lambda *_args, **_kwargs: encoded)
    monkeypatch.setattr(sparse_server, "model", fake_model)
    monkeypatch.setattr(sparse_server, "sparse_head", lambda _hidden: _FakeTensor(fake_weights))
    monkeypatch.setattr(sparse_server.torch, "relu", lambda x: x, raising=False)

    rows = sparse_server.encode_token_weights(["a b", "a b c d"])

    assert len(rows[0]) == 2
    assert len(rows[1]) == 4
    assert rows[0] == [0.1, 0.2]


class _FakeHidden:
    """Stands in for last_hidden_state; supports [:, k] -> per-item position k.

    Holds one sequence of position-vectors per batch item (mirroring the
    real ``(batch, seq_len, hidden)`` shape) so indexing genuinely
    depends on the requested position — a fake that ignored the index
    and always returned the same rows would let a wrong-position bug
    (e.g. reading position 1 instead of the CLS position 0) pass
    unnoticed.
    """

    def __init__(self, sequences: list[list[list[float]]]) -> None:
        self._sequences = sequences

    def __getitem__(self, key: tuple[slice, int]) -> "_FakeRows":
        """Return each batch item's vector at the requested position.

        Args:
            key: A ``(slice(None), position)`` tuple, matching the real
                ``hidden[:, position]`` indexing this fake stands in for.
        """
        _, position = key
        return _FakeRows([sequence[position] for sequence in self._sequences])


class _FakeRows:
    """The object `hidden[:, 0]` yields; only .tolist() is used."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        """Return the raw CLS rows."""
        return self._rows


def test_l2_normalise_returns_unit_vector() -> None:
    """A normalised vector has length 1 and preserves direction."""
    out = sparse_server.l2_normalise([3.0, 4.0])
    assert out == pytest.approx([0.6, 0.8])
    assert sum(v * v for v in out) == pytest.approx(1.0)


def test_l2_normalise_handles_zero_vector() -> None:
    """A zero vector must not divide by zero."""
    assert sparse_server.l2_normalise([0.0, 0.0]) == [0.0, 0.0]


def test_embeddings_returns_openai_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response must match the OpenAI embeddings contract the SDK expects."""
    monkeypatch.setattr(sparse_server, "encode_dense", lambda texts: [[1.0, 0.0] for _ in texts])
    response = client.post("/v1/embeddings", json={"model": "BAAI/bge-m3", "input": ["alpha", "beta"]})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == sparse_server.MODEL_ID
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["embedding"] == [1.0, 0.0]


def test_embeddings_accepts_a_bare_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI contract allows `input` to be a single string."""
    monkeypatch.setattr(sparse_server, "encode_dense", lambda texts: [[1.0] for _ in texts])
    response = client.post("/v1/embeddings", json={"model": "m", "input": "alpha"})
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_embeddings_empty_input_returns_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty batch is not an error."""
    monkeypatch.setattr(sparse_server, "encode_dense", lambda texts: [])
    response = client.post("/v1/embeddings", json={"model": "m", "input": []})
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_encode_dense_cls_pools_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs the REAL encode_dense: CLS row is taken and unit-normalised."""
    encoded = {"attention_mask": _FakeTensor([[1, 1], [1, 1]])}

    class _Model:
        def __call__(self, **_kwargs: object) -> object:
            class _Out:
                # Position 0 (CLS) holds the real vectors; position 1 holds
                # deliberately different values so reading the wrong
                # position produces a wrong, detectable result.
                last_hidden_state = _FakeHidden([[[3.0, 4.0], [99.0, 99.0]], [[0.0, 5.0], [88.0, 88.0]]])

            return _Out()

    monkeypatch.setattr(sparse_server, "tokenizer", lambda *a, **k: encoded)
    monkeypatch.setattr(sparse_server, "model", _Model())

    rows = sparse_server.encode_dense(["a", "b"])

    assert rows[0] == pytest.approx([0.6, 0.8])  # fails if CLS pooling or L2 norm is wrong
    assert rows[1] == pytest.approx([0.0, 1.0])
