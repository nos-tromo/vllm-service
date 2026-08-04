"""Route-level tests for the embed-only server.

The heavy ML deps (torch, transformers) are not installed in this env
(see pyproject: the eval group is torch-free by design), so they are
stubbed in ``sys.modules`` before ``embed_server`` is imported; the
tests then monkeypatch the tokenize/inference seams. If the real
``transformers`` is importable (the eval-run env), importing
``embed_server`` would try to load the actual checkpoint, so the module
is skipped there.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

if importlib.util.find_spec("transformers") is not None:
    pytest.skip(
        "embed_server unit tests need the torch-free env (real transformers would load the checkpoint at import)",
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

    bos_token_id = 0
    eos_token_id = 2

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
            {
                "texts": list(texts),
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "max_length": max_length,
            }
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

    load_calls: ClassVar[list[dict[str, object]]] = []

    @staticmethod
    def from_pretrained(*_args: object, **_kwargs: object) -> "_FakeModel":
        """Return the fake instance, recording the kwargs for assertions."""
        _FakeModel.load_calls.append(dict(_kwargs))
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


class _StubTokenizer:
    """Tokenizer fake whose batch call returns one fixed encoding.

    For tests that drive a real encoder against a hand-built attention
    mask. It covers BOTH tokenizer entry points the encoders use —
    ``encode`` (via ``tokenize_ids``, which ``_iter_batches`` calls to
    size sub-batches) and ``__call__`` (via ``_encode_batch``) — because
    a fake covering only one of them fails on the fixture rather than on
    the behaviour under test.
    """

    bos_token_id = 0
    eos_token_id = 2

    def __init__(self, encoded: dict[str, _FakeTensor]) -> None:
        """Wrap the batch encoding this fake always returns.

        Args:
            encoded: The ``input_ids``/``attention_mask`` mapping handed
                back for every batch call.
        """
        self._encoded = encoded

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        """Return one id per whitespace token, wrapped in BOS/EOS."""
        return [0, *(100 + len(word) for word in text.split()), 2]

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, _FakeTensor]:
        """Return the fixed encoding regardless of the batch."""
        return self._encoded


def _install_stubs() -> dict[str, types.ModuleType | None]:
    """Stub torch + transformers + huggingface_hub in sys.modules.

    All four names are assigned directly, never via ``setdefault``. An
    import-order-dependent ``setdefault`` is exactly what let this bite
    us: if some earlier-collected test file already imported
    ``huggingface_hub`` for real, ``setdefault`` would leave that real
    module in place, ``embed_server`` would bind the real
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
    import embed_server
finally:
    # embed_server holds its own references to the stub modules; put
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

client = TestClient(embed_server.app, raise_server_exceptions=False)


def test_model_loaded_with_explicit_float32_dtype() -> None:
    """The checkpoint load must state torch_dtype rather than inherit it.

    transformers' float32 default is exactly what a major version bump
    changes (see issue #76); dense CLS+L2 vectors drift silently across
    all 1024 dims if the dtype moves. The stub records the kwargs of the
    import-time ``AutoModel.from_pretrained`` call.
    """
    assert len(_FakeModel.load_calls) == 1
    assert _FakeModel.load_calls[0].get("torch_dtype") == embed_server.torch.float32


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
        embed_server,
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
    """/pooling returns exactly len(/tokenize) - 2 scores per text.

    vLLM's ``BgeM3EmbeddingModel`` wraps its ``token_classify`` pooler in
    ``BOSEOSFilter`` (verified on the CUDA stack for issue #75), so its
    sparse rows exclude the BOS and EOS positions. This server must match
    that arity or a consumer switching backends by base URL alone gets a
    different pairing of ids to scores.

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
        embed_server,
        "model",
        lambda **kwargs: types.SimpleNamespace(last_hidden_state=kwargs["attention_mask"]),
    )
    monkeypatch.setattr(embed_server, "sparse_head", lambda hidden: hidden)
    monkeypatch.setattr(embed_server.torch, "relu", lambda x: x, raising=False)

    texts = ["alpha beta gamma", "delta"]
    tokenize_bodies = [client.post("/tokenize", json={"model": "BAAI/bge-m3", "prompt": text}).json() for text in texts]
    pooling_body = client.post(
        "/pooling",
        json={"model": "BAAI/bge-m3", "task": "token_classify", "input": texts},
    ).json()

    for index, tokenize_body in enumerate(tokenize_bodies):
        assert len(pooling_body["data"][index]["data"]) == len(tokenize_body["tokens"]) - 2

    encode_kwargs = embed_server.tokenizer.encode_calls[-1]
    call_kwargs = embed_server.tokenizer.call_calls[-1]
    assert encode_kwargs["add_special_tokens"] == call_kwargs["add_special_tokens"]
    assert encode_kwargs["truncation"] == call_kwargs["truncation"]
    assert encode_kwargs["max_length"] == call_kwargs["max_length"]


def test_embeddings_and_pooling_tokenize_identically(monkeypatch: pytest.MonkeyPatch) -> None:
    """/v1/embeddings and /pooling must batch-tokenize the same text identically.

    ``encode_dense`` and ``encode_token_weights`` used to each carry their
    own verbatim copy of the tokenizer call before both were routed
    through the shared ``_encode_batch`` seam; a divergence there (e.g.
    one copy silently dropping ``add_special_tokens`` or shrinking
    ``max_length``) produces vectors that are still unit-length and the
    right dimension — silently wrong, not loudly broken. This drives
    both routes for the same text and compares the raw kwargs
    ``_FakeTokenizer.__call__`` was invoked with, so a reintroduced
    per-route copy that drifts on any of them fails here immediately.

    The two routes need different fake forward passes — ``encode_dense``
    indexes ``hidden[:, 0]`` (CLS) while ``encode_token_weights`` feeds
    the whole hidden state through the sparse head — so each is
    monkeypatched separately; only the captured tokenizer kwargs are
    compared.
    """
    text = "alpha beta gamma"

    monkeypatch.setattr(
        embed_server,
        "model",
        lambda **kwargs: types.SimpleNamespace(last_hidden_state=kwargs["attention_mask"]),
    )
    monkeypatch.setattr(embed_server, "sparse_head", lambda hidden: hidden)
    monkeypatch.setattr(embed_server.torch, "relu", lambda x: x, raising=False)
    client.post("/pooling", json={"model": "BAAI/bge-m3", "task": "token_classify", "input": [text]})
    pooling_kwargs = embed_server.tokenizer.call_calls[-1]

    monkeypatch.setattr(
        embed_server,
        "model",
        lambda **_kwargs: types.SimpleNamespace(last_hidden_state=_FakeHidden([[[0.0, 0.0]]])),
    )
    client.post("/v1/embeddings", json={"model": "BAAI/bge-m3", "input": [text]})
    dense_kwargs = embed_server.tokenizer.call_calls[-1]

    assert dense_kwargs["add_special_tokens"] == pooling_kwargs["add_special_tokens"]
    assert dense_kwargs["truncation"] == pooling_kwargs["truncation"]
    assert dense_kwargs["max_length"] == pooling_kwargs["max_length"]


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

    monkeypatch.setattr(embed_server, "tokenizer", _StubTokenizer(encoded))
    monkeypatch.setattr(embed_server, "model", fake_model)
    monkeypatch.setattr(embed_server, "sparse_head", lambda _hidden: _FakeTensor(fake_weights))
    monkeypatch.setattr(embed_server.torch, "relu", lambda x: x, raising=False)

    rows = embed_server.encode_token_weights(["a b", "a b c d"])

    assert len(rows[0]) == 2
    assert len(rows[1]) == 4
    assert rows[0] == [0.1, 0.2]


def test_encode_token_weights_strips_bos_eos_conditionally(monkeypatch: pytest.MonkeyPatch) -> None:
    """BOS/EOS positions are dropped only when the boundary ids actually match.

    Mirrors vLLM's ``BOSEOSFilter`` exactly: the first position goes iff
    its id is the BOS id, the last iff its id is the EOS id — not an
    unconditional slice. Row 0 carries both specials, row 1 neither, so
    an implementation that always slices ``[1:-1]`` fails on row 1 and
    one that never strips fails on row 0. The boundary weights are
    deliberately large: ReLU does NOT zero them (measured 0.11-0.24 on
    the real model), which is why the strip must be positional.
    """
    input_ids = [[0, 7, 8, 2], [5, 6, 7, 9]]
    mask = [[1, 1, 1, 1], [1, 1, 1, 1]]
    fake_weights = [[0.9, 0.1, 0.2, 0.8], [0.3, 0.4, 0.5, 0.6]]
    encoded = {"input_ids": _FakeTensor(input_ids), "attention_mask": _FakeTensor(mask)}

    monkeypatch.setattr(embed_server, "tokenizer", _StubTokenizer(encoded))
    monkeypatch.setattr(
        embed_server, "model", lambda **_kwargs: types.SimpleNamespace(last_hidden_state=_FakeTensor([]))
    )
    monkeypatch.setattr(embed_server, "sparse_head", lambda _hidden: _FakeTensor(fake_weights))
    monkeypatch.setattr(embed_server.torch, "relu", lambda x: x, raising=False)

    rows = embed_server.encode_token_weights(["a b c d", "e f g h"])

    assert rows[0] == [0.1, 0.2]
    assert rows[1] == [0.3, 0.4, 0.5, 0.6]


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
    out = embed_server.l2_normalise([3.0, 4.0])
    assert out == pytest.approx([0.6, 0.8])
    assert sum(v * v for v in out) == pytest.approx(1.0)


def test_l2_normalise_handles_zero_vector() -> None:
    """A zero vector must not divide by zero."""
    assert embed_server.l2_normalise([0.0, 0.0]) == [0.0, 0.0]


def test_embeddings_returns_openai_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response must match the OpenAI embeddings contract the SDK expects."""
    monkeypatch.setattr(embed_server, "encode_dense", lambda texts: ([[1.0, 0.0] for _ in texts], 0))
    response = client.post("/v1/embeddings", json={"model": "BAAI/bge-m3", "input": ["alpha", "beta"]})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == embed_server.MODEL_ID
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["embedding"] == [1.0, 0.0]


def test_embeddings_accepts_a_bare_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAI contract allows `input` to be a single string."""
    monkeypatch.setattr(embed_server, "encode_dense", lambda texts: ([[1.0] for _ in texts], 0))
    response = client.post("/v1/embeddings", json={"model": "m", "input": "alpha"})
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_embeddings_empty_input_returns_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty batch is not an error."""
    monkeypatch.setattr(embed_server, "encode_dense", lambda texts: ([], 0))
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

    monkeypatch.setattr(embed_server, "tokenizer", _StubTokenizer(encoded))
    monkeypatch.setattr(embed_server, "model", _Model())

    rows, token_total = embed_server.encode_dense(["a", "b"])

    assert rows[0] == pytest.approx([0.6, 0.8])  # fails if CLS pooling or L2 norm is wrong
    assert rows[1] == pytest.approx([0.0, 1.0])
    assert token_total == 6  # BOS + 1 word + EOS, per text


def test_positive_int_env_rejects_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero or negative knob is a misconfiguration, not a silent slow mode.

    ``EMBED_MAX_BATCH_TOKENS=0`` would degrade to one-text-per-forward
    batching — safe but slow, and invisible until someone profiles it.
    Failing at startup surfaces it in the container logs instead.
    """
    monkeypatch.setenv("EMBED_MAX_BATCH_TOKENS", "0")
    with pytest.raises(ValueError, match="EMBED_MAX_BATCH_TOKENS"):
        embed_server._positive_int_env("EMBED_MAX_BATCH_TOKENS", 16384)

    monkeypatch.setenv("EMBED_MAX_BATCH_TOKENS", "-5")
    with pytest.raises(ValueError, match="EMBED_MAX_BATCH_TOKENS"):
        embed_server._positive_int_env("EMBED_MAX_BATCH_TOKENS", 16384)


def test_positive_int_env_reads_value_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set variable wins; an unset one falls back to the default."""
    monkeypatch.setenv("EMBED_MAX_BATCH_TOKENS", "512")
    assert embed_server._positive_int_env("EMBED_MAX_BATCH_TOKENS", 16384) == 512

    monkeypatch.delenv("EMBED_MAX_BATCH_TOKENS", raising=False)
    assert embed_server._positive_int_env("EMBED_MAX_BATCH_TOKENS", 16384) == 16384


def test_embeddings_tokenizes_each_text_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """/v1/embeddings must not re-tokenize for the usage count.

    ``_iter_batches`` already tokenizes every text to size the
    sub-batches; the route's ``usage`` sum must reuse those lengths
    rather than running ``tokenize_ids`` a second time per text. This
    drives the real route + ``encode_dense`` + ``_iter_batches`` chain
    and counts ``tokenize_ids``-path calls on the fake tokenizer, so a
    reintroduced second pass fails on the call count; the usage
    assertion pins that the reused lengths are the same ones
    ``/tokenize`` would report (BOS + words + EOS per text: 4 + 3).
    """

    def fake_model(**_kwargs: object) -> types.SimpleNamespace:
        batch = embed_server.tokenizer.call_calls[-1]["texts"]
        return types.SimpleNamespace(last_hidden_state=_FakeHidden([[[1.0, 0.0]] for _ in batch]))

    monkeypatch.setattr(embed_server, "model", fake_model)

    before = len(embed_server.tokenizer.encode_calls)
    response = client.post("/v1/embeddings", json={"model": "BAAI/bge-m3", "input": ["alpha beta", "gamma"]})

    assert response.status_code == 200
    assert len(embed_server.tokenizer.encode_calls) - before == 2
    assert response.json()["usage"] == {"prompt_tokens": 7, "total_tokens": 7}


def test_plan_batches_keeps_one_batch_when_within_budget() -> None:
    """A batch whose padded cost fits the budget is not split."""
    assert embed_server.plan_batches([3, 3, 2], budget=9) == [(0, 3)]


def test_plan_batches_charges_padding_not_the_raw_token_sum() -> None:
    """Cost is rows x longest row, because padding=True pads to the batch max.

    This is the whole defect: the raw token sum of [5, 1, 1, 1] is 8 and
    fits a budget of 8, but batching them together pads all four rows to
    5 and actually pushes 20 tokens through the encoder. An implementation
    that budgeted against the raw sum would return a single span here and
    still blow the bound it claims to enforce.
    """
    assert embed_server.plan_batches([5, 1, 1, 1], budget=8) == [(0, 1), (1, 4)]


def test_plan_batches_isolates_a_text_that_alone_exceeds_the_budget() -> None:
    """An over-budget text gets its own batch rather than an empty one.

    ``EMBED_MAX_BATCH_TOKENS`` can be configured below ``EMBED_MAX_LENGTH``,
    so a single text can exceed the budget on its own. It must still be
    encoded (truncation to ``MAX_LENGTH`` is the only cap that drops
    content) — never dropped, and never emitted as a zero-width span that
    would tokenize an empty list.
    """
    assert embed_server.plan_batches([20, 1], budget=8) == [(0, 1), (1, 2)]


def test_plan_batches_covers_every_input_exactly_once_in_order() -> None:
    """Spans must be contiguous, non-empty, and cover the whole input.

    Both routes return one row per input positionally, so a gap, an
    overlap, or a reordering here corrupts the response alignment.
    """
    lengths = [4, 1, 9, 2, 2, 7, 1]
    spans = embed_server.plan_batches(lengths, budget=10)

    assert [index for start, end in spans for index in range(start, end)] == list(range(len(lengths)))
    assert all(start < end for start, end in spans)


def test_plan_batches_handles_empty_input() -> None:
    """No inputs means no forward passes."""
    assert embed_server.plan_batches([], budget=8) == []


def test_encode_token_weights_bounds_sub_batches_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """/pooling runs several bounded forwards, concatenated in input order.

    Runs the REAL ``encode_token_weights`` over a budget small enough to
    force three sub-batches. Asserting only on row lengths would not
    catch a missing split — the per-row mask strip yields the same
    lengths whether the texts went through one padded forward or three —
    so this asserts the actual partition handed to the tokenizer (i.e.
    what each forward pass really cost) as well as the concatenated
    output order.
    """
    monkeypatch.setattr(embed_server, "MAX_BATCH_TOKENS", 6)
    monkeypatch.setattr(
        embed_server,
        "model",
        lambda **kwargs: types.SimpleNamespace(last_hidden_state=kwargs["attention_mask"]),
    )
    monkeypatch.setattr(embed_server, "sparse_head", lambda hidden: hidden)
    monkeypatch.setattr(embed_server.torch, "relu", lambda x: x, raising=False)

    # _FakeTokenizer: 1 id per word + BOS/EOS -> lengths 5, 3, 4.
    texts = ["alpha beta gamma", "delta", "epsilon zeta"]
    before = len(embed_server.tokenizer.call_calls)
    rows = embed_server.encode_token_weights(texts)
    batched = [call["texts"] for call in embed_server.tokenizer.call_calls[before:]]

    assert batched == [["alpha beta gamma"], ["delta"], ["epsilon zeta"]]
    assert [len(row) for row in rows] == [3, 1, 2]  # word count: BOS/EOS are stripped


def test_encode_dense_bounds_sub_batches_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """/v1/embeddings splits on the same budget and keeps input order.

    The dense route pads identically, so it needs the same bound. Each
    sub-batch's fake CLS rows are distinct, so a concatenation that lost
    order (or dropped a sub-batch) changes the returned vectors, not just
    their count.
    """
    monkeypatch.setattr(embed_server, "MAX_BATCH_TOKENS", 6)

    cls_by_text = {"alpha beta gamma": [3.0, 4.0], "delta": [0.0, 5.0], "epsilon zeta": [5.0, 0.0]}

    def fake_model(**_kwargs: object) -> types.SimpleNamespace:
        batch = embed_server.tokenizer.call_calls[-1]["texts"]
        return types.SimpleNamespace(
            last_hidden_state=_FakeHidden([[cls_by_text[text], [99.0, 99.0]] for text in batch])
        )

    monkeypatch.setattr(embed_server, "model", fake_model)

    texts = ["alpha beta gamma", "delta", "epsilon zeta"]
    before = len(embed_server.tokenizer.call_calls)
    rows, token_total = embed_server.encode_dense(texts)
    batched = [call["texts"] for call in embed_server.tokenizer.call_calls[before:]]

    assert batched == [["alpha beta gamma"], ["delta"], ["epsilon zeta"]]
    assert rows[0] == pytest.approx([0.6, 0.8])
    assert rows[1] == pytest.approx([0.0, 1.0])
    assert rows[2] == pytest.approx([1.0, 0.0])
    assert token_total == 12  # 5 + 3 + 4, summed across sub-batches
