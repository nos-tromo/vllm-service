# Sparse-only CPU service (`vllm-service`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU `sparse-only` deployment shape to `vllm-service` that serves bge-m3 sparse embeddings over the same `POST /pooling` + `POST /tokenize` routes the full stack's LiteLLM router already passes through to `embed:8000`.

**Architecture:** One new FastAPI server (`src/sparse_server.py`) running XLM-R forward → `sparse_linear.pt` → ReLU on CPU, packaged in `Dockerfile.sparse.cpu` and deployed by `docker/compose.sparse-only.yaml`. Purely additive: `docker/compose.yaml` and `docker/litellm.config.yaml` are **not** touched, so the production sparse protocol and vector space stay frozen.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, transformers, CPU-only torch, Docker Compose. Lint/type via ruff + pyrefly (`uv run pre-commit run --all-files`). Tests via pytest in the torch-free `eval` dependency group.

**Design doc:** `docint/docs/2026-08-01-remote-sparse-encoder-design.md`

## Global Constraints

- Repo: `vllm-service`. Run all commands from that directory. Branch off `main` as `feat/sparse-only-service`.
- Python `>=3.11,<3.12`.
- **Do not modify `docker/compose.yaml` or `docker/litellm.config.yaml`.** Production collections depend on their current behaviour.
- **No FlagEmbedding.** It pulls `ir-datasets` → `zlib-state`, which fails to build on aarch64. Use `transformers` directly.
- CPU torch must come from `https://download.pytorch.org/whl/cpu`; the default PyPI wheel pulls a CUDA build on linux/amd64.
- Airgap-first: all model loading uses the local HF cache. Nothing fetches at runtime.
- No Bearer-token gate on the server — `inference-net` is the trust boundary, matching `rerank-only`, `gliner-only`, `clip-only`.
- Ruff: `line-length = 120`, google-convention docstrings required on every public function and class (`D` rules are enabled).
- pyrefly runs in `preset = "strict"`.
- The `eval` dependency group is **torch-free by design**. Unit tests must stub `torch` and `transformers` in `sys.modules`, following `eval/tests/test_clip_server.py`.

## File Structure

| File | Responsibility |
|---|---|
| `src/sparse_server.py` (create) | FastAPI app: `/pooling`, `/tokenize`, `/health`. Model load at import; two monkeypatchable seams for tests. |
| `eval/tests/test_sparse_server.py` (create) | Route-level unit tests against stubbed torch/transformers. |
| `docker/Dockerfile.sparse.cpu` (create) | CPU image, mirrors `Dockerfile.rerank.cpu`. |
| `docker/compose.sparse-only.yaml` (create) | Standalone compose project, mirrors `compose.rerank-only.yaml`. |
| `docker/compose.sparse-only.override.yaml` (create) | Dev overlay publishing the host port. |
| `Makefile` (modify) | `COMPOSE_SPARSE_ONLY` vars + six `*-sparse-only` targets. |
| `scripts/bundle_images.sh` (modify) | Add the `sparse-only` shape to the case statement and usage string. |
| `.env.example` (modify) | Document `SPARSE_MODEL`, `SPARSE_MAX_LENGTH`, `SPARSE_HOST_PORT`. |
| `README.md`, `CLAUDE.md` (modify) | Document the shape alongside the other `*-only` shapes. |

---

### Task 1: Sparse server — module skeleton, `/health`, `/tokenize`

**Files:**
- Create: `src/sparse_server.py`
- Create: `eval/tests/test_sparse_server.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `app: FastAPI`
  - `MODEL_ID: str`, `MAX_LENGTH: int`
  - `tokenize_ids(text: str) -> list[int]` — seam, monkeypatched by tests
  - `encode_token_weights(texts: list[str]) -> list[list[float]]` — seam, implemented in Task 2
  - `TokenizeRequest`, `TokenizeResponse` pydantic models

**Context an implementer needs:**

The consumer is docint's `VLLMSparseEncoder._extract_token_ids` (`docint/core/rag.py:1607-1642`). It probes the response dict for `token_ids`, then `tokens`, then `prompt_token_ids`, then recurses into `data`. Returning a `tokens` key satisfies it. Request body is `{"model": ..., "prompt": <str>}` (`rag.py:1594-1600`).

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_sparse_server.py`:

```python
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
    torch_stub.float32 = "float32"

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            return False

    torch_stub.no_grad = _NoGrad
    torch_stub.load = lambda *_a, **_k: {}
    nn_stub = types.ModuleType("torch.nn")

    class _Linear:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def load_state_dict(self, *_a: object, **_k: object) -> None:
            return None

        def train(self, mode: bool = True) -> "_Linear":
            return self

    nn_stub.Linear = _Linear
    torch_stub.nn = nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = _FakeTokenizer
    transformers_stub.AutoModel = _FakeModel
    sys.modules["transformers"] = transformers_stub

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.hf_hub_download = lambda *_a, **_k: "/nonexistent/sparse_linear.pt"

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group eval pytest eval/tests/test_sparse_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sparse_server'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sparse_server.py`:

```python
"""CPU sparse-embedding server for the sparse-only deployment shape.

Serves bge-m3's learned sparse (lexical) weights over the same two
routes the full-stack LiteLLM router passes through to the vLLM
``embed`` backend — ``POST /pooling`` with ``task: "token_classify"``
and ``POST /tokenize`` — so docint's ``RemoteSparseEncoder`` targets
either backend without protocol changes.

The full stack runs bge-m3 under vLLM's ``BgeM3EmbeddingModel``
architecture (see ``docker/compose.yaml``'s ``--hf-overrides``). This
server reproduces that model's ``token_classify`` output on CPU with
``transformers``: XLM-R forward, then the repo's ``sparse_linear.pt``
head (``Linear(hidden_size, 1)``), then ReLU. Special and padding
tokens fall out downstream — the consumer drops non-positive scores.

Inference uses ``transformers`` directly rather than ``FlagEmbedding``;
the latter pulls ``ir-datasets`` -> ``zlib-state``, which needs
``zlib.h`` to build from source on aarch64 and is irrelevant to
inference-only deployments.
"""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

MODEL_ID = os.environ.get("SPARSE_MODEL", "BAAI/bge-m3")
MAX_LENGTH = int(os.environ.get("SPARSE_MAX_LENGTH", "8192"))
SPARSE_LINEAR_FILE = os.environ.get("SPARSE_LINEAR_FILE", "sparse_linear.pt")

app = FastAPI(title="vllm-service sparse-only", version="1.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.train(False)


def _load_sparse_head(hidden_size: int) -> object:
    """Load bge-m3's sparse projection head from the local HF cache.

    ``weights_only=True`` is load-bearing, not decoration: the default
    (``False``) unpickles arbitrary Python objects, so a tampered
    checkpoint in the shared cache volume would execute code at server
    start. The file holds nothing but tensors, so the restricted loader
    is sufficient.

    Args:
        hidden_size: Encoder hidden width, the head's input dimension.

    Returns:
        A ``Linear(hidden_size, 1)`` module in eval mode.
    """
    weights_path = hf_hub_download(repo_id=MODEL_ID, filename=SPARSE_LINEAR_FILE)
    head = torch.nn.Linear(hidden_size, 1)
    head.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    head.train(False)
    return head


sparse_head = _load_sparse_head(model.config.hidden_size)


class TokenizeRequest(BaseModel):
    """vLLM-shape tokenize request body."""

    model: str | None = None
    prompt: str


class TokenizeResponse(BaseModel):
    """vLLM-shape tokenize response body."""

    count: int
    max_model_len: int
    tokens: list[int]


def tokenize_ids(text: str) -> list[int]:
    """Encode *text* to token ids including special tokens.

    Both routes go through this seam so ``/tokenize`` and ``/pooling``
    can never disagree on sequence length.

    Args:
        text: Input text.

    Returns:
        Token ids, truncated to ``MAX_LENGTH``.
    """
    return list(tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH))


@app.post("/tokenize")
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    """Tokenize a single prompt.

    Args:
        req: Request carrying the prompt.

    Returns:
        The token ids under a ``tokens`` key.

    Raises:
        HTTPException: 400 when the prompt is empty.
    """
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must be a non-empty string")
    token_ids = tokenize_ids(req.prompt)
    return TokenizeResponse(count=len(token_ids), max_model_len=MAX_LENGTH, tokens=token_ids)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID}
```

Note: `Field` is imported for Task 2's `PoolingRequest`; if ruff flags it as unused now, add the import in Task 2 instead.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group eval pytest eval/tests/test_sparse_server.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sparse_server.py eval/tests/test_sparse_server.py
git commit -m "feat(sparse): CPU sparse server skeleton with /health and /tokenize"
```

---

### Task 2: `/pooling` with `task: token_classify`

**Files:**
- Modify: `src/sparse_server.py`
- Modify: `eval/tests/test_sparse_server.py`

**Interfaces:**
- Consumes: `app`, `MODEL_ID`, `MAX_LENGTH`, `tokenize_ids` from Task 1.
- Produces: `encode_token_weights(texts: list[str]) -> list[list[float]]`, `PoolingRequest`, `PoolingItem`, `PoolingResponse`.

**Context an implementer needs:**

docint's `VLLMSparseEncoder._pool_token_scores` (`docint/core/rag.py:1563-1583`) POSTs `{"model": ..., "task": "token_classify", "input": [str, ...]}` and requires:
- top-level `data` to be a list, one entry per input, same order;
- each entry's `data` to be a list of per-token scores.

It then pairs those scores with `/tokenize`'s ids positionally in `_build_sparse_vector` (`rag.py:1678-1711`), which drops any score `<= 0` or non-finite and merges duplicate ids by max. So special/padding tokens need no special handling here — ReLU zeroes them and the consumer filters them out. **Per-text score lists must exclude padding**, or the positional pairing breaks for every text shorter than the batch max.

- [ ] **Step 1: Write the failing test**

Append to `eval/tests/test_sparse_server.py`:

```python
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
    length mismatch silently truncates the sparse vector via zip(),
    so this alignment is the contract that matters most.
    """
    monkeypatch.setattr(
        sparse_server,
        "encode_token_weights",
        lambda texts: [[1.0] * len(sparse_server.tokenize_ids(text)) for text in texts],
    )
    text = "alpha beta gamma"
    tokenize_body = client.post("/tokenize", json={"model": "BAAI/bge-m3", "prompt": text}).json()
    pooling_body = client.post(
        "/pooling",
        json={"model": "BAAI/bge-m3", "task": "token_classify", "input": [text]},
    ).json()
    assert len(pooling_body["data"][0]["data"]) == len(tokenize_body["tokens"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group eval pytest eval/tests/test_sparse_server.py -v`
Expected: FAIL — 404 on `/pooling` (route not registered)

- [ ] **Step 3: Write minimal implementation**

Append to `src/sparse_server.py`:

```python
class PoolingRequest(BaseModel):
    """vLLM-shape pooling request body."""

    model: str | None = None
    task: str = "token_classify"
    input: list[str] = Field(default_factory=list)


class PoolingItem(BaseModel):
    """One input's per-token score list."""

    index: int
    data: list[float]


class PoolingResponse(BaseModel):
    """vLLM-shape pooling response body."""

    model: str
    data: list[PoolingItem]


def encode_token_weights(texts: list[str]) -> list[list[float]]:
    """Compute bge-m3 sparse weights for each text.

    Runs the encoder forward, applies the sparse head and ReLU, then
    strips padding positions using the attention mask so each returned
    list aligns one-to-one with that text's own token ids.

    Args:
        texts: Input texts.

    Returns:
        One list of per-token weights per input, in input order.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(**encoded, return_dict=True).last_hidden_state
        weights = torch.relu(sparse_head(hidden)).squeeze(-1)

    rows: list[list[float]] = []
    for row, mask in zip(weights.tolist(), encoded["attention_mask"].tolist(), strict=False):
        rows.append([float(weight) for weight, keep in zip(row, mask, strict=False) if keep == 1])
    return rows


@app.post("/pooling")
def pooling(req: PoolingRequest) -> PoolingResponse:
    """Return per-token sparse weights for each input.

    Args:
        req: Pooling request carrying the task and the input batch.

    Returns:
        One ``PoolingItem`` per input, in input order.

    Raises:
        HTTPException: 400 when ``task`` is anything but ``token_classify``.
    """
    if req.task != "token_classify":
        raise HTTPException(
            status_code=400,
            detail=f"unsupported task {req.task!r}; this server implements only 'token_classify'",
        )
    if not req.input:
        return PoolingResponse(model=MODEL_ID, data=[])

    rows = encode_token_weights(req.input)
    return PoolingResponse(
        model=MODEL_ID,
        data=[PoolingItem(index=i, data=row) for i, row in enumerate(rows)],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group eval pytest eval/tests/test_sparse_server.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint and type-check**

Run: `uv run pre-commit run --all-files`
Expected: ruff check, ruff format, pyrefly all pass. Fix any docstring (`D`) or annotation (`ANN`) findings before committing.

- [ ] **Step 6: Commit**

```bash
git add src/sparse_server.py eval/tests/test_sparse_server.py
git commit -m "feat(sparse): serve bge-m3 token_classify weights on /pooling"
```

---

### Task 3: Container image and compose shape

**Files:**
- Create: `docker/Dockerfile.sparse.cpu`
- Create: `docker/compose.sparse-only.yaml`
- Create: `docker/compose.sparse-only.override.yaml`

**Interfaces:**
- Consumes: `src/sparse_server.py` from Tasks 1-2.
- Produces: network alias `sparse-only` on `inference-net`, port 8000 in-container; image `vllm-service-sparse-only`.

- [ ] **Step 1: Write the Dockerfile**

Create `docker/Dockerfile.sparse.cpu`:

```dockerfile
ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.11-bookworm-slim@sha256:4f5d923c9dcea037f57bda425dd209f3ec643da2f0b74227f68d09dab0b3bb36

FROM ${UV_IMAGE}

# curl is needed for the docker-compose healthcheck against /health.
# build-essential covers transitive deps that build from source on
# platforms without a prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch from the dedicated index. Same approach as
# Dockerfile.rerank.cpu — the default PyPI wheel pulls a CUDA build on
# linux/amd64, pinning to the cpu index avoids that.
RUN uv pip install --system --no-cache --compile-bytecode \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Sparse inference uses transformers directly rather than FlagEmbedding.
# FlagEmbedding pulls ir-datasets → zlib-state, which needs zlib.h to
# build from source on aarch64 and is irrelevant to inference-only
# deployments. huggingface_hub is needed to resolve sparse_linear.pt
# out of the mounted cache.
RUN uv pip install --system --no-cache --compile-bytecode \
    'fastapi>=0.115.0' \
    'uvicorn[standard]>=0.32.0' \
    'pydantic>=2.0' \
    'transformers>=4.44.2' \
    'huggingface_hub>=0.26.0' \
    'sentencepiece>=0.2.0'

COPY src/sparse_server.py /app/sparse_server.py

ENV TOKENIZERS_PARALLELISM=true

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "/app", "sparse_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write the base compose file**

Create `docker/compose.sparse-only.yaml`:

```yaml
name: vllm-service-sparse-only

# Standalone CPU sparse-embedding stack. Not a profile of compose.yaml —
# a separate compose project for hosts that cannot run the full CUDA
# stack (macOS, CPU-only Linux, AMD/ROCm boxes running Ollama for
# chat/embed).
#
# Topology:
#   - one container, `sparse-only`, on the external `inference-net`
#   - no LiteLLM router, no vLLM backends, no GPU reservation
#   - the full stack's pass-through URLs `http://vllm-router:4000/pooling`
#     and `/tokenize` are unavailable in this shape; consumers point
#     SPARSE_API_BASE at `http://sparse-only:8000`
#
# Auth: the container has no built-in Bearer-token gate. `inference-net`
# is a private Docker network shared only between trusted compose
# projects, mirroring how the `rerank-only` and `gliner-only` services
# expose /rerank and /gliner.

x-no-proxy-env: &no-proxy-env
  NO_PROXY: "sparse-only,localhost,127.0.0.1,172.16.0.0/12,10.0.0.0/8${EXTRA_NO_PROXY:-}"
  no_proxy: "sparse-only,localhost,127.0.0.1,172.16.0.0/12,10.0.0.0/8${EXTRA_NO_PROXY:-}"

x-env-file: &env-file
  - path: ../.env
    required: false

# Shared logging — the local driver rotates and compresses
# per-container; tail with `docker compose logs -f <svc>`.
x-logging: &default-logging
  driver: "local"
  options:
    max-size: "50m"
    max-file: "5"
    compress: "true"

services:
  sparse-only:
    image: vllm-service-sparse-only:${VLLM_SERVICE_VERSION:-latest}
    build:
      context: ..
      dockerfile: docker/Dockerfile.sparse.cpu
    logging: *default-logging
    env_file: *env-file
    environment:
      <<: *no-proxy-env
      SPARSE_MODEL: ${SPARSE_MODEL:-BAAI/bge-m3}
      SPARSE_MAX_LENGTH: ${SPARSE_MAX_LENGTH:-8192}
      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-1}
      TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-1}
    volumes:
      - huggingface-cache:/root/.cache/huggingface/hub
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 60
      start_period: 180s
    networks:
      inference-net:
        aliases:
          - sparse-only
    restart: unless-stopped

volumes:
  huggingface-cache:
    external: true

networks:
  inference-net:
    external: true
    name: ${INFERENCE_NETWORK:-inference-net}
```

- [ ] **Step 3: Write the dev override**

Create `docker/compose.sparse-only.override.yaml`:

```yaml
### Dev override for the sparse-only stack — publishes the sparse service
### on the host.
###
### NOT auto-loaded: it lives in docker/ and the Makefile passes explicit
### -f flags. `make up-dev-sparse-only` layers this on top of
### docker/compose.sparse-only.yaml so the service is reachable at
### localhost:${SPARSE_HOST_PORT} for direct testing; `make up-sparse-only`
### runs the base alone (production shape). Sibling containers reach the
### service as `sparse-only:8000` on inference-net regardless of this override.

services:
  sparse-only:
    ports:
      - "${SPARSE_HOST_PORT:-8005}:8000"
```

Before committing, confirm `8005` is not already claimed: `grep -rn "HOST_PORT" docker/*.override.yaml .env.example`. If taken, pick the next free port and use it consistently in Task 5's `.env.example` entry.

- [ ] **Step 4: Verify the compose files parse**

Run: `docker compose --env-file .env -f docker/compose.sparse-only.yaml config >/dev/null && echo OK`
Expected: `OK` (no schema errors). This does not build or start anything.

- [ ] **Step 5: Build the image**

Run: `docker compose --env-file .env -f docker/compose.sparse-only.yaml build`
Expected: image `vllm-service-sparse-only` builds. This is slow (CPU torch is a large wheel).

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile.sparse.cpu docker/compose.sparse-only.yaml docker/compose.sparse-only.override.yaml
git commit -m "feat(sparse): CPU image and sparse-only compose shape"
```

---

### Task 4: Makefile targets and bundle shape

**Files:**
- Modify: `Makefile` (compose vars near line 72; `.PHONY` block near line 53-58; a new target section after the rerank-only block at lines 183-201)
- Modify: `scripts/bundle_images.sh` (case statement at line 14, usage string at line 19)

**Interfaces:**
- Consumes: `docker/compose.sparse-only.yaml` and its override from Task 3.
- Produces: `make {build,bundle,up,up-dev,stop,down}-sparse-only`.

- [ ] **Step 1: Add the compose variables**

In `Makefile`, after the `COMPOSE_RERANK_ONLY*` lines (72-73), add:

```make
COMPOSE_SPARSE_ONLY     := docker compose --env-file .env -f docker/compose.sparse-only.yaml
COMPOSE_SPARSE_ONLY_DEV := docker compose --env-file .env -f docker/compose.sparse-only.yaml -f docker/compose.sparse-only.override.yaml
```

- [ ] **Step 2: Add the targets**

Append a new section after the rerank-only block:

```make
# --- Sparse-only stack --------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites. Serves bge-m3 sparse weights on the same /pooling +
# /tokenize routes the full stack's router passes through, so docint
# only has to repoint SPARSE_API_BASE.
build-sparse-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_SPARSE_ONLY) build

bundle-sparse-only:
	./scripts/bundle_images.sh sparse-only

up-sparse-only:
	$(COMPOSE_SPARSE_ONLY) up -d --no-build

# Like 'up-sparse-only' but publishes the sparse port on the host.
up-dev-sparse-only:
	$(COMPOSE_SPARSE_ONLY_DEV) up -d --no-build

stop-sparse-only:
	$(COMPOSE_SPARSE_ONLY) stop

# Stop + remove the sparse service. External huggingface-cache survives.
down-sparse-only:
	$(COMPOSE_SPARSE_ONLY) down
```

- [ ] **Step 3: Register the targets as phony**

In the `.PHONY` block (lines 53-58), add a line matching the existing style:

```make
        build-sparse-only bundle-sparse-only up-sparse-only up-dev-sparse-only stop-sparse-only down-sparse-only \
```

Also add a numbered entry to the header comment block (lines 9-48) describing the shape, matching the style of entries 2-7.

- [ ] **Step 4: Add the bundle shape**

In `scripts/bundle_images.sh`, add to the case statement alongside line 14:

```sh
  sparse-only)  PROFILE_LABEL="sparse-only";  COMPOSE_FILE="docker/compose.sparse-only.yaml" ;;
```

And extend the usage string at line 19 to include `sparse-only`.

- [ ] **Step 5: Verify the targets resolve**

Run: `make -n up-sparse-only && make -n build-sparse-only && make -n bundle-sparse-only`
Expected: each prints the command it would run, with no "No rule to make target" error.

- [ ] **Step 6: Commit**

```bash
git add Makefile scripts/bundle_images.sh
git commit -m "build(sparse): make targets and bundle shape for sparse-only"
```

---

### Task 5: Documentation and env example

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Document the env knobs**

In `.env.example`, next to the existing `RERANK_*` block, add:

```bash
# --- sparse-only shape ---------------------------------------------------
# CPU bge-m3 sparse-embedding service. Serves /pooling (task=token_classify)
# and /tokenize — the same routes the full stack's router passes through to
# the vLLM embed backend — so docint points SPARSE_API_BASE at either.
SPARSE_MODEL=BAAI/bge-m3
SPARSE_MAX_LENGTH=8192
# Host port published only by `make up-dev-sparse-only`.
SPARSE_HOST_PORT=8005
```

Use the same port value chosen in Task 3 Step 3.

- [ ] **Step 2: Document the shape in README.md**

Find the section listing the `*-only` shapes and add a `sparse-only` entry in the same format, covering: what it serves, that it pairs with `gliner-only`/`rerank-only`/`clip-only` on a CPU host, its `make` targets, and the consumer setting `SPARSE_API_BASE=http://sparse-only:8000`. State explicitly that the full stack serves the same routes via the router pass-throughs, so consumers change only the base URL.

- [ ] **Step 3: Document the shape in CLAUDE.md**

Add `sparse-only` wherever the other `*-only` shapes are enumerated. Note the two supported routes and that `task` must be `token_classify`.

- [ ] **Step 4: Verify**

Run: `uv run pre-commit run --all-files`
Expected: pass.

- [ ] **Step 5: Commit and open the PR**

```bash
git add .env.example README.md CLAUDE.md
git commit -m "docs(sparse): document the sparse-only deployment shape"
git push -u origin feat/sparse-only-service
gh pr create --fill
```

---

## Deferred to plan 2

The parity test — golden token-ids-and-weights fixtures captured from the real vLLM `embed` backend and asserted against this CPU server — is **not** in this plan. It needs a host with the CUDA stack to generate fixtures, which is not available where this work is being done. Until it exists, this server's fidelity to `BgeM3EmbeddingModel` is argued from the architecture, not measured. Track it as a follow-up issue on `vllm-service` and reference it from the docint PR.

## Self-review notes

- Spec coverage: server routes (Tasks 1-2), Dockerfile (3), compose + override (3), Makefile + bundle (4), docs (5). The spec's `compose.yaml`/`litellm.config.yaml` freeze is enforced by omission and stated in Global Constraints. The parity test is explicitly deferred above rather than silently dropped.
- Type consistency: `tokenize_ids` and `encode_token_weights` are named identically in the implementation, the tests that monkeypatch them, and the Interfaces blocks.
- The `Field` import lands in Task 1 but is first used in Task 2; flagged inline at Task 1 Step 3.
