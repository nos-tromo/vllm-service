# `embed-only`: dense route + rename (`vllm-service`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CPU container serve dense embeddings from the bge-m3 instance it already loads, and rename the shape `sparse-only` → `embed-only` to match.

**Architecture:** `src/sparse_server.py` gains an OpenAI-compatible `POST /v1/embeddings` computing CLS-pooled, L2-normalised dense vectors off the same `last_hidden_state` the sparse head already consumes. Then a pure mechanical rename across the shape's files, aliases, image name, make targets and docs. `docker/compose.yaml` and `docker/litellm.config.yaml` remain untouched.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, transformers, CPU torch, Docker Compose. ruff + pyrefly via `pre-commit`; pytest in the torch-free `eval` group.

**Design doc:** `docint/docs/2026-08-02-embed-only-dense-consolidation-design.md`

**This AMENDS an open, already-reviewed PR** — `nos-tromo/vllm-service#78`, branch `feat/sparse-only-service`, currently at `cfa4044`, 7 commits, MERGEABLE. Do not open a new PR.

## Global Constraints

- Repo: `vllm-service`. **The branch is checked out in that repo's main checkout** (the earlier worktree was removed externally). Confirm `git status` is clean and HEAD is `cfa4044` before starting. Do not try to create a worktree on this branch — it is already checked out, and git will refuse.
- Python `>=3.11,<3.12`.
- **Do not modify `docker/compose.yaml` or `docker/litellm.config.yaml`.** Production depends on their current behaviour.
- No FlagEmbedding — it pulls `ir-datasets` → `zlib-state`, which fails to build on aarch64.
- CPU torch from `https://download.pytorch.org/whl/cpu` only.
- Airgap-first: no runtime fetches.
- No Bearer gate — `inference-net` is the trust boundary.
- Ruff `line-length = 120`, google-convention docstrings (`D` rules on). pyrefly `preset = "strict"`.
- The `eval` group is **torch-free**; tests stub `torch`/`transformers`/`huggingface_hub` at module level and remove them from `sys.modules` after import so later test files are unaffected.
- `/pooling` and `/tokenize` behaviour must not change — a downstream consumer's production collections depend on the vectors they produce.

## File Structure

| File | Change |
|---|---|
| `src/sparse_server.py` → `src/embed_server.py` | Task 1 adds the dense route; Task 2 renames the file |
| `eval/tests/test_sparse_server.py` → `eval/tests/test_embed_server.py` | Task 1 adds dense tests; Task 2 renames |
| `docker/Dockerfile.sparse.cpu` → `docker/Dockerfile.embed.cpu` | Task 2 |
| `docker/compose.sparse-only.yaml` → `docker/compose.embed-only.yaml` | Task 2 |
| `docker/compose.sparse-only.override.yaml` → `docker/compose.embed-only.override.yaml` | Task 2 |
| `Makefile` | Task 2 — compose vars, 6 targets, `.PHONY`, header comment |
| `scripts/bundle_images.sh` | Task 2 — case arm + usage string |
| `.env.example`, `README.md`, `CLAUDE.md` | Task 3 |

**Task order is deliberate:** the dense route lands first as a reviewable functional change, then the rename lands as a pure mechanical commit with no logic change. One mixed commit would be far harder to review.

---

### Task 1: Dense `/v1/embeddings` route

**Files:**
- Modify: `src/sparse_server.py`
- Modify: `eval/tests/test_sparse_server.py`

**Interfaces:**
- Consumes: existing `MODEL_ID`, `MAX_LENGTH`, `tokenizer`, `model`, `app`.
- Produces: `l2_normalise(vec: list[float]) -> list[float]`, `encode_dense(texts: list[str]) -> list[list[float]]`, `EmbeddingsRequest`, `EmbeddingItem`, `EmbeddingsUsage`, `EmbeddingsResponse`.

**Context an implementer needs:**

The consumer is llama-index's `OpenAIEmbedding` via the OpenAI SDK, which appends `/embeddings` to its configured base — hence the route lives at `/v1/embeddings` while the vLLM pooling protocol stays root-anchored at `/pooling` and `/tokenize`. Both must coexist.

bge-m3's dense vector is **CLS pooling** — `last_hidden_state[:, 0]` — then L2 normalised. That is FlagEmbedding's `cls` sentence-pooling method for this model. Do not mean-pool.

**Do the normalisation in plain Python, not torch.** Extracting `hidden[:, 0].tolist()` and normalising the resulting lists keeps the load-bearing arithmetic unit-testable in the torch-free `eval` group, where torch is a stub with no real ops. The cost is negligible against the forward pass.

The request's `model` field is **ignored** and the response echoes `MODEL_ID`, matching what `/pooling` and `/tokenize` already do. The container serves exactly one model.

- [ ] **Step 1: Write the failing tests**

Append to `eval/tests/test_sparse_server.py`. Note the existing module-level fakes; you will need a hidden-state fake supporting `[:, 0]`:

```python
class _FakeHidden:
    """Stands in for last_hidden_state; supports [:, 0] -> rows object."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def __getitem__(self, key: object) -> "_FakeRows":
        """Return the CLS column for a ``[:, 0]`` style index."""
        return _FakeRows(self._rows)


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
                last_hidden_state = _FakeHidden([[3.0, 4.0], [0.0, 5.0]])

            return _Out()

    monkeypatch.setattr(sparse_server, "tokenizer", lambda *a, **k: encoded)
    monkeypatch.setattr(sparse_server, "model", _Model())

    rows = sparse_server.encode_dense(["a", "b"])

    assert rows[0] == pytest.approx([0.6, 0.8])   # fails if CLS pooling or L2 norm is wrong
    assert rows[1] == pytest.approx([0.0, 1.0])
```

If `_FakeTensor` does not already exist in the file from earlier work, build the `attention_mask` fake with whatever shape the existing tests use — `encode_dense` only needs the tokenizer's return value to be passable to `model(**encoded)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group eval pytest eval/tests/test_sparse_server.py -v`
Expected: FAIL — `AttributeError: module 'sparse_server' has no attribute 'l2_normalise'`, and 404s on `/v1/embeddings`.

- [ ] **Step 3: Write the implementation**

Add to `src/sparse_server.py`:

```python
class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request body."""

    model: str | None = None
    input: str | list[str]


class EmbeddingItem(BaseModel):
    """One embedding in an OpenAI-shape response."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsUsage(BaseModel):
    """Token accounting for an embeddings response."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    """OpenAI-compatible embeddings response body."""

    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingsUsage


def l2_normalise(vec: list[float]) -> list[float]:
    """Scale a vector to unit length.

    Done in plain Python rather than torch so the arithmetic stays
    testable in the torch-free eval group; the cost is negligible
    beside the encoder forward pass.

    Args:
        vec: Raw vector components.

    Returns:
        The unit-length vector, or the input unchanged when its norm is
        zero (a zero vector has no direction to preserve).
    """
    norm = math.sqrt(sum(component * component for component in vec))
    if norm == 0.0:
        return list(vec)
    return [component / norm for component in vec]


def encode_dense(texts: list[str]) -> list[list[float]]:
    """Compute bge-m3 dense embeddings for each text.

    Dense is CLS pooling — the first token of ``last_hidden_state`` —
    followed by L2 normalisation, matching FlagEmbedding's ``cls``
    sentence-pooling method for this model.

    Args:
        texts: Input texts.

    Returns:
        One unit-length vector per input, in input order.
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
    cls_rows = hidden[:, 0].tolist()
    return [l2_normalise([float(component) for component in row]) for row in cls_rows]


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest) -> EmbeddingsResponse:
    """Return dense embeddings in the OpenAI response shape.

    The request's ``model`` field is ignored and the configured model is
    echoed back, matching ``/pooling`` and ``/tokenize``: this container
    serves exactly one model.

    Args:
        req: Embeddings request carrying one or more input texts.

    Returns:
        One embedding per input, in input order.
    """
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        return EmbeddingsResponse(
            data=[], model=MODEL_ID, usage=EmbeddingsUsage(prompt_tokens=0, total_tokens=0)
        )

    vectors = encode_dense(texts)
    token_total = sum(len(tokenize_ids(text)) for text in texts)
    return EmbeddingsResponse(
        data=[EmbeddingItem(index=i, embedding=vector) for i, vector in enumerate(vectors)],
        model=MODEL_ID,
        usage=EmbeddingsUsage(prompt_tokens=token_total, total_tokens=token_total),
    )
```

Add `import math` to the imports if absent.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group eval pytest eval/tests/test_sparse_server.py -v`
Expected: PASS (existing tests plus 6 new).

- [ ] **Step 5: Prove the dense test has teeth**

Temporarily change `hidden[:, 0]` to `hidden[:, 1]` (wrong pooling), run `test_encode_dense_cls_pools_and_normalises`, confirm it FAILS, then revert. Then remove the `l2_normalise` call, confirm it FAILS, then revert. Run `git diff src/sparse_server.py` afterwards and confirm both reverts are complete — a partially reverted mutation is worse than the gap it checked.

- [ ] **Step 6: Lint**

Run: `uv run pre-commit run --all-files`
Expected: ruff check, ruff format, pyrefly all pass.

- [ ] **Step 7: Commit**

```bash
git add src/sparse_server.py eval/tests/test_sparse_server.py
git commit -m "feat(embed): serve OpenAI-compatible dense embeddings from the same bge-m3"
```

---

### Task 2: Rename `sparse-only` → `embed-only`

**Files:** all ten below.

**Interfaces:**
- Consumes: Task 1's server.
- Produces: alias `embed-only`, image `vllm-service-embed-only`, targets `*-embed-only`, container env `EMBED_MODEL` / `EMBED_MAX_LENGTH` / `EMBED_HOST_PORT`.

**Context an implementer needs:**

This is a **pure rename with no logic change**. Nothing about request handling, pooling, tokenization or the compose topology changes. Keep it that way so the diff is reviewable as mechanical.

The exhaustive inventory (verified against `cfa4044`):
`Makefile`, `.env.example`, `README.md`, `CLAUDE.md`, `docker/compose.sparse-only.override.yaml`, `docker/Dockerfile.sparse.cpu`, `docker/compose.sparse-only.yaml`, `scripts/bundle_images.sh`, `eval/tests/test_sparse_server.py`, `src/sparse_server.py`.

**What renames:**

| From | To |
|---|---|
| `src/sparse_server.py` | `src/embed_server.py` |
| `eval/tests/test_sparse_server.py` | `eval/tests/test_embed_server.py` |
| `docker/Dockerfile.sparse.cpu` | `docker/Dockerfile.embed.cpu` |
| `docker/compose.sparse-only.yaml` | `docker/compose.embed-only.yaml` |
| `docker/compose.sparse-only.override.yaml` | `docker/compose.embed-only.override.yaml` |
| compose project `vllm-service-sparse-only` | `vllm-service-embed-only` |
| service + alias `sparse-only` | `embed-only` |
| image `vllm-service-sparse-only` | `vllm-service-embed-only` |
| `COMPOSE_SPARSE_ONLY{,_DEV}` | `COMPOSE_EMBED_ONLY{,_DEV}` |
| 6 targets `{build,bundle,up,up-dev,stop,down}-sparse-only` | `*-embed-only` |
| `bundle_images.sh` case + usage `sparse-only` | `embed-only` |
| container env `SPARSE_MODEL` | `EMBED_MODEL` (default `BAAI/bge-m3`) |
| container env `SPARSE_MAX_LENGTH` | `EMBED_MAX_LENGTH` (default 8192) |
| `SPARSE_HOST_PORT` | `EMBED_HOST_PORT` (value stays **8007**) |
| `NO_PROXY` entry `sparse-only` | `embed-only` |

**What does NOT rename:** the `/pooling`, `/tokenize`, `/v1/embeddings` and `/health` route paths; the `task: token_classify` field; anything in `compose.yaml` or `litellm.config.yaml`.

Use `git mv` for file renames so history is preserved.

- [ ] **Step 1: Rename the files**

```bash
git mv src/sparse_server.py src/embed_server.py
git mv eval/tests/test_sparse_server.py eval/tests/test_embed_server.py
git mv docker/Dockerfile.sparse.cpu docker/Dockerfile.embed.cpu
git mv docker/compose.sparse-only.yaml docker/compose.embed-only.yaml
git mv docker/compose.sparse-only.override.yaml docker/compose.embed-only.override.yaml
```

- [ ] **Step 2: Update every reference**

Work through the inventory. In the test file, the `import sparse_server` and every `sparse_server.` qualifier become `embed_server`. In the Dockerfile, `COPY src/sparse_server.py` and the `CMD`'s `sparse_server:app` become the new name. In the compose files, the `name:`, service key, alias, `NO_PROXY` entries, image, `dockerfile:` path, and env var names. In the Makefile, the compose var pair, the six targets, the `.PHONY` line, and the numbered header-comment entry. In `bundle_images.sh`, the case arm and usage string.

Then verify nothing was missed:
```bash
grep -rn "sparse-only\|sparse_server\|sparse\.cpu\|SPARSE_MODEL\|SPARSE_MAX_LENGTH\|SPARSE_HOST_PORT" \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.superpowers \
  --exclude-dir=__pycache__ --exclude-dir=.ruff_cache --exclude-dir=.pytest_cache .
```
Expected: **no output**, except in `README.md`/`CLAUDE.md` if you defer those to Task 3 — in which case say so. Use a case-**insensitive** grep as well (`-i`); a case-sensitive sweep for a lowercase term structurally cannot match an all-caps variable, which is a mistake already made once on this work.

- [ ] **Step 3: Verify nothing functional changed**

Run: `uv run --group eval pytest eval/tests/test_embed_server.py -v`
Expected: same test count and results as before the rename.

Run: `docker compose --env-file .env -f docker/compose.embed-only.yaml config >/dev/null && echo OK`
Expected: `OK`.

Run: `make -n build-embed-only up-embed-only up-dev-embed-only stop-embed-only down-embed-only bundle-embed-only`
Expected: all six print their commands; no "No rule to make target".

Run: `uv run pre-commit run --all-files`
Expected: pass.

- [ ] **Step 4: Rebuild the image under the new name**

Run: `docker compose --env-file .env -f docker/compose.embed-only.yaml build`
Expected: builds. Slow (CPU torch). If it fails for an environmental reason (network, registry, disk, no daemon), report BLOCKED rather than working around it.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(embed): rename the sparse-only shape to embed-only

It now serves dense, sparse and tokenize from one bge-m3, so the
name no longer describes it. Pure rename: no request handling,
pooling, tokenization or topology changes."
```

---

### Task 3: Docs, env example, and PR update

**Files:** `.env.example`, `README.md`, `CLAUDE.md`

**Interfaces:** consumes Tasks 1-2; produces nothing consumed later.

- [ ] **Step 1: Update `.env.example`**

Rename the sparse block to embed, and document the new surface: `EMBED_MODEL` (default `BAAI/bge-m3`), `EMBED_MAX_LENGTH` (8192), `EMBED_HOST_PORT` (8007). State that the shape serves **both** dense (`/v1/embeddings`) and sparse (`/pooling`, `/tokenize`) from one model, so a consumer points its embedding base and its sparse base at the same URL.

- [ ] **Step 2: Update `README.md` and `CLAUDE.md`**

Rename all `sparse-only` references. Update the shape's description to say it serves dense + sparse + tokenize. Add the operator-relevant consequence: **on a dev host this replaces Ollama's bge-m3**, so Ollama serves chat only and the model is loaded once rather than twice. Match the surrounding sections' depth and tone; do not invent a new format.

Also update the shape count if the docs state one — this is a rename, not an addition, so the count does not change, but check the wording still reads correctly.

- [ ] **Step 3: Verify**

Run: `uv run pre-commit run --all-files` → pass.
Run the case-insensitive sweep from Task 2 Step 2 again → no stale references outside deliberate historical mentions.

- [ ] **Step 4: Commit and update the PR**

```bash
git add .env.example README.md CLAUDE.md
git commit -m "docs(embed): document the embed-only shape serving dense and sparse"
git push
```

The branch already tracks `origin/feat/sparse-only-service`, so `git push` updates PR #78 in place. **Do not open a new PR.**

Then update the PR title and body to reflect the widened scope: it now adds a CPU shape serving dense **and** sparse, and it replaces Ollama's bge-m3 on dev rather than sitting alongside it. Mention that the branch name still says `sparse-only` for history; that is cosmetic and not worth a force-push.

```bash
gh pr edit 78 --title "feat(embed): add embed-only CPU shape serving dense + sparse bge-m3"
```

Report the PR URL and CI status.

---

## Self-review notes

- Spec coverage: dense route (Task 1), rename incl. container env knobs (Task 2), docs + `.env.example` + PR update (Task 3). The design's "ignore the request `model` field" decision is implemented in Task 1 Step 3 and stated in its context block.
- The design's explicit non-goal — fusing dense and sparse into one forward pass — is not attempted anywhere here, correctly.
- Type consistency: `l2_normalise` and `encode_dense` are spelled identically in the tests, the implementation, and the Interfaces block.
- Task 1 lands before the rename so the functional change is reviewable separately from a mechanical one; Task 1 therefore still edits `sparse_server.py` by its old name, which Task 2 then moves.
- Issue `vllm-service#74` (bound `/pooling` batch size) now also applies to `/v1/embeddings`, which has the same unbounded-`input` padding blow-up. Not fixed here; the issue needs updating to cover both routes.
