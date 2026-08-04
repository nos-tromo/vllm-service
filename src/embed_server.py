"""CPU dense + sparse embedding server for the embed-only deployment shape.

Serves bge-m3 dense embeddings alongside its learned sparse (lexical)
weights and tokenization — the same three routes the full-stack LiteLLM
router passes through to the vLLM ``embed`` backend: ``POST
/v1/embeddings`` (OpenAI-compatible dense), ``POST /pooling`` with
``task: "token_classify"`` (sparse), and ``POST /tokenize`` — so docint's
embedding client and its ``RemoteSparseEncoder`` target either backend
without protocol changes.

The full stack runs bge-m3 under vLLM's ``BgeM3EmbeddingModel``
architecture (see ``docker/compose.yaml``'s ``--hf-overrides``). This
server reproduces that model's dense and ``token_classify`` output on CPU
with ``transformers``: each route runs its own XLM-R forward, then either
CLS pooling + L2 normalisation (dense) or the repo's ``sparse_linear.pt``
head (``Linear(hidden_size, 1)``) + ReLU (sparse). Special and padding
tokens fall out downstream for sparse — the consumer drops non-positive
scores.

Inference uses ``transformers`` directly rather than ``FlagEmbedding``;
the latter pulls ``ir-datasets`` -> ``zlib-state``, which needs
``zlib.h`` to build from source on aarch64 and is irrelevant to
inference-only deployments.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


def _positive_int_env(name: str, default: int) -> int:
    """Read an integer env var that must be at least 1.

    A zero or negative value for either knob would not crash anything —
    ``MAX_BATCH_TOKENS < 1`` silently degrades to one-text-per-forward
    batching, ``MAX_LENGTH < 1`` to empty truncation — so the
    misconfiguration would be invisible until someone profiled a slow
    container. Failing at startup puts it in the logs instead.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        The parsed value.

    Raises:
        ValueError: When the value parses below 1 (or not as an integer,
            via ``int()`` itself).
    """
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


MODEL_ID = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
MAX_LENGTH = _positive_int_env("EMBED_MAX_LENGTH", 8192)
MAX_BATCH_TOKENS = _positive_int_env("EMBED_MAX_BATCH_TOKENS", 16384)
SPARSE_LINEAR_FILE = os.environ.get("SPARSE_LINEAR_FILE", "sparse_linear.pt")

app = FastAPI(title="vllm-service embed-only", version="1.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Explicit dtype: the float32 default is exactly what a transformers
# major bump changes, and dense CLS+L2 vectors drift silently if it moves.
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.train(False)


def _load_sparse_head(hidden_size: int) -> torch.nn.Linear:
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


def _encode_batch(texts: list[str]) -> dict[str, torch.Tensor]:
    """Tokenize a batch of texts for the dense and sparse encoders.

    Both ``encode_token_weights`` and ``encode_dense`` route through this
    seam so the two encoders can never silently diverge on
    ``truncation``/``max_length``/special-token handling — the same
    rationale as ``tokenize_ids`` above (the ``/tokenize`` seam), applied
    to the batched shape the other two routes share.

    Args:
        texts: Input texts.

    Returns:
        The tokenizer's batch encoding (``input_ids``, ``attention_mask``,
        ...) as padded PyTorch tensors.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def plan_batches(lengths: list[int], budget: int) -> list[tuple[int, int]]:
    """Split token lengths into contiguous spans within a padded-token budget.

    ``_encode_batch`` pads with ``padding=True``, so a batch's real cost
    is ``rows x longest row``, not the sum of its rows: one 4k-token
    chunk among 63 short ones pads all 64 to 4k. Cost is therefore
    charged against the padded rectangle, which is what bounds both peak
    activation memory and wall-clock per forward pass.

    Greedy and order-preserving: inputs are never reordered to pack
    batches more tightly, because both routes return one row per input
    positionally.

    Args:
        lengths: Token count of each input, in input order.
        budget: Maximum padded tokens (rows x longest row) per batch.

    Returns:
        ``(start, end)`` half-open index spans covering every input
        exactly once, in order. A single input longer than *budget*
        occupies its own span rather than being dropped or split — it is
        already capped at ``MAX_LENGTH`` by truncation.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    widest = 0
    for index, length in enumerate(lengths):
        candidate = max(widest, length)
        if index > start and candidate * (index - start + 1) > budget:
            spans.append((start, index))
            start = index
            widest = length
        else:
            widest = candidate
    if start < len(lengths):
        spans.append((start, len(lengths)))
    return spans


def _iter_batches(texts: list[str]) -> Iterator[tuple[list[str], list[int]]]:
    """Yield *texts* in sub-batches bounded by ``MAX_BATCH_TOKENS``.

    The single bounding seam for both encoders: ``/pooling`` and
    ``/v1/embeddings`` pad identically, so they must split identically.
    Two copies of this logic could drift, and the symptom of the drift
    would be a timeout under load rather than a wrong answer — invisible
    until an ingest job failed.

    Lengths come from ``tokenize_ids``, the same seam ``/tokenize``
    reports, so the budget is charged against the token counts a client
    can actually observe. Each batch's lengths are yielded alongside it
    so callers that need token accounting (``/v1/embeddings``'s
    ``usage``) reuse them instead of tokenizing a second time.

    Args:
        texts: Input texts, in input order.

    Yields:
        ``(batch, lengths)`` pairs — consecutive non-empty slices of
        *texts* in input order, each with its texts' token counts.
    """
    lengths = [len(tokenize_ids(text)) for text in texts]
    for start, end in plan_batches(lengths, MAX_BATCH_TOKENS):
        yield texts[start:end], lengths[start:end]


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

    The input is encoded in ``MAX_BATCH_TOKENS``-bounded sub-batches
    (see ``_iter_batches``) and the results concatenated, so a caller's
    batch size does not decide how much work one forward pass does.

    Args:
        texts: Input texts.

    Returns:
        One list of per-token weights per input, in input order.
    """
    rows: list[list[float]] = []
    for batch, _lengths in _iter_batches(texts):
        encoded = _encode_batch(batch)
        with torch.no_grad():
            hidden = model(**encoded, return_dict=True).last_hidden_state
            weights = torch.relu(sparse_head(hidden)).squeeze(-1)

        for row, mask in zip(weights.tolist(), encoded["attention_mask"].tolist(), strict=True):
            rows.append([float(weight) for weight, keep in zip(row, mask, strict=True) if keep == 1])
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


def encode_dense(texts: list[str]) -> tuple[list[list[float]], int]:
    """Compute bge-m3 dense embeddings for each text.

    Dense is CLS pooling — the first token of ``last_hidden_state`` —
    followed by L2 normalisation, matching FlagEmbedding's ``cls``
    sentence-pooling method for this model.

    Bounded in ``MAX_BATCH_TOKENS``-sized sub-batches on the same seam as
    the sparse route (see ``_iter_batches``); the dense route pads
    identically, so it inflates identically. The token total rides along
    from that seam's own counts, so the route's ``usage`` field never
    tokenizes a second time — and can never disagree with the lengths
    the budget was charged against.

    Args:
        texts: Input texts.

    Returns:
        One unit-length vector per input, in input order, plus the total
        token count across all inputs.
    """
    vectors: list[list[float]] = []
    token_total = 0
    for batch, lengths in _iter_batches(texts):
        token_total += sum(lengths)
        encoded = _encode_batch(batch)
        with torch.no_grad():
            hidden = model(**encoded, return_dict=True).last_hidden_state
        for row in hidden[:, 0].tolist():
            vectors.append(l2_normalise([float(component) for component in row]))
    return vectors, token_total


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
        return EmbeddingsResponse(data=[], model=MODEL_ID, usage=EmbeddingsUsage(prompt_tokens=0, total_tokens=0))

    vectors, token_total = encode_dense(texts)
    return EmbeddingsResponse(
        data=[EmbeddingItem(index=i, embedding=vector) for i, vector in enumerate(vectors)],
        model=MODEL_ID,
        usage=EmbeddingsUsage(prompt_tokens=token_total, total_tokens=token_total),
    )
