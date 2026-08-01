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
from pydantic import BaseModel
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
