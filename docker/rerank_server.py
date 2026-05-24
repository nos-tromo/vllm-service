"""CPU rerank server for the rerank-only deployment shape.

Wraps a Hugging Face sequence-classification cross-encoder
(default ``BAAI/bge-reranker-v2-m3`` — the same model the GPU stack
uses) in a tiny FastAPI app and exposes ``POST /rerank`` in the same
Jina-shape contract as the full-stack vLLM ``rerank`` service, so
consumers (docint's ``VLLMRerankPostprocessor``) target either backend
without protocol changes.

Request:
    {"model": "...", "query": "...", "documents": [...], "top_n": 5}

Response:
    {"id": "rerank-...", "model": "...",
     "results": [{"index": 0, "relevance_score": 0.95}, ...]}

Inference uses ``transformers`` directly rather than ``FlagEmbedding``;
the latter pulls ``ir-datasets`` → ``zlib-state``, which needs
``zlib.h`` to build from source on aarch64 and is irrelevant to
inference-only deployments. For bge-reranker-style cross-encoders the
scores are identical: tokenize the (query, doc) pair, forward through
the seq-classification head, take the logit, sigmoid-normalize.
"""

from __future__ import annotations

import os
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_FP16 = os.environ.get("RERANK_USE_FP16", "false").lower() == "true"
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))

app = FastAPI(title="vllm-service rerank-only", version="1.0")

_dtype = torch.float16 if USE_FP16 else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=_dtype)
model.train(False)


class RerankRequest(BaseModel):
    """Jina-shape rerank request body."""

    model: str | None = None
    query: str
    documents: list[str] = Field(default_factory=list)
    top_n: int | None = None


class RerankResult(BaseModel):
    """One reranked document with its relevance score."""

    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Jina-shape rerank response body."""

    id: str
    model: str
    results: list[RerankResult]


def _score_pairs(query: str, documents: list[str]) -> list[float]:
    """Score every (query, doc) pair with the cross-encoder.

    Args:
        query: User query.
        documents: Candidate documents to rerank.

    Returns:
        Sigmoid-normalized relevance scores in [0, 1], same order as
        ``documents``.
    """
    pairs = [[query, doc] for doc in documents]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits.view(-1).float()
    return torch.sigmoid(logits).tolist()


@app.post("/rerank")
def rerank(req: RerankRequest) -> RerankResponse:
    """Score every document against the query and return them top-N sorted.

    Args:
        req: Rerank request carrying the query, candidate documents, and
            optional ``top_n`` cutoff.

    Returns:
        A Jina-shape response with one ``RerankResult`` per kept document,
        sorted by ``relevance_score`` descending.
    """
    request_id = f"rerank-{uuid.uuid4().hex[:12]}"
    if not req.documents:
        return RerankResponse(id=request_id, model=MODEL_ID, results=[])

    scores = _score_pairs(req.query, req.documents)
    ranked = sorted(
        (RerankResult(index=i, relevance_score=s) for i, s in enumerate(scores)),
        key=lambda r: r.relevance_score,
        reverse=True,
    )
    if req.top_n is not None:
        ranked = ranked[: req.top_n]
    return RerankResponse(id=request_id, model=MODEL_ID, results=list(ranked))


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID}
