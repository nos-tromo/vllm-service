"""CPU rerank server for the rerank-only deployment shape.

Wraps ``FlagEmbedding.FlagReranker`` in a tiny FastAPI app and exposes
``POST /rerank`` in the same Jina-shape contract as the full-stack vLLM
``rerank`` service, so consumers (docint's ``VLLMRerankPostprocessor``)
can target either backend without protocol changes.

Request:
    {"model": "...", "query": "...", "documents": [...], "top_n": 5}

Response:
    {"id": "rerank-...", "model": "...",
     "results": [{"index": 0, "relevance_score": 0.95}, ...]}

The model is selected at startup via the ``RERANK_MODEL`` env var (default
``BAAI/bge-reranker-v2-m3`` — same as the GPU stack's ``rerank`` service)
and loaded once at import time so the first request is hot.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI
from FlagEmbedding import FlagReranker
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_FP16 = os.environ.get("RERANK_USE_FP16", "false").lower() == "true"

app = FastAPI(title="vllm-service rerank-only", version="1.0")
reranker = FlagReranker(MODEL_ID, use_fp16=USE_FP16)


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

    pairs = [[req.query, doc] for doc in req.documents]
    raw = reranker.compute_score(pairs, normalize=True)
    scores: list[float] = [float(raw)] if isinstance(raw, (int, float)) else [float(s) for s in raw]

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
