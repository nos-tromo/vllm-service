"""CPU/GPU CLIP image-embedding server.

Mirrors the behaviour of ``docint.core.ingest.images_service.
CLIPImageEmbeddingBackend`` over HTTP so docint can drop the in-process
CLIP runtime and reach embedding from a shared inference container, the
same way it now reaches GLiNER and rerank.

Endpoints:

    POST /clip/embed_image
        Body either multipart with ``file=<bytes>`` OR JSON
        ``{"image_b64": "<base64-encoded image bytes>"}``.
        Returns ``{"embedding": [float, ...], "dimension": int}``.

    POST /clip/embed_text
        Body JSON ``{"text": "<query>"}``.
        Returns ``{"embedding": [float, ...], "dimension": int}``.

    GET /clip/dimension
        Returns ``{"dimension": int}`` for a one-shot fetch at the
        consumer side; lets docint check Qdrant collection compatibility
        without burning an embed call.

    GET /health
        Liveness probe; returns ``{"status": "ok", "model": ...}``.

Model identity is fixed at container startup via ``CLIP_MODEL`` (default
``openai/clip-vit-base-patch32``). Loaded once at module import so the
first request is hot. The image and text encoders both use the standard
HuggingFace ``transformers`` CLIP pipeline:

* image: ``Image.open(BytesIO(bytes)).convert("RGB") -> processor ->
  model.get_image_features -> L2 normalize``
* text:  ``processor(text=[t], padding, truncation) ->
  model.get_text_features -> L2 normalize``

Identical to the legacy in-process path, so existing ``_images`` Qdrant
collections stay compatible as long as ``CLIP_MODEL`` matches the model
used at ingest time.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, CLIPModel

MODEL_ID = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
DEVICE = os.environ.get("CLIP_DEVICE", "cpu")

app = FastAPI(title="vllm-service clip", version="1.0")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID)
model.train(False)
model.to(DEVICE)
DIMENSION = int(model.config.projection_dim)


class EmbedTextRequest(BaseModel):
    """JSON body for the text embed endpoint."""

    text: str


class EmbedResponse(BaseModel):
    """Embedding response shared by image and text endpoints."""

    embedding: list[float]
    dimension: int


def _embed_image_bytes(image_bytes: bytes) -> list[float]:
    """Run the CLIP image tower on raw bytes and L2-normalize the output.

    Args:
        image_bytes: Raw image bytes (any format Pillow can decode).

    Returns:
        A list of floats of length ``DIMENSION``, L2-normalized.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].detach().cpu().tolist()


def _embed_text(text: str) -> list[float]:
    """Run the CLIP text tower on a single string and L2-normalize.

    Args:
        text: The query text.

    Returns:
        A list of floats of length ``DIMENSION``, L2-normalized.
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].detach().cpu().tolist()


@app.post("/clip/embed_image", response_model=EmbedResponse)
async def embed_image(request: Request) -> EmbedResponse:
    """Embed an image submitted as multipart upload or base64 JSON.

    Accepts two body shapes (dispatched by ``Content-Type``):

    * ``multipart/form-data`` with a ``file=<bytes>`` field — preferred
      for bulk ingestion paths that already hold the bytes locally.
    * ``application/json`` with ``{"image_b64": "..."}`` — convenient
      for HTTP clients that only speak JSON.

    FastAPI can't bind both shapes on one endpoint via a single
    function signature (multipart parsing greedily intercepts the body
    before the JSON model gets a chance), so we read the raw request
    and branch on the content type.

    Args:
        request: The incoming HTTP request.

    Returns:
        EmbedResponse: Normalized CLIP image embedding + dimension.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="multipart body missing 'file' field")
        image_bytes = await upload.read()
    elif content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        image_b64 = payload.get("image_b64") if isinstance(payload, dict) else None
        if not isinstance(image_b64, str) or not image_b64:
            raise HTTPException(status_code=400, detail="JSON body must include non-empty 'image_b64' string")
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be multipart/form-data or application/json",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image payload")
    try:
        embedding = _embed_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"failed to decode/embed image: {exc}") from exc
    return EmbedResponse(embedding=embedding, dimension=DIMENSION)


@app.post("/clip/embed_text", response_model=EmbedResponse)
def embed_text(req: EmbedTextRequest) -> EmbedResponse:
    """Embed a query string with the CLIP text tower.

    Args:
        req: Text embed request body.

    Returns:
        EmbedResponse: Normalized CLIP text embedding + dimension.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    embedding = _embed_text(req.text)
    return EmbedResponse(embedding=embedding, dimension=DIMENSION)


@app.get("/clip/dimension")
def get_dimension() -> dict[str, int]:
    """Return the projection dimensionality of the loaded CLIP model.

    Lets consumers verify Qdrant ``_images`` collection compatibility
    without spending a real embed call.
    """
    return {"dimension": DIMENSION}


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe target for the compose healthcheck."""
    return {"status": "ok", "model": MODEL_ID, "dimension": DIMENSION, "device": DEVICE}
