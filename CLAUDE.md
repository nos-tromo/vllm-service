# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A pure infrastructure repo: a Docker Compose stack that fronts several vLLM
backends with a single LiteLLM Proxy router. The Docker assets live under
`docker/` (`docker/compose.yaml`, `docker/compose.override.yaml`,
`docker/compose.gliner-only.yaml`, `docker/compose.rerank-only.yaml`,
`docker/compose.clip-only.yaml`, `docker/Dockerfile.vllm`,
`docker/Dockerfile.gliner.cuda`, `docker/Dockerfile.gliner.cpu`,
`docker/Dockerfile.rerank.cpu`, `docker/Dockerfile.clip.cuda`,
`docker/Dockerfile.clip.cpu`, `docker/litellm.config.yaml`) plus `.env`
and `.dockerignore` at the repo root. The only Python sources are
`docker/rerank_server.py` and `docker/clip_server.py` — small FastAPI
wrappers around Hugging Face models that ship because there is no
off-the-shelf CPU server that speaks the Jina-shape `/rerank` or
`/clip` contracts the full stack exposes.

## Deployment shapes

Four independent compose projects, picked per host:

- **Full stack** (`docker/compose.yaml`, CUDA-required) — chat, embed, rerank,
  gliner, router; optional audio + translate via `PROFILE=media`. The original
  shape; reached as `http://vllm-router:4000/...` on `inference-net`.
  GLiNER is routed via the router's `/gliner` pass-through.
- **NER-only** (`docker/compose.gliner-only.yaml`, CPU OK) — a single
  `gliner-only` container on `inference-net`, no router, no GPU requirement.
  Intended for hosts that run Ollama (or another non-vLLM provider) for
  chat/embeddings but still want NER out of the consuming app. Reached
  directly as `http://gliner-only:8000/gliner` on `inference-net`; there is
  no Bearer auth (trust `inference-net` the way `data-net` is trusted for
  Qdrant). Uses the CPU-only `Dockerfile.gliner.cpu` (non-CUDA PyTorch
  base, multi-arch) and defaults `NER_MODEL` to `gliner_medium-v2.5`.
- **Rerank-only** (`docker/compose.rerank-only.yaml`, CPU OK) — a single
  `rerank-cpu` container on `inference-net`, no router, no GPU. Same
  audience as NER-only; pairs with it so an Ollama-on-CPU host can offer
  both `/gliner` and `/rerank` without the full CUDA stack. Reached as
  `http://rerank-cpu:8000/rerank` on `inference-net`. Speaks the same
  Jina-shape `{model, query, documents, top_n}` → `{results: [{index,
  relevance_score}]}` contract as the full-stack vLLM `rerank` service,
  so consumers (docint's `VLLMRerankPostprocessor`) target either
  backend by changing the base URL alone. Uses `Dockerfile.rerank.cpu`
  (uv-managed Python 3.11, CPU torch, transformers) and ships a tiny
  FastAPI server at `docker/rerank_server.py` that drives the cross-
  encoder directly (tokenize → forward → sigmoid).
- **CLIP-only** (`docker/compose.clip-only.yaml`, CPU OK) — a single
  `clip-embed` container on `inference-net`, no router, no GPU. Same
  audience as NER-only / Rerank-only; co-deployable so a non-CUDA host
  can offer `/gliner`, `/rerank`, and `/clip/embed_{image,text}` at
  once. Reached as `http://clip-embed:8000/clip/*`. Speaks the same
  contract the full-stack `clip` service exposes (also a FastAPI app
  on the same Python file), so docint's image-ingestion path
  (`docint/utils/clip_client.py`) targets either backend by changing
  the base URL alone. Uses `Dockerfile.clip.cpu` (uv-managed
  Python 3.11, CPU torch, transformers, Pillow) and ships
  `docker/clip_server.py`.

The shapes are **not profiles of one compose file** — they have different
images, different topologies, and (gliner-only, rerank-only, clip-only) no
router. Pick one per host. They reuse the same external `inference-net`
network and `huggingface-cache` volume, so the one-time
`make network` / `make volumes` prerequisites apply to all of them. The
three CPU-only shapes can coexist on a single host because they target
different network aliases and host ports.

## Common commands

The `Makefile` is the entry point — it points Compose at `docker/compose.yaml`,
since a bare `docker compose` from the repo root no longer finds the compose
file. The service set is read from `PROFILE` in `.env`: empty = core stack;
`PROFILE=media` adds translate + audio. Override per-invocation as
`make up PROFILE=media`.

Prerequisites (one-time per host):

```bash
make network           # create the external inference-net
make volume            # create the huggingface-cache Docker volume
cp .env.example .env   # then edit model IDs / API key / GPU placement
```

Bring the stack up:

```bash
make build                 # build images for the active service set
make up                    # core (production shape, no host ports)
make up-dev                # core with host ports published (dev)
make up PROFILE=media      # add translate + audio (production shape)
make up-dev PROFILE=media  # add translate + audio with host ports published
```

`make up` runs the base `docker/compose.yaml` alone (production shape — no host
ports); `make up-dev` layers `docker/compose.override.yaml` so the router is
published on the host for dev.

Or, for the NER-only shape (no CUDA, no router — pairs with Ollama):

```bash
make build-gliner-only    # builds vllm-service-gliner-cpu
make up-gliner-only       # one gliner-only container on inference-net (no host port)
make up-dev-gliner-only   # like 'up-gliner-only', but publishes the GLiNER port on the host
make stop-gliner-only
make bundle-gliner-only   # versioned .tar.gz of the gliner-cpu image
```

Or, for the Rerank-only shape (no CUDA, no router — pairs with Ollama,
typically co-deployed with NER-only):

```bash
make build-rerank-only    # builds vllm-service-rerank-cpu
make up-rerank-only       # one rerank-cpu container on inference-net (no host port)
make up-dev-rerank-only   # like 'up-rerank-only', but publishes the rerank port on the host
make stop-rerank-only
make bundle-rerank-only   # versioned .tar.gz of the rerank-cpu image
```

Or, for the CLIP-only shape (no CUDA, no router — pairs with Ollama,
typically co-deployed with NER-only and Rerank-only):

```bash
make build-clip-only      # builds vllm-service-clip-cpu
make up-clip-only         # one clip-embed container on inference-net (no host port)
make up-dev-clip-only     # like 'up-clip-only', but publishes the CLIP port on the host
make stop-clip-only
make bundle-clip-only     # versioned .tar.gz of the clip-cpu image
```

Other useful operations use the raw compose form, pointed at the compose file
(append `--profile media` when the media service set is active):

```bash
docker compose --env-file .env -f docker/compose.yaml logs -f router    # follow LiteLLM proxy logs
docker compose --env-file .env -f docker/compose.yaml logs -f chat      # follow a single backend
docker compose --env-file .env -f docker/compose.yaml restart chat      # reload one backend after .env change
docker compose --env-file .env -f docker/compose.yaml ps                # health status of each service
make stop                                                               # stop the active service set
```

Build and bundle commands:

```bash
make build               # build images for the active service set
make bundle              # build + ship the active service set as versioned .tar.gz pair
make bundle PROFILE=media  # build + ship core + media as versioned .tar.gz pair
```

There is no test suite or linter. The Python files
(`docker/rerank_server.py`, `docker/clip_server.py`) are small enough to
verify by curl against the running standalone containers — see the
Architecture / Rerank-only and Architecture / CLIP-only sections below.

## Architecture

### Routing model

`router` (LiteLLM Proxy, port 4000 inside, published on
`${ROUTER_HOST_PORT:-9000}` on the host by `docker/compose.override.yaml`
when `make up-dev` is used) is the **only** entry point. Clients always send to
the router and select a backend by the `model` field in the request body —
there is no path-based dispatch. `docker/litellm.config.yaml` maps each
`model_name` (read from env vars at startup) to an upstream `api_base`
(`http://chat:8000/v1`, `http://embed:8000/v1`, etc.).

LiteLLM natively exposes `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/translations`, and
`/v1/models`. vLLM-specific paths (`/v1/rerank`, `/pooling`, `/tokenize`) and
GLiNER's `/gliner` are forwarded by `pass_through_endpoints` in
`docker/litellm.config.yaml`. `/gliner` is the only pass-through whose upstream
is *not* a vLLM container — it goes to the `gliner` service (Ray Serve).

### Backends

Most backends share the same `docker/Dockerfile.vllm` image, launched with a
different model and per-service env-driven flags:

- `chat` — general LLM (`TEXT_MODEL`)
- `embed` — embeddings, `--runner pooling --convert embed` (`EMBED_MODEL`)
- `rerank` — reranker, `--runner pooling` (`RERANK_MODEL`)
- `translate` *(profile: media)* — TranslateGemma fork (`TRANSLATE_MODEL`)
- `audio` *(profile: media)* — Whisper (`WHISPER_MODEL`)

`gliner` is the one exception — it uses **`docker/Dockerfile.gliner.cuda`** (pytorch
base + `gliner[serve]`) and runs Ray Serve, not vLLM. GLiNER's span-matching head
isn't a stock HF classification head and the DeBERTa-v2/v3 disentangled
attention used by the v2.5 checkpoints isn't natively supported by vLLM,
so vLLM-native serving is not viable. The endpoint is `POST /gliner` with
body `{text, labels, threshold}` and is reached via the router's
`/gliner` pass-through (not `/v1/...`).

- `gliner` — GLiNER zero-shot NER via Ray Serve (`NER_MODEL`, default
  `gliner-community/gliner_large-v2.5` on CUDA; set `NER_DEVICE=cpu` with
  the `gliner_medium-v2.5` variant for CPU-only hosts)

Each buildable service in `docker/compose.yaml` carries
`image: vllm-service-<svc>:${VLLM_SERVICE_VERSION:-latest}`. A raw `docker
compose -f docker/compose.yaml build` produces `:latest` tags for dev
workflows; `make bundle` exports `VLLM_SERVICE_VERSION=<date>-<short-sha>` so
the same compose file also produces version-tagged tarballs for offline
shipping.

All backends listen on internal port 8000 and expose only `vllm-net` — they are
not reachable from outside the compose project. Only `router` joins the external
`inference-net` (with alias `vllm-router`) for cross-project consumers.

### Dependency overlay

The vLLM base image (`vllm/vllm-openai:v0.20.1`, pinned by digest) ships with
plain vLLM and its full CUDA runtime preinstalled in the system Python prefix.
The Dockerfile adds vLLM's `[audio]` extras (`av`, `scipy`, `soundfile`,
`mistral_common[audio]`) so the `audio` (Whisper) and `translate` services
have what they need, plus `orjson` — vLLM picks it up opportunistically for
faster OpenAI-endpoint JSON serialization (falls back to stdlib `json` if
absent, so it's a perf bump rather than a hard requirement).

No lockfile, no `uv`, no `pyproject.toml` — the overlay is one line. If you
add another dependency later and want transitive pinning, that's the moment
to reintroduce a lockfile-driven workflow; for one extras specifier the
ergonomics aren't worth it.

### Service startup ordering

`depends_on … condition: service_healthy` chains the backends serially:
`chat → embed → rerank → gliner → audio → translate → router`. This is intentional —
backends compete for GPU memory at startup, so they are brought up one at a time.
Healthchecks hit `http://localhost:8000/health` (vLLM backends),
`http://localhost:8000/-/healthz` (the `gliner` Ray Serve container), and
`/health/liveliness` (router). Allow ~120s `start_period` before treating a
backend as unhealthy.

### Configuration surface

All tuning is done via `.env` (see `.env.example`). Per-service env vars follow
the pattern `<SERVICE>_<KNOB>` (e.g. `CHAT_GPU_MEMORY_UTILIZATION`,
`TRANSLATE_MAX_MODEL_LEN`, `EMBED_HF_OVERRIDES`, `NER_ENABLE_FLASHDEBERTA`). The
`chat`, `translate`, and `gliner` entrypoints use a shell builder pattern (`set --
<cmd> …` then conditional `set -- "$@" --flag`) so optional flags are only
passed when the corresponding env var is set — when adding a new optional flag,
follow that same pattern rather than hard-coding it in `command:`. The `gliner`
shell builder invokes `python -m gliner.serve` instead of `vllm serve`, but the
structure is identical.

`OPENAI_API_KEY` serves double duty: it is both the upstream API key passed to
each vLLM `--api-key` and the LiteLLM `master_key` that gates the router.

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are the default — models are
expected to already be present in the shared `huggingface-cache` volume. To
download a new model, temporarily flip both to `0` (and set `HF_TOKEN` if the
model is gated).

### Switching models

Update the relevant `*_MODEL` variable in `.env` and restart the stack. **No
edits to `docker/litellm.config.yaml` are needed** — model names are resolved
via `os.environ/<VAR>` at startup. Clients must send the exact model ID string
in their request `model` field; `/v1/models` returns the active set.

`NER_MODEL` is the exception: GLiNER's server has no OpenAI-shaped routes, so
it is not declared in `model_list` and does not appear in `/v1/models`.
Clients hit `/gliner` directly with a GLiNER-native body and never use the
`model` field. Switching `NER_MODEL` and restarting `gliner` still works.

### Rerank-only shape (CPU)

The `rerank-only` compose project runs `docker/rerank_server.py` — a small
FastAPI app that loads a Hugging Face cross-encoder
(`AutoModelForSequenceClassification`), tokenizes each (query, document)
pair, takes the seq-classification logit, and sigmoid-normalizes — the
same forward pass FlagEmbedding does internally for bge-reranker-style
models, just without the heavyweight FlagEmbedding dep tree
(`ir-datasets` → `zlib-state`, fails to build on aarch64). Exposes the
same Jina-shape `POST /rerank` contract as the full-stack vLLM `rerank`
service:

```
POST /rerank
{"model": "...", "query": "...", "documents": [...], "top_n": 5}
→
{"id": "rerank-...", "model": "...",
 "results": [{"index": 0, "relevance_score": 0.95}, ...]}
```

Model identity is fixed at container startup via `RERANK_MODEL`
(defaults to `BAAI/bge-reranker-v2-m3`, the same model the GPU stack
uses, so scores match). The request `model` field is accepted but not
enforced — the server always uses the model it loaded at boot.
`RERANK_USE_FP16=true` is honored on hosts that benefit (rare on CPU).

`GET /health` returns `{"status": "ok", "model": "..."}` and is the
healthcheck target. There is no `/-/healthz` (Ray Serve isn't used here).

Smoke-test:

```bash
curl -fsS -X POST http://localhost:${RERANK_HOST_PORT:-8001}/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is RAG", "documents": ["retrieval augmented generation", "lunch menu"], "top_n": 2}'
```

### CLIP-only shape (CPU)

The `clip-only` compose project runs `docker/clip_server.py` — a small
FastAPI app that loads a Hugging Face CLIP model (`CLIPModel` +
`AutoProcessor`), runs the image or text tower, and L2-normalizes the
output. Same forward pass the legacy in-process
`CLIPImageEmbeddingBackend` ran in docint, so existing `_images` Qdrant
collections stay compatible as long as `CLIP_MODEL` matches the
ingestion-time model. Endpoints:

```
POST /clip/embed_image      # multipart `file=<bytes>` OR JSON {"image_b64": ...}
  -> {"embedding": [float, ...], "dimension": int}

POST /clip/embed_text       # JSON {"text": "..."}
  -> {"embedding": [float, ...], "dimension": int}

GET  /clip/dimension        # one-shot fetch for collection compat checks
  -> {"dimension": int}
```

Model identity is fixed at container startup via `CLIP_MODEL` (default
`openai/clip-vit-base-patch32`). The full-stack `clip` service runs
the same file on the same FastAPI surface, and the LiteLLM router
exposes `/clip/embed_image` (multipart), `/clip/embed_text`, and
`/clip/dimension` as pass-throughs.

`GET /health` returns
`{"status": "ok", "model": "...", "dimension": N, "device": "..."}`
and is the healthcheck target.

Smoke-test:

```bash
# text tower
curl -fsS -X POST http://localhost:${CLIP_HOST_PORT:-8002}/clip/embed_text \
  -H 'Content-Type: application/json' \
  -d '{"text": "a photo of a cat"}'

# image tower (multipart)
curl -fsS -X POST http://localhost:${CLIP_HOST_PORT:-8002}/clip/embed_image \
  -F 'file=@/path/to/img.jpg'

# dimension probe
curl -fsS http://localhost:${CLIP_HOST_PORT:-8002}/clip/dimension
```

### TranslateGemma quirk

`translate` runs `Infomaniak-AI/vllm-translategemma-4b-it` (a vLLM-compatible
repackaging — the stock `google/translategemma-*` cannot be served by a vanilla
OpenAI client). Clients must encode source/target/text inline:

```
<<<source>>>{iso_src}<<<target>>>{iso_tgt}<<<text>>>{text}
```

Trained for ~2K context — keep `TRANSLATE_MAX_MODEL_LEN=2048` unless you have a
specific reason to raise it.
