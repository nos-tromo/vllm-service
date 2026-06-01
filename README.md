# Standalone vLLM Service

This repository runs the standalone routed vLLM deployment used by [docint](https://github.com/nos-tromo/docint) and
other consumers.

## Purpose

The stack exposes one routed HTTP endpoint fronted by [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy).
LiteLLM dispatches each request to the right vLLM backend based on the `model`
field in the request body, and natively exposes:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/v1/models`

Additional vLLM-specific endpoints are pass-through forwarded by LiteLLM to the
relevant backend:

- `/rerank`
- `/pooling`
- `/tokenize`
- `/gliner` (zero-shot NER; non-OpenAI shape)
- `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension` (CLIP image+text tower)

Internally it runs:

- `router` (LiteLLM Proxy)
- `chat`
- `embed`
- `rerank`
- `ner` (GLiNER, served via Ray Serve rather than vLLM)
- `clip` (CLIP image+text tower, served via FastAPI rather than vLLM)

The following services are optional and only started with `--profile media`:

- `audio`
- `translate`

Model-to-backend routing is declared in `docker/litellm.config.yaml`. Clients
select a backend purely by the `model` field they send; there is no path-based
dispatch.

For hosts that cannot run the CUDA stack (Mac dev boxes, ROCm or CPU-only
Linux running Ollama for chat/embed), this repo also ships three standalone
CPU deployments:

- **NER-only** (`docker/compose.gliner-only.yaml`) — a single `gliner-ner`
  container exposing `/gliner`. See "NER-only deployment" below.
- **Rerank-only** (`docker/compose.rerank-only.yaml`) — a single
  `rerank-cpu` container exposing the same Jina-shape `/rerank` contract
  as the full stack. See "Rerank-only deployment" below.
- **CLIP-only** (`docker/compose.clip-only.yaml`) — a single `clip-embed`
  container exposing the same `/clip/embed_{image,text}` contract as the
  full stack. See "CLIP-only deployment" below.

The three can be co-deployed on the same host so the consuming app has
all of `/gliner`, `/rerank`, and `/clip/*` available without the full
CUDA stack.

## Usage

The Docker assets live under `docker/`: a base `compose.yaml`, a
`compose.override.yaml` dev overlay, the Dockerfiles, and
`litellm.config.yaml`. The `Makefile` is the entry point — it points Compose
at `docker/compose.yaml`, since a bare `docker compose` from the repo root no
longer finds the compose file.

1. Copy `.env.example` to `.env` and set the model IDs, API key, and any
   GPU-placement settings. `make up-dev` publishes the router on host port
   `9000`; if that port is already in use, set `ROUTER_HOST_PORT` in `.env`
   to another free port such as `9001`.

  If `TEXT_MODEL` is a Gemma 4 instruct checkpoint and you want OpenAI-style
  tool calling, also set these chat flags in `.env`:

  ```bash
  CHAT_ENABLE_AUTO_TOOL_CHOICE=true
  CHAT_TOOL_CALL_PARSER=gemma4
  CHAT_REASONING_PARSER=gemma4
  CHAT_CHAT_TEMPLATE=examples/tool_chat_template_gemma4.jinja
  ```

  Without them, vLLM rejects `tool_choice="auto"` even though the model
  itself supports tool use.
2. Ensure the external `huggingface-cache` Docker volume exists.
3. Initialize the shared proxy network and persistent model cache:

   ```bash
   make network   # create the external inference-net
   make volume    # create the huggingface-cache Docker volume
   ```

4. Build and start the stack. The service set is read from `PROFILE` in
   `.env`: leave it empty for the core stack, or set `PROFILE=media` to also
   start the `translate` and `audio` services. Override per-invocation as
   `make up-dev PROFILE=media`:

   ```bash
   make build                 # build images for the active service set
   make up-dev                # core stack (router, chat, embed, rerank, ner) with the router published on the host
   make up-dev PROFILE=media  # core + media (translate, audio) with host ports
   make up                    # same as up-dev but production shape (no host ports)
   ```

   `make up-dev` layers `docker/compose.override.yaml` so the router is
   published on the host for local development; `make up` runs the base
   `docker/compose.yaml` alone (production shape, no host ports) —
   in-network consumers reach the router as `vllm-router:4000` on
   `inference-net` regardless.

5. Point third-party app at the router.

If the consuming app is on the same shared Docker network, use the router
alias directly:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://vllm-router:4000/v1
   OPENAI_API_KEY=<token>
   ```

If the consuming app is outside that network, use a host or reverse-proxy URL:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://<host>:${ROUTER_HOST_PORT:-9000}/v1
   OPENAI_API_KEY=<token>
   ```

## NER-only deployment

`docker/compose.gliner-only.yaml` is a standalone compose project for hosts
that don't run the full vLLM stack — typically because they're on macOS,
have no NVIDIA GPU, or rely on Ollama for chat/embeddings. It runs one
container, `gliner-ner`, built from `Dockerfile.gliner.cpu` (non-CUDA
PyTorch base, multi-arch). No LiteLLM router, no GPU reservation.

Bring it up:

```bash
make network            # if not already created
make volumes            # if not already created
make build-gliner-only     # builds vllm-service-gliner-cpu
make up-gliner-only        # starts the gliner-ner container
```

On first start the container downloads the GLiNER weights to the shared
`huggingface-cache` volume (~1.2 GB for the medium variant). The
healthcheck reports healthy once Ray Serve is accepting requests. If your
host is offline you'll need to pre-populate the cache by temporarily
setting `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in `.env`.

Consumers on `inference-net` reach it directly — there's no router in this
shape:

```bash
curl http://gliner-ner:8000/gliner \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at Acme Corp in Berlin.",
    "labels": ["person", "organization", "location"],
    "threshold": 0.3
  }'
```

No `Authorization` header is required: the container has no built-in
Bearer-token gate, and `inference-net` is a private Docker network shared
only between trusted compose projects (the same posture `data-net` uses
for Qdrant).

Override defaults via `.env` — only `NER_*` knobs apply in this shape:

```bash
NER_MODEL=gliner-community/gliner_medium-v2.5   # default in gliner-only
NER_DEVICE=cpu                                  # default in gliner-only
# NER_MAX_BATCH_SIZE=8
# NER_BATCH_WAIT_TIMEOUT_MS=50
# NER_NUM_REPLICAS=1
```

CPU GLiNER (medium-v2.5) lands around 200 ms – 1 s per request on modern
CPUs. Fine for batch ingestion workloads; not suitable for interactive
per-keystroke use.

## Rerank-only deployment

`docker/compose.rerank-only.yaml` is a standalone compose project for the
same audience as NER-only — hosts without an NVIDIA GPU that still want
to offer rerank to the consuming app. It runs one container,
`rerank-cpu`, built from `Dockerfile.rerank.cpu` (uv-managed Python 3.11,
CPU torch, `transformers`). No LiteLLM router, no GPU reservation. The
container ships a tiny FastAPI server (`docker/rerank_server.py`) that
drives a Hugging Face cross-encoder directly (tokenize → forward →
sigmoid — the same forward pass FlagEmbedding does internally for
bge-reranker-style models, without FlagEmbedding's heavyweight
`ir-datasets`/`zlib-state` dep tree). Exposes the **same Jina-shape
`POST /rerank` contract** as the full-stack vLLM `rerank` service, so
consumers can target either backend by changing only the base URL.

Bring it up:

```bash
make network              # if not already created
make volumes              # if not already created
make build-rerank-only    # builds vllm-service-rerank-cpu
make up-rerank-only       # starts the rerank-cpu container
```

On first start the container downloads the reranker weights to the
shared `huggingface-cache` volume (~570 MB for `BAAI/bge-reranker-v2-m3`,
the default — same model the GPU stack uses, so scores match). The
healthcheck reports healthy once FastAPI is accepting requests. If your
host is offline you'll need to pre-populate the cache by temporarily
setting `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in `.env`.

Consumers on `inference-net` reach it directly — there's no router in
this shape:

```bash
curl http://rerank-cpu:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is RAG?",
    "documents": [
      "Retrieval-augmented generation grounds an LLM in retrieved context.",
      "The cafeteria menu changes daily.",
      "RAG combines a retriever and a generator model."
    ],
    "top_n": 2
  }'
```

Response:

```json
{
  "id": "rerank-...",
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {"index": 0, "relevance_score": 0.96},
    {"index": 2, "relevance_score": 0.91}
  ]
}
```

No `Authorization` header is required: the container has no built-in
Bearer-token gate (same posture as `gliner-ner`).

Override defaults via `.env` — only `RERANK_*` knobs apply in this shape:

```bash
RERANK_MODEL=BAAI/bge-reranker-v2-m3   # default
# RERANK_USE_FP16=true                 # rare on CPU; default false
# RERANK_HOST_PORT=8001                # host publish port for dev
```

CPU rerank of `BAAI/bge-reranker-v2-m3` lands around 50–300 ms per
document on modern CPUs. Fine for typical top-K rerank workloads (K ≤
20); large candidate sets may be noticeably slower than the GPU stack.

> Pair with the NER-only deployment on the same host to give a non-CUDA
> dev box both `/gliner` and `/rerank` against `inference-net`.

## CLIP-only deployment

`docker/compose.clip-only.yaml` is a standalone compose project for the
same audience as NER-only / Rerank-only. It runs one container,
`clip-embed`, built from `Dockerfile.clip.cpu` (uv-managed Python 3.11,
CPU torch, `transformers`, `Pillow`). No LiteLLM router, no GPU
reservation. The container ships `docker/clip_server.py` — a small
FastAPI app that loads the same `CLIPModel` + `AutoProcessor` pair
docint used to load in-process, runs the image or text tower, and
L2-normalizes the output. Exposes the **same `/clip/embed_{image,text}`
contract** as the full-stack `clip` service, so consumers (docint's
`clip_client.py`) target either backend by changing only the base URL.
Default `CLIP_MODEL` is `openai/clip-vit-base-patch32` — same default
docint used in-process, so existing `_images` Qdrant collections stay
compatible.

Bring it up:

```bash
make network            # if not already created
make volumes            # if not already created
make build-clip-only    # builds vllm-service-clip-cpu
make up-clip-only       # starts the clip-embed container
```

On first start the container downloads the CLIP weights to the shared
`huggingface-cache` volume (~600 MB for the base patch32 variant). The
healthcheck reports healthy once FastAPI is accepting requests. If your
host is offline you'll need to pre-populate the cache by temporarily
setting `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in `.env`.

Consumers on `inference-net` reach it directly — there's no router in
this shape:

```bash
# text tower
curl http://clip-embed:8000/clip/embed_text \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'

# image tower (multipart)
curl http://clip-embed:8000/clip/embed_image \
  -F "file=@/path/to/img.jpg"

# dimension probe (for Qdrant collection compat checks)
curl http://clip-embed:8000/clip/dimension
```

Response shape for both embed endpoints:

```json
{"embedding": [0.012, -0.034, ...], "dimension": 512}
```

No `Authorization` header is required (same posture as `gliner-ner` and
`rerank-cpu`).

Override defaults via `.env` — only `CLIP_*` knobs apply in this shape:

```bash
CLIP_MODEL=openai/clip-vit-base-patch32   # default
CLIP_DEVICE=cpu                           # default in clip-only
# CLIP_HOST_PORT=8002                     # host publish port for dev
```

CPU CLIP base-patch32 lands around 80–200 ms per image and ~20–40 ms
per text query on modern CPUs. Fine for the document-image ingestion
workload docint runs; image-search latency is dominated by Qdrant
search, not CLIP inference.

> Pair with the NER-only and Rerank-only deployments on the same host
> to give a non-CUDA dev box `/gliner`, `/rerank`, and `/clip/*`
> against `inference-net`.

## Offline image bundles

For airgapped hosts, customer deployments, or any environment without
Docker Hub access, `make bundle` produces a versioned `.tar.gz` pair you
can ship alongside the `docker/` directory (which holds `compose.yaml` and
`litellm.config.yaml`) and `.env`.

### Producing the bundle

On a build host with internet (`make bundle` follows `PROFILE` from `.env`;
override with `make bundle PROFILE=media`):

```bash
make bundle              # core only (chat, embed, rerank, ner, clip, router)
make bundle PROFILE=media  # core + media (translate, audio)
make bundle-gliner-only     # NER-only shape (just vllm-service-gliner-cpu)
make bundle-rerank-only  # Rerank-only shape (just vllm-service-rerank-cpu)
make bundle-clip-only    # CLIP-only shape (just vllm-service-clip-cpu)
```

This computes `VLLM_SERVICE_VERSION` as `YYYY-MM-DD-<short-sha>` (override by
exporting it before invocation), builds the locally-buildable services with
that version tag, pulls the externally-hosted images (LiteLLM Proxy), then
writes two gzipped tarballs in the cwd:

| File | Contents |
|---|---|
| `vllm-service-built-<profile>-<version>.tar.gz` | Locally-built `vllm-service-{chat,embed,rerank,ner,...}` images. |
| `vllm-service-pulled-<profile>-<version>.tar.gz` | Externally-hosted images (LiteLLM router); re-tagged so the `name:tag@digest` references in `docker/compose.yaml` resolve after `docker load`. |

The compose file references the version through
`image: vllm-service-<svc>:${VLLM_SERVICE_VERSION:-latest}`, so it falls
back to `:latest` for normal dev workflows and uses the pinned tag whenever
the variable is set.

### Loading and running the bundle

Ship the two tarballs along with the matching `docker/` directory (which
holds `compose.yaml` and `litellm.config.yaml`) and a `.env`. Then on the
target host:

```bash
docker load -i vllm-service-built-core-<version>.tar.gz
docker load -i vllm-service-pulled-core-<version>.tar.gz
export VLLM_SERVICE_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml up --no-build -d
```

The target host runs the production shape — `docker/compose.yaml` without
the dev override — so no host ports are published.

The version is embedded in the tarball filenames, so the operator just
reads it off the file. Verify with `docker images | grep vllm-service`
between `load` and `up`.

> `--no-build` does **not** suppress pulls from a registry. If the tagged
> image isn't loaded locally, Compose still tries to resolve it against
> Docker Hub and errors with a DNS / "no such host" failure on offline
> machines. Always `docker load` first.

## Networking

- `vllm-net` is private to this compose project and carries traffic between the
  router and the worker containers.
- `inference-net` is an external shared Docker network used for cross-project
  service discovery and reverse-proxy access.
- Only the `router` service joins `inference-net`; `chat`, `embed`,
  `rerank`, `ner`, `audio`, and `translate` stay on the private network.
- The `router` service keeps its `vllm-router` alias on `inference-net` so
  existing consumers do not need to change their `OPENAI_API_BASE`.

## Updating the model catalog

`docker/litellm.config.yaml` is model-agnostic: all model names are read at
startup from the environment variables `TEXT_MODEL`, `EMBED_MODEL`,
`RERANK_MODEL`, `TRANSLATE_MODEL`, and `WHISPER_MODEL`. To switch a model,
update the relevant variable in `.env` and restart the stack. No changes to
`docker/litellm.config.yaml` are required.

Clients must use the exact model ID set in `.env` as the `model` field in
their requests (e.g. `"model": "BAAI/bge-m3"`). The `/v1/models` endpoint
returns the currently active IDs.

`NER_MODEL` is the exception: GLiNER's server has no OpenAI-shaped endpoint,
so it is not in `model_list` and does not appear in `/v1/models`. Switching
it still works by updating `NER_MODEL` in `.env` and restarting `ner`.

## Calling the audio service

The audio service runs Whisper via vLLM and exposes OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/translations` endpoints.

```bash
curl http://vllm-router:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F model="$WHISPER_MODEL" \
  -F file="@recording.mp3"
```

The maximum accepted file size defaults to 200 MB and can be raised with
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` in `.env`.

The audio service is only started when the `media` profile is active —
set `PROFILE=media` in `.env` or override per-invocation:

```bash
make up PROFILE=media
```

## Calling the translate service

The translate service runs
[`Infomaniak-AI/vllm-translategemma-4b-it`](https://huggingface.co/Infomaniak-AI/vllm-translategemma-4b-it),
a vLLM-compatible repackaging of Google's TranslateGemma 4B. Unlike a general
chat model, it expects the source language, target language, and text to be
encoded in the message content using a delimiter format:

```python
<<<source>>>{iso_src}<<<target>>>{iso_tgt}<<<text>>>{text_to_translate}
```

Language codes are ISO 639-1 (`en`, `de`, `fr`, ...) with optional regional
variants (`en_US`, `zh_CN`). 55 languages are supported. Example request:

```bash
curl http://vllm-router:9000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$TRANSLATE_MODEL"'",
    "messages": [
      {"role": "user", "content": "<<<source>>>en<<<target>>>de<<<text>>>Hello world"}
    ]
  }'
```

The model is trained for a ~2K context window, so keep `TRANSLATE_MAX_MODEL_LEN`
at 2048 unless you have a specific reason to raise it.

The translate service is only started when the `media` profile is active —
set `PROFILE=media` in `.env` or override per-invocation:

```bash
make up PROFILE=media
```

## Calling the NER service

The `ner` service runs [GLiNER](https://github.com/urchade/GLiNER), a
zero-shot Named Entity Recognition model, behind Ray Serve. Unlike the
other backends it is **not** vLLM and does **not** expose OpenAI-compatible
routes — its request/response shape is GLiNER-native, and it is reached
through the router's `/gliner` pass-through:

```bash
curl http://vllm-router:9000/gliner \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at Acme Corp in Berlin.",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5
  }'
```

Response:

```json
{
  "entities": [
    {"start": 0,  "end": 5,  "text": "Alice",     "label": "person",       "score": 0.97},
    {"start": 15, "end": 24, "text": "Acme Corp", "label": "organization", "score": 0.92},
    {"start": 28, "end": 34, "text": "Berlin",    "label": "location",     "score": 0.95}
  ]
}
```

`labels` is the candidate set for this single request — GLiNER is
zero-shot, so labels can change request to request without retraining.

The default model is `gliner-community/gliner_large-v2.5` on CUDA. For
CPU-only hosts, set both `NER_MODEL=gliner-community/gliner_medium-v2.5`
and `NER_DEVICE=cpu` in `.env`. See `.env.example` for the full list of
`NER_*` knobs (dtype, batch size, FlashDeBERTa, sequence packing, etc.).

> Hosts that can't run the CUDA stack at all should use the
> [NER-only deployment](#gliner-only-deployment) instead — same `/gliner`
> request/response shape, but reached at `http://gliner-ner:8000/gliner`
> with no Bearer auth.
