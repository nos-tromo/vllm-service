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
- `/diarize` (speaker diarization; non-OpenAI shape)
- `/vad` (Silero voice activity detection; non-OpenAI shape)

Internally it runs:

- `router` (LiteLLM Proxy)
- `chat`
- `embed`
- `rerank`
- `gliner` (GLiNER, served via Ray Serve rather than vLLM)

  The gliner container is supervised by `scripts/gliner_watchdog.sh` (installed as
  `/usr/local/bin/gliner-watchdog`): it probes `POST /gliner` and self-exits after
  `NER_WATCHDOG_FAILURES` consecutive failures so `restart: unless-stopped` recreates
  the container, recovering from the Ray Serve rank-consistency wedge
  (ray-project/ray#63862). The Docker healthcheck uses the same functional probe;
  set `NER_WATCHDOG_ENABLED=false` to disable. Knobs: `NER_WATCHDOG_*` in `.env.example`.

- `clip` (CLIP image+text tower, served via FastAPI rather than vLLM)
- `asr` (Whisper ASR, served via vLLM)
- `diarize` (pyannote speaker diarization, served via FastAPI rather than vLLM)
- `vad` (Silero voice activity detection, served via FastAPI rather than vLLM)

Model-to-backend routing is declared in `docker/litellm.config.yaml`. Clients
select a backend purely by the `model` field they send; there is no path-based
dispatch.

For hosts that cannot run the CUDA stack (Mac dev boxes, ROCm or CPU-only
Linux running Ollama for chat/embed), this repo also ships six standalone
CPU deployments:

- **NER-only** (`docker/compose.gliner-only.yaml`) — a single `gliner-only`
  container exposing `/gliner`. See "NER-only deployment" below.
- **Rerank-only** (`docker/compose.rerank-only.yaml`) — a single
  `rerank-only` container exposing the same Jina-shape `/rerank` contract
  as the full stack. See "Rerank-only deployment" below.
- **CLIP-only** (`docker/compose.clip-only.yaml`) — a single `clip-only`
  container exposing the same `/clip/embed_{image,text}` contract as the
  full stack. See "CLIP-only deployment" below.
- **Diarize-only** (`docker/compose.diarize-only.yaml`) — a single
  `diarize-only` container exposing the same multipart `/diarize` contract
  as the full stack. See "Diarize-only deployment" below.
- **ASR-only** (`docker/compose.asr-only.yaml`) — a single `asr-only`
  container exposing the same OpenAI `/v1/audio/transcriptions` contract as
  the full stack (CPU openai-whisper instead of vLLM). See "ASR-only
  deployment" below.
- **VAD-only** (`docker/compose.vad-only.yaml`) — a single `vad-only`
  container exposing the same multipart `/vad` contract as the full stack.
  See "VAD-only deployment" below.

The six can be co-deployed on the same host so the consuming app has all of
`/gliner`, `/rerank`, `/clip/*`, `/diarize`, `/v1/audio/transcriptions`, and
`/vad` available without the full CUDA stack.

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
   itself supports tool use. `CHAT_REASONING_PARSER` is optional — enable it
   only if you want the model's thinking traces, as it can interfere with
   tool-call parsing.
2. Ensure the external `huggingface-cache` Docker volume exists.
3. Initialize the shared proxy network and persistent model cache:

   ```bash
   make network   # create the external inference-net
   make volume    # create the huggingface-cache Docker volume
   ```

4. Build and start the stack:

   ```bash
   make build                 # build images for the full stack
   make up-dev                # full stack with the router published on the host (router, chat, embed, rerank, clip, asr, diarize, vad, gliner)
   make up                    # same as up-dev but production shape (no host ports)
   make dev                   # build, then up-dev (dev convenience)
   ```

   `make up` and `make up-dev` are detached and never build (`up -d
   --no-build`) — run `make build` first, or use `make dev` (build + up-dev).
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
container, `gliner-only`, built from `Dockerfile.gliner.cpu` (non-CUDA
PyTorch base, multi-arch). No LiteLLM router, no GPU reservation.

Bring it up:

```bash
make network            # if not already created
make volumes            # if not already created
make build-gliner-only     # builds vllm-service-gliner-cpu
make up-gliner-only        # starts the gliner-only container
```

On first start the container downloads the GLiNER weights to the shared
`huggingface-cache` volume (~1.2 GB for the medium variant). The
healthcheck reports healthy once Ray Serve is accepting requests. If your
host is offline you'll need to pre-populate the cache by temporarily
setting `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in `.env`.

Consumers on `inference-net` reach it directly — there's no router in this
shape:

```bash
curl http://gliner-only:8000/gliner \
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
`rerank-only`, built from `Dockerfile.rerank.cpu` (uv-managed Python 3.11,
CPU torch, `transformers`). No LiteLLM router, no GPU reservation. The
container ships a tiny FastAPI server (`src/rerank_server.py`) that
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
make build-rerank-only    # builds vllm-service-rerank-only
make up-rerank-only       # starts the rerank-only container
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
curl http://rerank-only:8000/rerank \
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
Bearer-token gate (same posture as `gliner-only`).

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
`clip-only`, built from `Dockerfile.clip.cpu` (uv-managed Python 3.11,
CPU torch, `transformers`, `Pillow`). No LiteLLM router, no GPU
reservation. The container ships `src/clip_server.py` — a small
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
make up-clip-only       # starts the clip-only container
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
curl http://clip-only:8000/clip/embed_text \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'

# image tower (multipart)
curl http://clip-only:8000/clip/embed_image \
  -F "file=@/path/to/img.jpg"

# dimension probe (for Qdrant collection compat checks)
curl http://clip-only:8000/clip/dimension
```

Response shape for both embed endpoints:

```json
{"embedding": [0.012, -0.034, ...], "dimension": 512}
```

No `Authorization` header is required (same posture as `gliner-only` and
`rerank-only`).

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

## Diarize-only deployment

`docker/compose.diarize-only.yaml` is a standalone compose project for the
same audience as NER-only / Rerank-only / CLIP-only. It runs one container,
`diarize-only`, built from `Dockerfile.diarize.cpu` (uv-managed Python 3.11,
CPU torch + torchaudio, `pyannote.audio`, `ffmpeg`). No LiteLLM router, no
GPU reservation. The container ships `src/diarize_server.py` — the same
FastAPI app the full-stack `diarize` service runs — so it exposes the
**same multipart `/diarize` contract**, and consumers (Nextext) target
either backend by changing only the base URL. Default `DIARIZE_MODEL` is
`pyannote/speaker-diarization-3.1`.

Bring it up:

```bash
make network              # if not already created
make volumes              # if not already created
make build-diarize-only   # builds vllm-service-diarize-cpu
make up-diarize-only      # starts the diarize-only container
```

The pyannote weights are **gated** on the Hugging Face Hub, so unlike the
other CPU shapes the cache cannot be populated anonymously. One-time setup:

1. Accept the access conditions for both
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
   with your Hugging Face account.
2. In `.env`, set `HF_TOKEN=hf_...`, `HF_HUB_OFFLINE=0`, and
   `TRANSFORMERS_OFFLINE=0`, then start the container once so it downloads
   the weights (~30 MB segmentation + ~26 MB embedding) into the shared
   `huggingface-cache` volume.
3. Revert `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Subsequent starts
   serve from the cache with no network access.

Consumers on `inference-net` reach it directly — there's no router in
this shape:

```bash
curl http://diarize-only:8000/diarize \
  -F "file=@recording.mp3" \
  -F "max_speakers=4"
```

Response:

```json
{
  "segments": [
    {"start": 0.51, "end": 4.32, "speaker": "SPEAKER_00"},
    {"start": 4.80, "end": 9.11, "speaker": "SPEAKER_01"}
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

No `Authorization` header is required (same posture as `gliner-only`,
`rerank-only`, and `clip-only`). `num_speakers` (exact count) is mutually
exclusive with the `min_speakers`/`max_speakers` bounds; sending both
returns 400.

Override defaults via `.env` — only `DIARIZE_*` knobs apply in this shape:

```bash
DIARIZE_MODEL=pyannote/speaker-diarization-community-1   # default (pyannote.audio 4.x; gated)
DIARIZE_DEVICE=cpu                               # default in diarize-only
# DIARIZE_HOST_PORT=8004                          # host publish port for dev
# DIARIZE_FB=0.4                                  # community-1 clustering: LOWER → MORE speakers; 0.4 validated best
# DIARIZE_FA=0.07                                 # community-1 PLDA companion weight (leave at stock 0.07)
# DIARIZE_CLUSTERING_THRESHOLD / DIARIZE_SEG_MIN_DURATION_OFF   # further overrides (threshold inert for community-1)
```

`DIARIZE_FB`/`DIARIZE_FA` are community-1's speaker-granularity knobs (unset →
stock defaults, Fb 0.8 / Fa 0.07). `Fb=0.4` is the deploy value — best
speaker-attribution accuracy and turn precision on real labeled clips
(`eval/reports/2026-07-14-fb-realdata-validation.md`; the benchmark sweep's
provisional 0.2 over-splits real content — see
`eval/reports/2026-07-14-fa-fb-sweep.md`). Leave `Fa` at stock — both
directions measured worse. Validate on labelled clips with `eval/` (the
transcript metric + `--fb` sweep) before deploying a different value; an
unparseable value warns and is ignored rather than crashing startup.

CPU diarization is the slowest of the standalone shapes — expect roughly
real-time-to-several-times-real-time per audio minute, dominated by the
segmentation and embedding passes. Fine for batch transcription pipelines
(Nextext's workload); not suitable for interactive use.

> Pair with the NER-only, Rerank-only, CLIP-only, ASR-only, and VAD-only
> deployments on the same host to give a non-CUDA dev box `/gliner`,
> `/rerank`, `/clip/*`, `/diarize`, `/v1/audio/transcriptions`, and `/vad`
> against `inference-net`.

## ASR-only deployment

`docker/compose.asr-only.yaml` is a standalone compose project for the same
audience as NER-only / Rerank-only / CLIP-only / Diarize-only. It runs one
container, `asr-only`, built from `Dockerfile.asr.cpu` (uv-managed Python 3.11,
CPU torch, `openai-whisper`, `ffmpeg`). No LiteLLM router, no GPU reservation.

Unlike the full-stack `asr` service (Whisper on vLLM, CUDA-only), this shape
ships `src/asr_server.py` — a small FastAPI app that drives the reference
**openai-whisper** decoder (the same one Nextext runs in-process) — so it
exposes the **same OpenAI `/v1/audio/transcriptions` (and
`/v1/audio/translations`) contract** on CPU, and consumers target either
backend by changing only the base URL. Default `WHISPER_MODEL` is
`openai/whisper-large-v3`, mapped to the openai-whisper checkpoint name
(`large-v3`) at load.

Bring it up:

```bash
make network            # if not already created
make volumes            # if not already created
make build-asr-only     # builds vllm-service-asr-cpu
make up-asr-only        # starts the asr-only container
```

On first start the container downloads the Whisper weights into the shared
`huggingface-cache` volume (~3 GB for `large-v3`; openai-whisper fetches from
its own CDN, not the HF Hub — the weights are public, no gated access). The
healthcheck reports healthy once FastAPI is accepting requests. If your host is
offline, pre-populate the cache on a networked machine first. For a quick CPU
smoke test, set a smaller model such as `WHISPER_MODEL=openai/whisper-base` —
`large-v3` on CPU is very slow.

Consumers on `inference-net` reach it directly — there's no router in this
shape:

```bash
curl http://asr-only:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "response_format=verbose_json"
```

Response (`verbose_json`):

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.34,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "...", "no_speech_prob": 0.01}
  ]
}
```

No `Authorization` header is required: the container has no built-in
Bearer-token gate (same posture as the other `-only` shapes). The `model` form
field is accepted but ignored — the server always uses the model it loaded at
boot.

Override defaults via `.env` — only `WHISPER_MODEL` / `ASR_*` knobs apply in
this shape:

```bash
WHISPER_MODEL=openai/whisper-large-v3   # default
ASR_DEVICE=cpu                          # default in asr-only
# ASR_HOST_PORT=8005                     # host publish port for dev
```

CPU openai-whisper is the slowest of the standalone shapes for large models —
`large-v3` runs several times slower than real time; smaller checkpoints
(`base`, `small`) are far quicker. Fine for batch transcription; not suitable
for interactive use.

> Pair with the NER-only, Rerank-only, CLIP-only, and VAD-only deployments on
> the same host to give a non-CUDA dev box `/gliner`, `/rerank`, `/clip/*`,
> `/v1/audio/transcriptions`, and `/vad` against `inference-net`.

## VAD-only deployment

`docker/compose.vad-only.yaml` is a standalone compose project for the same
audience as the other `-only` shapes. It runs one container, `vad-only`, built
from `Dockerfile.vad.cpu` (uv-managed Python 3.11, CPU torch + torchaudio,
`silero-vad`, `ffmpeg`). No LiteLLM router, no GPU reservation. The container
ships `src/vad_server.py` — the same FastAPI app the full-stack `vad` service
runs — so it exposes the **same multipart `/vad` contract**, and consumers
target either backend by changing only the base URL.

Unlike diarize-only, the `silero-vad` package bundles its model weights, so
**nothing is downloaded** — this shape works fully offline on first start.

Bring it up:

```bash
make network            # if not already created
make volumes            # if not already created
make build-vad-only     # builds vllm-service-vad-cpu
make up-vad-only        # starts the vad-only container
```

Consumers on `inference-net` reach it directly — there's no router in this
shape:

```bash
curl http://vad-only:8000/vad \
  -F "file=@recording.mp3"
```

Response:

```json
{
  "segments": [
    {"start": 0.51, "end": 4.32},
    {"start": 4.80, "end": 9.11}
  ],
  "has_speech": true,
  "sampling_rate": 16000
}
```

Times are absolute seconds. Like `/diarize`, the service returns raw speech
turns — consumers reduce them (e.g. to a speech / no-speech gate) client-side.
No `Authorization` header is required (same posture as the other `-only`
shapes). Optional `threshold`, `min_speech_duration_ms`,
`min_silence_duration_ms`, `speech_pad_ms`, and `max_speech_duration_s` form
fields tune Silero; omitted ones use its defaults.

Override defaults via `.env` — only `VAD_*` knobs apply in this shape:

```bash
VAD_MODEL=silero_vad     # default (informational; one bundled model)
VAD_DEVICE=cpu           # default in vad-only
# VAD_USE_ONNX=false      # true runs the bundled ONNX graph
# VAD_HOST_PORT=8006       # host publish port for dev
```

Silero VAD is by far the fastest of the standalone shapes — a fraction of real
time per audio minute on CPU.

> Pair with the NER-only, Rerank-only, CLIP-only, and ASR-only deployments on
> the same host to give a non-CUDA dev box the full set against `inference-net`.

## Offline image bundles

For airgapped hosts, customer deployments, or any environment without
Docker Hub access, `make bundle` produces a versioned `.tar.gz` pair you
can ship alongside the `docker/` directory (which holds `compose.yaml` and
`litellm.config.yaml`) and `.env`.

### Producing the bundle

On a build host with internet:

```bash
make bundle              # full stack (chat, embed, rerank, gliner, clip, asr, diarize, vad, router)
make bundle-gliner-only     # NER-only shape (just vllm-service-gliner-cpu)
make bundle-rerank-only  # Rerank-only shape (just vllm-service-rerank-only)
make bundle-clip-only    # CLIP-only shape (just vllm-service-clip-cpu)
make bundle-diarize-only # Diarize-only shape (just vllm-service-diarize-cpu)
make bundle-asr-only     # ASR-only shape (just vllm-service-asr-cpu)
make bundle-vad-only     # VAD-only shape (just vllm-service-vad-cpu)
```

This computes `VLLM_SERVICE_VERSION` as `YYYY-MM-DD-<short-sha>` (override by
exporting it before invocation), builds the locally-buildable services with
that version tag, pulls the externally-hosted images (LiteLLM Proxy), then
writes two gzipped tarballs in the cwd:

| File | Contents |
|---|---|
| `vllm-service-built-<profile>-<version>.tar.gz` | Locally-built `vllm-service-{chat,embed,rerank,gliner,...}` images. |
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
  `rerank`, `gliner`, `clip`, `asr`, `diarize`, and `vad` stay on the private
  network.
- The `router` service keeps its `vllm-router` alias on `inference-net` so
  existing consumers do not need to change their `OPENAI_API_BASE`.

## Updating the model catalog

`docker/litellm.config.yaml` is model-agnostic: all model names are read at
startup from the environment variables `TEXT_MODEL`, `EMBED_MODEL`,
`RERANK_MODEL`, and `WHISPER_MODEL`. To switch a model,
update the relevant variable in `.env` and restart the stack. No changes to
`docker/litellm.config.yaml` are required.

Clients must use the exact model ID set in `.env` as the `model` field in
their requests (e.g. `"model": "BAAI/bge-m3"`). The `/v1/models` endpoint
returns the currently active IDs.

`NER_MODEL`, `CLIP_MODEL`, `DIARIZE_MODEL`, and `VAD_MODEL` are the exceptions:
their servers have no OpenAI-shaped endpoints, so they are not in `model_list`
and do not appear in `/v1/models`. Switching them still works by updating the
variable in `.env` and restarting the matching service (`gliner`, `clip`,
`diarize`, `vad`).

## Calling the ASR service

The `asr` service runs Whisper via vLLM and exposes OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/translations` endpoints.

```bash
curl http://vllm-router:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F model="$WHISPER_MODEL" \
  -F file="@recording.mp3"
```

The maximum accepted file size defaults to 200 MB and can be raised with
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` in `.env`.

> Hosts that can't run the CUDA stack at all should use the
> [ASR-only deployment](#asr-only-deployment) instead — same
> `/v1/audio/transcriptions` request/response shape, but reached at
> `http://asr-only:8000/v1/audio/transcriptions` with no Bearer auth (CPU
> openai-whisper instead of vLLM).

## Calling the diarization service

The `diarize` service runs the
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
pipeline behind FastAPI. Like `gliner` and `clip` it is **not** vLLM and
does **not** expose OpenAI-compatible routes — it is reached through the
router's `/diarize` pass-through. The uploaded file may be any container
ffmpeg can decode (wav, mp3, m4a, mp4, ...); it is resampled to 16 kHz
mono server-side.

```bash
curl http://vllm-router:9000/diarize \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@recording.mp3" \
  -F "max_speakers=4"
```

Response:

```json
{
  "segments": [
    {"start": 0.51, "end": 4.32, "speaker": "SPEAKER_00"},
    {"start": 4.80, "end": 9.11, "speaker": "SPEAKER_01"}
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

`num_speakers` (exact count) is mutually exclusive with the
`min_speakers`/`max_speakers` bounds; sending both returns 400. Times are
absolute seconds — consumers (Nextext) assign speakers to their ASR
segments by maximum overlap client-side.

The pipeline weights are gated on the Hugging Face Hub: accept the
conditions for both `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0`, then run once with `HF_HUB_OFFLINE=0`,
`TRANSFORMERS_OFFLINE=0`, and `HF_TOKEN` set in `.env` to populate the
shared `huggingface-cache` volume. The default offline mode serves from
the cache afterwards.

> Hosts that can't run the CUDA stack at all should use the
> [Diarize-only deployment](#diarize-only-deployment) instead — same
> `/diarize` request/response shape, but reached at
> `http://diarize-only:8000/diarize` with no Bearer auth.

## Calling the VAD service

The `vad` service runs [Silero VAD](https://github.com/snakers4/silero-vad)
behind FastAPI. Like `gliner`, `clip`, and `diarize` it is **not** vLLM and
does **not** expose OpenAI-compatible routes — it is reached through the
router's `/vad` pass-through. The uploaded file may be any container ffmpeg can
decode; it is resampled to 16 kHz mono server-side.

```bash
curl http://vllm-router:9000/vad \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@recording.mp3"
```

Response:

```json
{
  "segments": [
    {"start": 0.51, "end": 4.32},
    {"start": 4.80, "end": 9.11}
  ],
  "has_speech": true,
  "sampling_rate": 16000
}
```

Times are absolute seconds; the service returns raw speech turns, leaving the
speech / no-speech reduction to the consumer. Optional `threshold`,
`min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, and
`max_speech_duration_s` form fields tune Silero; omitted ones use its defaults.
The Silero weights ship inside the `silero-vad` package, so no download or HF
token is needed.

> Hosts that can't run the CUDA stack at all should use the
> [VAD-only deployment](#vad-only-deployment) instead — same `/vad`
> request/response shape, but reached at `http://vad-only:8000/vad` with no
> Bearer auth.

## Calling the NER service

The `gliner` service runs [GLiNER](https://github.com/urchade/GLiNER), a
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
> request/response shape, but reached at `http://gliner-only:8000/gliner`
> with no Bearer auth.

## Linting the Python servers

The FastAPI servers in `src/` are linted with the nos-tromo org-wide strict
regime — `ruff` + `pyrefly (strict)` via
[`.pre-commit-config.yaml`](.pre-commit-config.yaml), mirroring the canonical
config in [`nos-tromo/.github`](https://github.com/nos-tromo/.github)
`configs/python-strict/`. The `python-lint` CI job (in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same hooks and
additionally runs `validate_strict_config.py`, which fails the build if this
repo's `pyproject.toml` / `.pre-commit-config.yaml` drift from the canonical
config.

Run it locally with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                              # create the lint venv (pyrefly, pre-commit, typed deps)
uv run pre-commit run --all-files    # ruff check + ruff format + pyrefly over src/
```

The heavy ML backends (torch, transformers, openai-whisper, pyannote, silero)
are **not** installed for linting — `pyrefly` treats them as `Any`
(`ignore-missing-imports`). Only the light, typed shared deps (`fastapi`,
`pydantic`, `numpy`) are installed, so strict mode type-checks the first-party
code.
