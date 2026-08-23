# CPU-only deployment shapes

For hosts that cannot run the CUDA stack (Mac dev boxes, ROCm or CPU-only
Linux running Ollama for chat/embed), this repo also ships seven standalone
CPU deployments — one container each, no LiteLLM router, no GPU
reservation.

The seven can be co-deployed on the same host so the consuming app has all
of `/gliner`, `/rerank`, `/clip/*`, `/v1/embeddings`, `/pooling`,
`/tokenize`, `/diarize`, `/v1/audio/transcriptions`, and `/vad` available
without the full CUDA stack.

Each is its own compose project (`docker/compose.<shape>.yaml`), not a profile
of the full stack, and each is built and run through its own `make` targets.

## How every shape works

The seven are seven instances of one template. Everything in this section
applies to all of them; the per-shape sections below cover only what differs.

### Bring-up

Substitute the shape's `make` suffix from the table below (`gliner-only`,
`rerank-only`, `clip-only`, `embed-only`, `diarize-only`, `asr-only`,
`vad-only`):

```bash
make network            # if not already created
make volumes            # if not already created
make build-<shape>      # builds the shape's image
make up-<shape>         # starts the container on inference-net
```

For example, the NER-only shape is `make build-gliner-only` then
`make up-gliner-only`; the ASR-only shape is `make build-asr-only` then
`make up-asr-only`.

`make up-<shape>` is the production shape — no host ports. `make
up-dev-<shape>` layers the shape's `docker/compose.<shape>.override.yaml` and
publishes the container's port 8000 on the host for direct testing. Each shape
also has `stop-<shape>`, `down-<shape>`, and `bundle-<shape>` targets; run
`make help` for the full list, and see
[airgap-bundles.md](airgap-bundles.md) for the bundle targets.

### Populating the model cache

Every shape that needs weights expects them to already be present in the shared
`huggingface-cache` volume — `.env.example` ships `HF_HUB_OFFLINE=1`, so
nothing is downloaded at start. To populate the cache on a networked host,
temporarily set `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in `.env` for
the first start, then flip both back.

Three shapes deviate: **diarize-only** needs gated weights and an `HF_TOKEN`
(see [Diarize-only](#diarize-only)), **asr-only** downloads from
openai-whisper's own CDN rather than the HF Hub (see [ASR-only](#asr-only)),
and **vad-only** downloads nothing at all (see [VAD-only](#vad-only)).
Per-shape download sizes are in the table below.

### Auth posture

No `Authorization` header is required: the container has no built-in
Bearer-token gate, and `inference-net` is a private Docker network shared
only between trusted compose projects (the same posture `data-net` uses
for Qdrant).

That holds for all seven — none of them runs a router, so none of them has a
gate. The full stack's master-key gate is described in
[api-reference.md](api-reference.md#authentication).

### Pairing shapes on one host

Shapes share the `huggingface-cache` volume and the `make network` / `make
volumes` prerequisites, so pairing costs nothing beyond the extra
`build-`/`up-` pair. Two pairings matter in practice:

- **diarize-only + vad-only** — the diarizer VAD-gates its output against
  `vad-only` when both are up. Without `vad-only` the gate fails open. See
  [api-reference.md](api-reference.md#diarization).
- **embed-only replaces Ollama's `bge-m3`** rather than sitting alongside it
  on a dev host — see [Embed-only](#embed-only).

### Overriding defaults

Each shape reads only its own prefix from `.env`; the other prefixes are
inert in that shape. The per-shape blocks below list the knobs, and
`.env.example` carries the annotated full set. Knobs with real tuning
semantics — the embed batch budget and the diarize speaker-granularity
weights — are explained in [configuration.md](configuration.md).

## The seven shapes

| Shape | `make` suffix | Image | Exposes, reached at |
|---|---|---|---|
| NER-only | `gliner-only` | `vllm-service-gliner-cpu` | `/gliner` at `http://gliner-only:8000` |
| Rerank-only | `rerank-only` | `vllm-service-rerank-only` | `/rerank` at `http://rerank-only:8000` |
| CLIP-only | `clip-only` | `vllm-service-clip-cpu` | `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension` at `http://clip-only:8000` |
| Embed-only | `embed-only` | `vllm-service-embed-only` | `/v1/embeddings`, `/pooling`, `/tokenize` at `http://embed-only:8000` |
| Diarize-only | `diarize-only` | `vllm-service-diarize-cpu` | `/diarize` at `http://diarize-only:8000` |
| ASR-only | `asr-only` | `vllm-service-asr-cpu` | `/v1/audio/transcriptions`, `/v1/audio/translations` at `http://asr-only:8000` |
| VAD-only | `vad-only` | `vllm-service-vad-cpu` | `/vad` at `http://vad-only:8000` |

Every container listens on port 8000 inside `inference-net`; the compose file
is `docker/compose.<make suffix>.yaml` and the dev overlay
`docker/compose.<make suffix>.override.yaml`.

| Shape | Env prefix | Dev host-port var | Weight source | Gated |
|---|---|---|---|---|
| NER-only | `NER_*` | `NER_HOST_PORT` | HF Hub, ~1.2 GB (medium variant) | no |
| Rerank-only | `RERANK_*` | `RERANK_HOST_PORT` | HF Hub, ~570 MB | no |
| CLIP-only | `CLIP_*` | `CLIP_HOST_PORT` | HF Hub, ~600 MB (base patch32) | no |
| Embed-only | `EMBED_*` | `EMBED_HOST_PORT` | HF Hub (encoder + `sparse_linear.pt`) | no |
| Diarize-only | `DIARIZE_*` | `DIARIZE_HOST_PORT` | HF Hub, ~30 MB segmentation + ~26 MB embedding | **yes** |
| ASR-only | `WHISPER_MODEL`, `ASR_*` | `ASR_HOST_PORT` | openai-whisper CDN, ~3 GB (`large-v3`) | no |
| VAD-only | `VAD_*` | `VAD_HOST_PORT` | bundled in the `silero-vad` package | n/a |

## NER-only

`docker/compose.gliner-only.yaml` is a standalone compose project for hosts
that don't run the full vLLM stack — typically because they're on macOS,
have no NVIDIA GPU, or rely on Ollama for chat/embeddings. It runs one
container, `gliner-only`, built from `Dockerfile.gliner.cpu` (non-CUDA
PyTorch base, multi-arch). No LiteLLM router, no GPU reservation.

Cache population downloads ~1.2 GB for the medium variant. The healthcheck
reports healthy once Ray Serve is accepting requests.

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

Request and response shapes: [api-reference.md](api-reference.md#ner).

## Rerank-only

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

Cache population downloads ~570 MB for `BAAI/bge-reranker-v2-m3`, the default —
same model the GPU stack uses, so scores match. The healthcheck reports healthy
once FastAPI is accepting requests.

Override defaults via `.env` — only `RERANK_*` knobs apply in this shape:

```bash
RERANK_MODEL=BAAI/bge-reranker-v2-m3   # default
# RERANK_USE_FP16=true                 # rare on CPU; default false
# RERANK_HOST_PORT=8001                # host publish port for dev
```

CPU rerank of `BAAI/bge-reranker-v2-m3` lands around 50–300 ms per
document on modern CPUs. Fine for typical top-K rerank workloads (K ≤
20); large candidate sets may be noticeably slower than the GPU stack.

Request and response shapes: [api-reference.md](api-reference.md#rerank).

## CLIP-only

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

Cache population downloads ~600 MB for the base patch32 variant. The
healthcheck reports healthy once FastAPI is accepting requests.

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

Request and response shapes: [api-reference.md](api-reference.md#clip).

## Embed-only

`docker/compose.embed-only.yaml` is a standalone compose project for the
same audience as NER-only / Rerank-only / CLIP-only. It runs one container,
`embed-only`, built from `Dockerfile.embed.cpu` (uv-managed Python 3.11,
CPU torch, `transformers`). No LiteLLM router, no GPU reservation. The
container ships a small FastAPI server (`src/embed_server.py`) that drives
`BAAI/bge-m3` directly with `transformers` rather than `FlagEmbedding`
(whose `ir-datasets`/`zlib-state` dep tree fails to build on aarch64), and
serves **both** dense and sparse embeddings from that one loaded model:
CLS pooling + L2 normalization for dense, and the model's own
`sparse_linear.pt` head + ReLU for sparse. It exposes the **same
`POST /v1/embeddings` (OpenAI-compatible dense), `POST /pooling` (with
`task: "token_classify"`, sparse) and `POST /tokenize` routes** the full
stack's router already passes through to the vLLM `embed` backend, so a
consumer points both its embedding base and its sparse base at the same
`http://embed-only:8000` — no separate deployment needed for each.

Cache population pulls the `BAAI/bge-m3` weights — the encoder plus the
`sparse_linear.pt` head. It is the same model
and pooling definition the GPU stack uses, so scores are equivalent up
to dtype: this server runs float32, while vLLM's `dtype="auto"` casts
bge-m3's float32 checkpoint to float16, so the two diverge by roughly
1e-3. Parity against the CUDA stack was verified side-by-side on
2026-08-04 — see
[2026-08-04-embed-parity-checklist.md](2026-08-04-embed-parity-checklist.md),
the golden fixtures in `eval/fixtures/embed_parity/`, and
`eval/tests/test_embed_parity.py`, which re-runs the four comparisons
against any live backend. The healthcheck reports healthy once FastAPI is
accepting requests against `/health`.

On a dev host running Ollama for chat/embed, this shape **replaces**
Ollama's `bge-m3` rather than sitting alongside it: point the embedding
consumer at `embed-only` instead, and Ollama then serves chat only — the
model is loaded once (here) instead of twice (once in Ollama, once in this
container).

`EMBED_*` knobs, including the batch-token budget that bounds one forward
pass, are documented in
[configuration.md](configuration.md#embedding-batch-budget).

Request and response shapes: [api-reference.md](api-reference.md#embeddings).

## Diarize-only

`docker/compose.diarize-only.yaml` is a standalone compose project for the
same audience as NER-only / Rerank-only / CLIP-only. It runs one container,
`diarize-only`, built from `Dockerfile.diarize.cpu` (uv-managed Python 3.11,
CPU torch + torchaudio, `pyannote.audio`, `ffmpeg`). No LiteLLM router, no
GPU reservation. The container ships `src/diarize_server.py` — the same
FastAPI app the full-stack `diarize` service runs — so it exposes the
**same multipart `/diarize` contract**, and consumers (Nextext) target
either backend by changing only the base URL. Default `DIARIZE_MODEL` is
`pyannote/speaker-diarization-community-1` (pyannote.audio 4.x); the 3.1
pipeline still loads if you configure it.

The pyannote weights are **gated** on the Hugging Face Hub, so unlike the
other CPU shapes the cache cannot be populated anonymously. One-time setup:

1. Accept the access conditions with your Hugging Face account for the
   pipeline you are running:
   - the default,
     [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
     — its exact gated dependency set is still being confirmed;
   - or, if you have configured the 3.1 pipeline instead, both
     [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
     and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0).
2. In `.env`, set `HF_TOKEN=hf_...`, `HF_HUB_OFFLINE=0`, and
   `TRANSFORMERS_OFFLINE=0`, then start the container once so it downloads
   the weights (~30 MB segmentation + ~26 MB embedding) into the shared
   `huggingface-cache` volume.
3. Revert `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Subsequent starts
   serve from the cache with no network access.

`DIARIZE_*` knobs, including the `DIARIZE_FB`/`DIARIZE_FA` speaker-granularity
weights and how to validate a change, are documented in
[configuration.md](configuration.md#diarization-speaker-granularity).

In this shape the compose file hardcodes `DIARIZE_VAD_URL` to
`http://vad-only:8000` — co-deploy `vad-only` and VAD gating engages; without
it the gate fails open. See
[api-reference.md](api-reference.md#diarization).

CPU diarization is the slowest of the standalone shapes — expect roughly
real-time-to-several-times-real-time per audio minute, dominated by the
segmentation and embedding passes. Fine for batch transcription pipelines
(Nextext's workload); not suitable for interactive use.

Request and response shapes: [api-reference.md](api-reference.md#diarization).

## ASR-only

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

On first start the container downloads the Whisper weights into the shared
`huggingface-cache` volume (~3 GB for `large-v3`; openai-whisper fetches from
its own CDN, not the HF Hub — the weights are public, no gated access). The
healthcheck reports healthy once FastAPI is accepting requests. If your host is
offline, pre-populate the cache on a networked machine first. For a quick CPU
smoke test, set a smaller model such as `WHISPER_MODEL=openai/whisper-base` —
`large-v3` on CPU is very slow.

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

Request and response shapes:
[api-reference.md](api-reference.md#speech-to-text-asr).

## VAD-only

`docker/compose.vad-only.yaml` is a standalone compose project for the same
audience as the other `-only` shapes. It runs one container, `vad-only`, built
from `Dockerfile.vad.cpu` (uv-managed Python 3.11, CPU torch + torchaudio,
`silero-vad`, `ffmpeg`). No LiteLLM router, no GPU reservation. The container
ships `src/vad_server.py` — the same FastAPI app the full-stack `vad` service
runs — so it exposes the **same multipart `/vad` contract**, and consumers
target either backend by changing only the base URL.

Unlike diarize-only, the `silero-vad` package bundles its model weights, so
**nothing is downloaded** — this shape works fully offline on first start.

Override defaults via `.env` — only `VAD_*` knobs apply in this shape:

```bash
VAD_MODEL=silero_vad     # default (informational; one bundled model)
VAD_DEVICE=cpu           # default in vad-only
# VAD_USE_ONNX=false      # true runs the bundled ONNX graph
# VAD_HOST_PORT=8006       # host publish port for dev
```

Silero VAD is by far the fastest of the standalone shapes — a fraction of real
time per audio minute on CPU.

Request and response shapes: [api-reference.md](api-reference.md#vad).
