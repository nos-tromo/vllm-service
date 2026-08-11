# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Data confidentiality — hard rule

**NEVER expose actual production or testing data in any file committed or
pushed to git.** This covers not only file contents but also metadata that
references real data: filenames, file descriptions, social-media account
names or handles, user identifications, sample records, log excerpts, and
screenshots. It applies everywhere git sees — source code, tests, fixtures,
docs, examples, configs, commit messages, and CI files. Use fully synthetic,
invented placeholders instead.

**Likewise, NEVER expose local filepaths from development machines** —
absolute paths or home directories such as `/Users/<name>/...`,
`/home/<name>/...`, or `C:\Users\...` — anywhere git sees. The only
permitted paths are relative project paths starting from the project's
root (e.g. `docker/compose.yaml`).

## What this repo is

A pure infrastructure repo: a Docker Compose stack that fronts several vLLM
backends with a single LiteLLM Proxy router. The Docker assets live under
`docker/` (`docker/compose.yaml`, `docker/compose.override.yaml`,
`docker/compose.gliner-only.yaml`, `docker/compose.rerank-only.yaml`,
`docker/compose.clip-only.yaml`, `docker/compose.embed-only.yaml`,
`docker/compose.diarize-only.yaml`,
`docker/compose.asr-only.yaml`, `docker/compose.vad-only.yaml`,
`docker/Dockerfile.vllm`,
`docker/Dockerfile.gliner.cuda`, `docker/Dockerfile.gliner.cpu`,
`docker/Dockerfile.rerank.cpu`, `docker/Dockerfile.clip.cuda`,
`docker/Dockerfile.clip.cpu`, `docker/Dockerfile.embed.cpu`,
`docker/Dockerfile.diarize.cuda`,
`docker/Dockerfile.diarize.cpu`, `docker/Dockerfile.asr.cpu`,
`docker/Dockerfile.vad.cpu`, `docker/litellm.config.yaml`) plus `.env`
and `.dockerignore` at the repo root. The only Python sources are
`src/rerank_server.py`, `src/clip_server.py`, `src/embed_server.py`,
`src/diarize_server.py` (and its `src/diarize_audio.py`,
`src/diarize_pipeline.py`, `src/diarize_compat.py`, and
`src/diarize_gate.py` helpers),
`src/asr_server.py`, and `src/vad_server.py` —
small FastAPI wrappers around Hugging Face models that ship because there
is no off-the-shelf server that speaks the Jina-shape `/rerank`, `/clip`,
`/pooling`/`/tokenize`, `/diarize`, or `/vad` contracts the full stack
exposes. (`asr_server.py` is
the exception — it speaks the standard OpenAI `/v1/audio/transcriptions`
contract, but exists to serve Whisper on CPU via openai-whisper where the
full stack uses vLLM.) (`diarize_compat.py`
holds the two pyannote-vs-base-image compat shims, applied before
pyannote is imported: it restores the handful of `torchaudio` symbols
that torchaudio 2.9+ removed (each stub is `hasattr`-guarded, so it's a
no-op where pyannote no longer needs them — the server decodes audio via
ffmpeg and never uses torchaudio's file I/O) — and it
allowlists the trusted checkpoint globals (`TRUSTED_CHECKPOINT_GLOBALS`)
so PyTorch 2.6+'s `weights_only=True` `torch.load` can load the gated
weights. Both diarize Dockerfiles' build smoke tests round-trip those
globals so a base-image bump that breaks either shim fails the build.)

## Container hardening (deploy ADR 0001)

Every first-party image runs as the non-root `app` user (uid `10001`,
`HOME=/home/app`); the router uses upstream's `litellm-non_root` image
variant. The shared `huggingface-cache` volume is therefore mounted at
**`/home/app/.cache/huggingface/hub`** (was `/root/...`) — `HF_HUB_CACHE`
is baked into every image and set in `x-vllm-env` as belt-and-braces, and
`ASR_DOWNLOAD_ROOT`/`PYANNOTE_CACHE` follow the same path. Compose applies
`no-new-privileges` + `cap_drop: ALL` everywhere; most services also run
`read_only` with a `/tmp` tmpfs (2g on the audio services — they stage
uploads in temp files). Two documented deferrals keep a writable rootfs:
`chat` (vLLM's torch-compile cache under `~/.cache/vllm`) and `gliner`
(Ray session dirs under `/tmp/ray`).

**Migration (destructive if skipped):** on existing hosts the
`huggingface-cache` volume holds root-owned model weights — a one-time
`chown -R 10001:10001` is required before the first hardened start, with a
snapshot first on airgapped hosts: `make migrate-cache` does snapshot +
chown + verify (runbook: `docs/hardening-migration.md` in the `deploy`
repo). The tell-tale of a skipped migration is vLLM starting but spamming
"Ignoring corrupted tree cache file ... Permission denied", then the
pooling backends dying in EngineCore — not a clear ownership error. The same
uid is shared with docint's mounts of this volume, so one chown serves both.

## Deployment shapes

Eight independent compose projects, picked per host:

- **Full stack** (`docker/compose.yaml`, CUDA-required) — chat, embed, rerank,
  clip, asr, diarize, vad, gliner, router. The original shape; reached as
  `http://vllm-router:4000/...` on `inference-net`. GLiNER is routed via the
  router's `/gliner` pass-through, diarization via `/diarize`, voice activity
  detection via `/vad`. (`vad` is a tiny CPU Silero service even in the full
  stack — Silero gains nothing from CUDA.)

  The gliner container is supervised by `scripts/gliner_watchdog.sh` (installed as
  `/usr/local/bin/gliner-watchdog`): it probes `POST /gliner` and self-exits after
  `NER_WATCHDOG_FAILURES` consecutive failures so `restart: unless-stopped` recreates
  the container, recovering from the Ray Serve rank-consistency wedge
  (ray-project/ray#63862). The Docker healthcheck uses the same functional probe;
  set `NER_WATCHDOG_ENABLED=false` to disable. Knobs: `NER_WATCHDOG_*` in `.env.example`.
- **NER-only** (`docker/compose.gliner-only.yaml`, CPU OK) — a single
  `gliner-only` container on `inference-net`, no router, no GPU requirement.
  Intended for hosts that run Ollama (or another non-vLLM provider) for
  chat/embeddings but still want NER out of the consuming app. Reached
  directly as `http://gliner-only:8000/gliner` on `inference-net`; there is
  no Bearer auth (trust `inference-net` the way `data-net` is trusted for
  Qdrant). Uses the CPU-only `Dockerfile.gliner.cpu` (non-CUDA PyTorch
  base, multi-arch) and defaults `NER_MODEL` to `gliner_medium-v2.5`.
- **Rerank-only** (`docker/compose.rerank-only.yaml`, CPU OK) — a single
  `rerank-only` container on `inference-net`, no router, no GPU. Same
  audience as NER-only; pairs with it so an Ollama-on-CPU host can offer
  both `/gliner` and `/rerank` without the full CUDA stack. Reached as
  `http://rerank-only:8000/rerank` on `inference-net`. Speaks the same
  Jina-shape `{model, query, documents, top_n}` → `{results: [{index,
  relevance_score}]}` contract as the full-stack vLLM `rerank` service,
  so consumers (docint's `VLLMRerankPostprocessor`) target either
  backend by changing the base URL alone. Uses `Dockerfile.rerank.cpu`
  (uv-managed Python 3.11, CPU torch, transformers) and ships a tiny
  FastAPI server at `src/rerank_server.py` that drives the cross-
  encoder directly (tokenize → forward → sigmoid).
- **CLIP-only** (`docker/compose.clip-only.yaml`, CPU OK) — a single
  `clip-only` container on `inference-net`, no router, no GPU. Same
  audience as NER-only / Rerank-only; co-deployable so a non-CUDA host
  can offer `/gliner`, `/rerank`, and `/clip/embed_{image,text}` at
  once. Reached as `http://clip-only:8000/clip/*`. Speaks the same
  contract the full-stack `clip` service exposes (also a FastAPI app
  on the same Python file), so docint's image-ingestion path
  (`docint/utils/clip_client.py`) targets either backend by changing
  the base URL alone. Uses `Dockerfile.clip.cpu` (uv-managed
  Python 3.11, CPU torch, transformers, Pillow) and ships
  `src/clip_server.py`.
- **Embed-only** (`docker/compose.embed-only.yaml`, CPU OK) — a single
  `embed-only` container on `inference-net`, no router, no GPU. Same
  audience as NER-only / Rerank-only / CLIP-only; co-deployable so a
  non-CUDA host can offer `/gliner`, `/rerank`, `/clip/*`, and
  dense + sparse embeddings at once. Reached as `http://embed-only:8000`.
  Serves **all three** of the full stack's `embed`-backend contracts from
  one loaded `BAAI/bge-m3`: `POST /v1/embeddings` (OpenAI-compatible dense
  embeddings), `POST /pooling` (`task: "token_classify"`, the learned
  sparse/lexical weights), and `POST /tokenize` — so consumers (docint's
  dense and sparse encoders) target either backend by changing the base
  URL alone, and both a consumer's embedding base and its sparse base can
  point at this same container. Uses `Dockerfile.embed.cpu` (uv-managed
  Python 3.11, CPU torch, transformers) and ships `src/embed_server.py`,
  which drives `BAAI/bge-m3` directly rather than through `FlagEmbedding`
  (same aarch64-build rationale as `rerank-only`). On a dev host this
  replaces Ollama's `bge-m3` outright rather than sitting alongside it —
  Ollama then serves chat only, and the model is loaded once instead of
  twice.
- **Diarize-only** (`docker/compose.diarize-only.yaml`, CPU OK) — a single
  `diarize-only` container on `inference-net`, no router, no GPU. Same
  audience as NER-only / Rerank-only / CLIP-only / Embed-only;
  co-deployable so a non-CUDA host can offer `/gliner`, `/rerank`,
  `/clip/*`, `/v1/embeddings`+`/pooling`+`/tokenize`, and `/diarize` at
  once. Reached as
  `http://diarize-only:8000/diarize`. Runs the same
  `src/diarize_server.py` the full-stack `diarize` service does, so it
  speaks the identical multipart `/diarize` contract; consumers (Nextext)
  target either backend by changing the base URL alone. Uses
  `Dockerfile.diarize.cpu` (uv-managed Python 3.11, CPU torch +
  torchaudio + torchcodec, `pyannote.audio`, `ffmpeg`). The pyannote
  weights are gated
  on the HF Hub (see "Diarization backend" below for the one-time
  pre-download), so unlike the other CPU shapes its cache cannot be
  populated anonymously.
- **ASR-only** (`docker/compose.asr-only.yaml`, CPU OK) — a single `asr-only`
  container on `inference-net`, no router, no GPU. Same audience as the other
  CPU shapes. The full-stack `asr` runs Whisper on vLLM (CUDA-only), so this
  shape instead ships `src/asr_server.py` around **openai-whisper** but
  exposes the identical OpenAI `/v1/audio/transcriptions` contract, so
  consumers (Nextext) target either backend by changing the base URL alone.
  Reached as `http://asr-only:8000/v1/audio/transcriptions`. Uses
  `Dockerfile.asr.cpu` (uv-managed Python 3.11, CPU torch, `openai-whisper`,
  `ffmpeg`); `WHISPER_MODEL` defaults to `openai/whisper-large-v3` (mapped to
  the openai-whisper name `large-v3`). Whisper weights are public — no gated
  download.
- **VAD-only** (`docker/compose.vad-only.yaml`, CPU OK) — a single `vad-only`
  container on `inference-net`, no router, no GPU. Runs the same
  `src/vad_server.py` the full-stack `vad` service does (Silero VAD), so it
  speaks the identical multipart `/vad` contract; consumers (Nextext) target
  either backend by changing the base URL alone. Reached as
  `http://vad-only:8000/vad`. Uses `Dockerfile.vad.cpu` (uv-managed
  Python 3.11, CPU torch + torchaudio, `silero-vad`, `ffmpeg`). The
  `silero-vad` package bundles its weights, so — unlike diarize-only —
  nothing is downloaded; airgap-clean out of the box.

The shapes are **not profiles of one compose file** — they have different
images, different topologies, and (gliner-only, rerank-only, clip-only,
embed-only, diarize-only, asr-only, vad-only) no router. Pick one per host.
They reuse the same external `inference-net` network and `huggingface-cache`
volume, so the one-time `make network` / `make volumes` prerequisites apply
to all of them. The seven CPU-only shapes can coexist on a single host
because they target different network aliases and host ports.

## Common commands

The `Makefile` is the entry point — it points Compose at `docker/compose.yaml`,
since a bare `docker compose` from the repo root no longer finds the compose
file. It builds and runs the full stack (chat, embed, rerank, clip, asr,
diarize, vad, gliner, router) — there are no optional profiles.

Prerequisites (one-time per host):

```bash
make network           # create the external inference-net
make volume            # create the huggingface-cache Docker volume
cp .env.example .env   # then edit model IDs / API key / GPU placement
```

Bring the stack up:

```bash
make build                 # build images for the full stack
make up                    # detached, no build; production shape (no host ports)
make up-dev                # detached, no build; with host ports published (dev)
make dev                   # build, then up-dev (dev convenience)
```

`make up` and `make up-dev` are detached and never build (`up -d --no-build`) —
build first with `make build`, or use `make dev` (build + up-dev). `make up`
runs the base `docker/compose.yaml` alone (production shape — no host ports);
`make up-dev` layers `docker/compose.override.yaml` so the router is published
on the host for dev.

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
make build-rerank-only    # builds vllm-service-rerank-only
make up-rerank-only       # one rerank-only container on inference-net (no host port)
make up-dev-rerank-only   # like 'up-rerank-only', but publishes the rerank port on the host
make stop-rerank-only
make bundle-rerank-only   # versioned .tar.gz of the rerank-only image
```

Or, for the CLIP-only shape (no CUDA, no router — pairs with Ollama,
typically co-deployed with NER-only and Rerank-only):

```bash
make build-clip-only      # builds vllm-service-clip-cpu
make up-clip-only         # one clip-only container on inference-net (no host port)
make up-dev-clip-only     # like 'up-clip-only', but publishes the CLIP port on the host
make stop-clip-only
make bundle-clip-only     # versioned .tar.gz of the clip-cpu image
```

Or, for the Embed-only shape (no CUDA, no router; on a dev host this
replaces Ollama's `bge-m3` rather than pairing with it — Ollama then
serves chat only, and the model loads once instead of twice — typically
co-deployed with NER-only, Rerank-only, and CLIP-only):

```bash
make build-embed-only    # builds vllm-service-embed-only
make up-embed-only       # one embed-only container on inference-net (no host port)
make up-dev-embed-only   # like 'up-embed-only', but publishes the embed port on the host
make stop-embed-only
make bundle-embed-only   # versioned .tar.gz of the embed-only image
```

Or, for the Diarize-only shape (no CUDA, no router — pairs with Ollama,
typically co-deployed with NER-only, Rerank-only, CLIP-only, and
Embed-only):

```bash
make build-diarize-only   # builds vllm-service-diarize-cpu
make up-diarize-only      # one diarize-only container on inference-net (no host port)
make up-dev-diarize-only  # like 'up-diarize-only', but publishes the diarize port on the host
make stop-diarize-only
make bundle-diarize-only  # versioned .tar.gz of the diarize-cpu image
```

Or, for the ASR-only shape (no CUDA, no router — CPU openai-whisper; pairs
with Ollama):

```bash
make build-asr-only       # builds vllm-service-asr-cpu
make up-asr-only          # one asr-only container on inference-net (no host port)
make up-dev-asr-only      # like 'up-asr-only', but publishes the ASR port on the host
make stop-asr-only
make bundle-asr-only      # versioned .tar.gz of the asr-cpu image
```

Or, for the VAD-only shape (no CUDA, no router — Silero VAD; pairs with
Ollama):

```bash
make build-vad-only       # builds vllm-service-vad-cpu
make up-vad-only          # one vad-only container on inference-net (no host port)
make up-dev-vad-only      # like 'up-vad-only', but publishes the VAD port on the host
make stop-vad-only
make bundle-vad-only      # versioned .tar.gz of the vad-cpu image
```

Other useful operations use the raw compose form, pointed at the compose file:

```bash
docker compose --env-file .env -f docker/compose.yaml logs -f router    # follow LiteLLM proxy logs
docker compose --env-file .env -f docker/compose.yaml logs -f chat      # follow a single backend
docker compose --env-file .env -f docker/compose.yaml up -d chat        # recreate one backend to apply .env changes (restart does not re-read env)
docker compose --env-file .env -f docker/compose.yaml ps                # health status of each service
make stop                                                               # stop the active service set
```

Build and bundle commands:

```bash
make build               # build images for the full stack
make bundle              # versioned .tar.gz pair from the latest annotated release tag (production)
make bundle-dev          # .tar.gz pair of the current working tree (dev/soak)
```

There is no test suite, but the Python sources in `src/` are linted with the
org-wide strict regime — ruff + pyrefly via `.pre-commit-config.yaml`, mirroring
`nos-tromo/.github`'s canonical `configs/python-strict/` (only `target-version`
differs → `py311`). CI enforces it in the `python-lint` job, which first runs
that repo's `validate_strict_config.py` to fail on any drift from the canonical
config. Run locally with `uv sync` then `uv run pre-commit run --all-files`. The
heavy ML backends (torch, transformers, openai-whisper, pyannote, silero) are
not installed for linting — `pyrefly` treats them as `Any` (`ignore-missing-imports`);
only the light typed deps (`fastapi`, `pydantic`, `numpy`) are declared so
strict mode can check the first-party code.

The servers (`src/rerank_server.py`, `src/clip_server.py`,
`src/diarize_server.py`, `src/asr_server.py`, `src/vad_server.py`) are also
small enough to verify by curl — rerank, clip, asr, and vad against their
running standalone containers, diarize through the full-stack router — see the
Architecture / Rerank-only, CLIP-only, Diarization, ASR-only, and VAD sections
below.

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
`/v1/models`. vLLM-specific paths (`/v1/rerank`, `/pooling`, `/tokenize`),
GLiNER's `/gliner`, CLIP's `/clip/*`, diarization's `/diarize`, and VAD's
`/vad` are forwarded by `pass_through_endpoints` in
`docker/litellm.config.yaml`. `/gliner`, `/clip/*`, `/diarize`, and `/vad`
are the pass-throughs whose upstreams are *not* vLLM containers — they go to
the `gliner` (Ray Serve), `clip` (FastAPI), `diarize` (FastAPI), and `vad`
(FastAPI) services.

### Backends

Most backends share the same `docker/Dockerfile.vllm` image, launched with a
different model and per-service env-driven flags:

- `chat` — general LLM (`TEXT_MODEL`)
- `embed` — embeddings, `--runner pooling --convert embed` (`EMBED_MODEL`)
- `rerank` — reranker, `--runner pooling` (`RERANK_MODEL`)
- `asr` — Whisper (`WHISPER_MODEL`)

Four backends are exceptions that do not run vLLM:

- `gliner` — GLiNER zero-shot NER via Ray Serve (`NER_MODEL`, default
  `gliner-community/gliner_large-v2.5` on CUDA; set `NER_DEVICE=cpu` with
  the `gliner_medium-v2.5` variant for CPU-only hosts). Uses
  **`docker/Dockerfile.gliner.cuda`** (pytorch base + `gliner[serve]`).
  GLiNER's span-matching head isn't a stock HF classification head and the
  DeBERTa-v2/v3 disentangled attention used by the v2.5 checkpoints isn't
  natively supported by vLLM, so vLLM-native serving is not viable. The
  endpoint is `POST /gliner` with body `{text, labels, threshold}`, reached
  via the router's `/gliner` pass-through (not `/v1/...`).
- `clip` — CLIP image+text embedding via FastAPI (`CLIP_MODEL`). Uses
  **`docker/Dockerfile.clip.cuda`** and runs `src/clip_server.py`;
  reached via the router's `/clip/*` pass-throughs.
- `diarize` — speaker diarization via FastAPI (`DIARIZE_MODEL`, default
  `pyannote/speaker-diarization-community-1`). Uses
  **`docker/Dockerfile.diarize.cuda`** and runs `src/diarize_server.py`.
  pyannote is a multi-model pipeline (PyanNet segmentation + WeSpeaker
  embedding + agglomerative clustering), none of which are vLLM-supported
  architectures, so vLLM-native serving is not viable. The endpoint is
  `POST /diarize` (multipart audio + optional speaker-count form fields),
  reached via the router's `/diarize` pass-through.
- `vad` — Silero voice activity detection via FastAPI (`VAD_MODEL`, default
  `silero_vad`). Uses **`docker/Dockerfile.vad.cpu`** and runs
  `src/vad_server.py`. Silero is a tiny JIT speech/non-speech classifier,
  not a vLLM-supported architecture; it is CPU-only (no GPU benefit), so it
  runs on CPU even in the full stack. The endpoint is `POST /vad` (multipart
  audio + optional tuning form fields), reached via the router's `/vad`
  pass-through.

Each buildable service in `docker/compose.yaml` carries
`image: vllm-service-<svc>:${VLLM_SERVICE_VERSION:-latest}`. A raw `docker
compose -f docker/compose.yaml build` produces `:latest` tags for dev
workflows; `make bundle` exports `VLLM_SERVICE_VERSION=<release-tag>` (the latest
annotated tag — `make bundle-dev` or an off-tag build falls back to
`<date>-<short-sha>`) so the same compose file also produces version-tagged
tarballs for offline shipping.

All backends listen on internal port 8000 and join both `vllm-net` and the
external `inference-net` — the latter only so `obs-plane` can scrape their
metrics endpoints by service name; they are not an app-facing entry point.
Apps still reach the stack exclusively through `router`, which keeps the
`vllm-router` alias on `inference-net` for cross-project consumers.

### Dependency overlay

The vLLM base image (`vllm/vllm-openai:v0.26.0`, pinned by digest) ships with
plain vLLM and its full CUDA runtime preinstalled in the system Python prefix.
The Dockerfile adds vLLM's `[audio]` extras (`av`, `scipy`, `soundfile`,
`mistral_common[audio]`) so the `asr` (Whisper) service has what it needs,
plus `orjson` — vLLM picks it up opportunistically for
faster OpenAI-endpoint JSON serialization (falls back to stdlib `json` if
absent, so it's a perf bump rather than a hard requirement).

No lockfile, no `uv`, no `pyproject.toml` — the overlay is one line. If you
add another dependency later and want transitive pinning, that's the moment
to reintroduce a lockfile-driven workflow; for one extras specifier the
ergonomics aren't worth it.

### Service startup ordering

`depends_on … condition: service_healthy` chains the backends serially:
`chat → embed → embed-sparse → rerank → clip → asr → diarize → vad → gliner
→ router`. This
is intentional — backends compete for GPU memory at startup, so they are
brought up one at a time. `gliner` is deliberately **last**: it runs on Ray
Serve with `--target-memory-fraction` (a share of whatever GPU memory is still
free when it starts, not a fixed reservation like the vLLM backends'
`--gpu-memory-utilization`), so every other allocator — `asr` and the small
ad-hoc `diarize` footprint included — must be healthy before it comes up, or
it would claim the remaining memory and starve them. `vad` is a CPU service
(no GPU reservation), so its place in the chain is just ordering, not memory
contention. Healthchecks hit `http://localhost:8000/health` (vLLM backends,
`clip`, `diarize`, and `vad`), `http://localhost:8000/-/healthz` (the `gliner`
Ray Serve container), and `/health/liveliness` (router). Allow ~120s
`start_period` before treating a backend as unhealthy.

### Configuration surface

All tuning is done via `.env` (see `.env.example`). Per-service env vars follow
the pattern `<SERVICE>_<KNOB>` (e.g. `CHAT_GPU_MEMORY_UTILIZATION`,
`EMBED_HF_OVERRIDES`, `NER_ENABLE_FLASHDEBERTA`). The `chat` and `gliner`
entrypoints use a shell builder pattern (`set --
<cmd> …` then conditional `set -- "$@" --flag`) so optional flags are only
passed when the corresponding env var is set — when adding a new optional flag,
follow that same pattern rather than hard-coding it in `command:`. The `gliner`
shell builder invokes `python -m gliner.serve` instead of `vllm serve`, but the
structure is identical.

`CHAT_KV_CACHE_DTYPE` and `CHAT_SPECULATIVE_CONFIG` follow this pattern:
when set they append `--kv-cache-dtype` / `--speculative-config` to the
chat backend, enabling vLLM's FP8 quantized KV cache (~half the KV memory;
chat is the only backend where this matters — embed/rerank are pooling
runners and Whisper's decode is tiny) and speculative decoding (native MTP
drafting on MTP-native checkpoints). Unset means full-precision KV cache
and no speculation, the prior behavior. (K/V scales for fp8 load from the
checkpoint when present, else 1.0 — vLLM removed the startup calibration
flag `--calculate-kv-scales` upstream, so the repo no longer exposes it.)

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

`NER_MODEL`, `CLIP_MODEL`, `DIARIZE_MODEL`, and `VAD_MODEL` are the
exceptions: their servers have no OpenAI-shaped routes, so they are not
declared in `model_list` and do not appear in `/v1/models`. Clients hit
`/gliner`, `/clip/*`, `/diarize`, and `/vad` directly with service-native
bodies and never use the `model` field. Switching one of these vars and
restarting the matching service still works. (`VAD_MODEL` is informational —
the `silero-vad` package bundles a single model.)

### Rerank-only shape (CPU)

The `rerank-only` compose project runs `src/rerank_server.py` — a small
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

The `clip-only` compose project runs `src/clip_server.py` — a small
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

### Embed-only shape (CPU)

The `embed-only` compose project runs `src/embed_server.py` — a small
FastAPI app that loads `BAAI/bge-m3` with `transformers` directly (not
`FlagEmbedding`, whose `ir-datasets` → `zlib-state` dep tree fails to build
on aarch64) and serves **all three** of the full stack's `embed`-backend
contracts from that one loaded model: each route runs its own XLM-R
forward pass over the shared loaded model; dense takes CLS pooling + L2
normalization, sparse the model's own `sparse_linear.pt` head + ReLU,
with padding positions stripped via the attention mask for the sparse
route, then the BOS/EOS positions dropped the way vLLM's `BOSEOSFilter`
drops them (each sparse row has `len(/tokenize) - 2` entries; the
boundary weights do *not* ReLU to zero, so keeping them would diverge
from the CUDA backend). The win is one model load shared across routes, not one forward
pass — each route still runs its own full forward, and a consumer wanting
both makes two independent HTTP calls.

```
POST /v1/embeddings
{"input": "..." | ["...", ...]}
→
{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.01, -0.02, ...]}, ...],
 "model": "...", "usage": {"prompt_tokens": int, "total_tokens": int}}

POST /pooling
{"task": "token_classify", "input": ["...", ...]}
→
{"model": "...", "data": [{"index": 0, "data": [0.0, 0.31, ...]}, ...]}

POST /tokenize
{"prompt": "..."}
→
{"count": int, "max_model_len": int, "tokens": [int, ...]}
```

`/v1/embeddings` is OpenAI-compatible dense embeddings (the request's
`model` field is ignored — the container serves exactly one model, and
echoes its configured name back). `/pooling`'s `task` must be
`token_classify` — any other value returns HTTP 400; this server
implements no other pooling task. Model identity is fixed at container
startup via `EMBED_MODEL` (defaults to `BAAI/bge-m3`, the same model and
pooling definition the GPU stack's `embed` backend uses under vLLM's
`BgeM3EmbeddingModel` architecture, so scores are equivalent up to dtype:
this server runs float32, while vLLM's `dtype="auto"` casts bge-m3's
float32 checkpoint to float16, diverging by roughly 1e-3. Parity against
the CUDA stack was verified side-by-side on 2026-08-04 — see
`docs/2026-08-04-embed-parity-checklist.md`, the golden fixtures in
`eval/fixtures/embed_parity/`, and `eval/tests/test_embed_parity.py`,
which re-runs the four comparisons against any live backend);
`EMBED_MAX_LENGTH` (default `8192`) truncates all three routes identically
so `/tokenize`, `/pooling`, and `/v1/embeddings` never disagree on
sequence length.

`EMBED_MAX_BATCH_TOKENS` (default `16384`) bounds one forward pass.
`_encode_batch` pads with `padding=True`, so a batch's real cost is
**rows × longest row**, not the sum of its rows — one 4k-token chunk
among 63 short ones pads all 64 to 4k (~262k tokens). Both batch routes
run their input through `_iter_batches`, the single bounding seam:
`plan_batches` greedily splits the per-text token counts (taken from
`tokenize_ids`, the same seam `/tokenize` reports) into contiguous spans
under that padded budget, and each encoder concatenates its sub-batch
results. Both routes share the one seam deliberately — two copies could
drift, and the symptom of drift is a timeout under load, not a wrong
answer. Order and arity are preserved (one entry per input, in input
order), and a text that alone exceeds the budget gets its own pass
rather than being dropped or split. When adding a fourth batch route,
route it through `_iter_batches` too rather than calling `_encode_batch`
directly.

`GET /health` returns `{"status": "ok", "model": "..."}` and is the
healthcheck target.

Because one container now serves both embedding kinds, a consumer points
its dense embedding base and its sparse base at the same
`http://embed-only:8000` — no separate deployment needed for each. On a
dev host this shape replaces Ollama's `bge-m3` outright: Ollama then
serves chat only, and the model is loaded once (here) instead of twice
(once in Ollama, once in this container).

Smoke-test:

```bash
curl -fsS -X POST http://localhost:${EMBED_HOST_PORT:-8007}/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "what is RAG?"}'

curl -fsS -X POST http://localhost:${EMBED_HOST_PORT:-8007}/pooling \
  -H 'Content-Type: application/json' \
  -d '{"task": "token_classify", "input": ["what is RAG?"]}'
```

### Diarization backend (full stack)

The `diarize` service runs `src/diarize_server.py` — a small FastAPI
app around a pyannote speaker-diarization pipeline (pyannote.audio 4.x;
default model `pyannote/speaker-diarization-community-1`). Decoding
lives in `src/diarize_audio.py`, pipeline construction in
`src/diarize_pipeline.py` (`build_pipeline` handles 4.x's renamed auth
kwarg `token=`; the server unwraps 4.x's `DiarizeOutput` via its
`.speaker_diarization` attribute, falling back to the bare `Annotation`
3.x returns). Uploaded
bytes are decoded to 16 kHz mono float32 by piping through `ffmpeg`
(any container ffmpeg can decode), then handed to the pipeline as a
pre-decoded waveform dict; torchaudio file decoding is never used (the
server stubs the backend-probing API torchaudio 2.9+ removed before
importing pyannote). The Dockerfile pins `pyannote.audio>=4,<5` (4.x is
required by the community-1 checkpoint and imports torchcodec instead of
torchaudio for decoding) and installs `torchcodec==0.11.1` as the `+cpu`
build from the PyTorch index, pinned and installed *before* pyannote so
pip doesn't resolve pyannote's own `torchcodec>=0.7` dependency to a
broken wheel: the plain PyPI wheels (0.12+) are CUDA-13 builds that
dlopen `libnvrtc.so.13`, and the `+cu128` build dlopens `libnppicc.so.12`
(NPP) — neither library ships in the pytorch `-runtime` base image. The
`+cpu` build costs nothing: torchcodec's CUDA path is video-only NVDEC
decoding, and this service pre-decodes via ffmpeg, so torchcodec is an
import-time formality; torch itself stays the base image's cu128 build
and inference runs on CUDA. 0.11 is the torch-2.11 row of torchcodec's
compatibility table — re-pick the pin when bumping `PYTORCH_IMAGE` (the
build smoke test fails on a mismatch). The old `huggingface_hub<1.0` cap
is dropped — it was only needed for the `use_auth_token` alias pyannote
3.x passed. Endpoints:

```
POST /diarize               # multipart `file=<bytes>` + optional integer
                            # form fields num_speakers OR min_speakers/
                            # max_speakers (combining both -> 400)
  -> {"segments": [{"start": <float sec>, "end": <float sec>,
      "speaker": "SPEAKER_00"}, ...], "speakers": [...]}

GET  /health
  -> {"status": "ok", "model": "...", "device": "..."}
```

Model identity is fixed at container startup via `DIARIZE_MODEL`
(default `pyannote/speaker-diarization-community-1`; the 3.1 pipeline
still loads on pyannote 4.x if configured). The checkpoints are gated
on the Hugging Face Hub: accept the conditions for the model repo —
for community-1 that's `pyannote/speaker-diarization-community-1` (its
exact gated dependency set is still being confirmed; the README runbook
documents the 3.1 procedure, which needs both
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`) —
then run once with `HF_HUB_OFFLINE=0`, `TRANSFORMERS_OFFLINE=0`, and
`HF_TOKEN` set so the weights land in the shared `huggingface-cache`
volume — the compose env points `PYANNOTE_CACHE` there, since pyannote
would otherwise download to `~/.cache/torch/pyannote`, outside the
volume. Consumers (Nextext) do speaker-to-ASR-segment alignment
client-side by maximum overlap, so the service returns raw turns only.
When `DIARIZE_VAD_URL` is set (the full-stack compose default is
`http://vad:8000`; the diarize-only compose hardcodes `http://vad-only:8000`
so a shared `.env`'s full-stack value can't leak in — gating engages when
`vad-only` is co-deployed), the server VAD-gates its output first — turns are
cropped to the Silero `/vad` speech timeline (fail-open;
`DIARIZE_VAD_GATE=off` disables; tuned `DIARIZE_VAD_THRESHOLD=0.4` /
`DIARIZE_VAD_PAD_MS=100`), so music/noise false alarms are dropped
server-side for every consumer.

A `diarize-only` standalone CPU shape (`docker/compose.diarize-only.yaml`,
`make up-diarize-only`) runs the same `src/diarize_server.py` without the
router — built from `Dockerfile.diarize.cpu` (uv-managed Python 3.11, CPU
torch + torchaudio + torchcodec, `pyannote.audio`, `ffmpeg`), reached directly at
`http://diarize-only:8000/diarize` with no Bearer auth, same posture as the
other `-only` shapes. In the full stack, requests go through the router,
which gates with the master key. Smoke-test (full stack):

```bash
curl -fsS -X POST http://localhost:${ROUTER_HOST_PORT:-9000}/diarize \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F 'file=@/path/to/recording.mp3' \
  -F 'max_speakers=4'
```

Smoke-test (diarize-only, no auth):

```bash
curl -fsS -X POST http://localhost:${DIARIZE_HOST_PORT:-8004}/diarize \
  -F 'file=@/path/to/recording.mp3' \
  -F 'max_speakers=4'
```

### ASR-only shape (CPU)

The full-stack `asr` service runs Whisper on vLLM (CUDA-only). The `asr-only`
compose project is its CPU counterpart: it runs `src/asr_server.py` — a
small FastAPI app around **openai-whisper** (the reference decoder Nextext also
runs in-process) — and exposes the same OpenAI `POST /v1/audio/transcriptions`
(and `/v1/audio/translations`) contract, so consumers swap backends by base URL
alone. This mirrors how `rerank_server.py` reimplements the forward pass rather
than running vLLM.

`WHISPER_MODEL` (default `openai/whisper-large-v3`) is mapped to the
openai-whisper checkpoint name by stripping the `openai/whisper-` prefix
(→ `large-v3`). `ASR_DEVICE` defaults to `cpu`. Weights are public and download from openai-whisper's CDN
into a subdirectory of the shared `huggingface-cache` volume (not the HF Hub,
so no token). `verbose_json` responses carry per-segment `no_speech_prob` and
the detected `language` — the fields Nextext filters on. The request `model`
field is accepted but ignored (the server uses the model it loaded at boot).
`GET /health` returns `{"status": "ok", "model": "...", "device": "..."}`.

Built from `Dockerfile.asr.cpu` (uv-managed Python 3.11, CPU torch,
`openai-whisper`, `ffmpeg`). Reached at
`http://asr-only:8000/v1/audio/transcriptions` with no Bearer auth. Smoke-test
(use a small model — `large-v3` on CPU is slow):

```bash
curl -fsS -X POST http://localhost:${ASR_HOST_PORT:-8005}/v1/audio/transcriptions \
  -F 'file=@/path/to/recording.mp3' \
  -F 'response_format=verbose_json'
```

### VAD backend (full stack + vad-only)

The `vad` service runs `src/vad_server.py` — a small FastAPI app around
**Silero VAD** (`silero-vad` pip package). It is a CPU service in **both** the
full stack and the standalone `vad-only` shape (Silero gains nothing from
CUDA), so a single `Dockerfile.vad.cpu` (uv-managed Python 3.11, CPU torch +
torchaudio, `silero-vad`, `ffmpeg`) builds both. The `silero-vad` package
bundles its weights, so nothing is downloaded at runtime — airgap-clean out of
the box, unlike the gated diarize weights. Endpoint:

```
POST /vad                   # multipart `file=<bytes>` + optional float/int form
                            # fields threshold, min_speech_duration_ms,
                            # min_silence_duration_ms, speech_pad_ms,
                            # max_speech_duration_s
  -> {"segments": [{"start": <float sec>, "end": <float sec>}, ...],
      "has_speech": <bool>, "sampling_rate": 16000}

GET  /health
  -> {"status": "ok", "model": "...", "device": "..."}
```

Like `/diarize`, the service returns raw speech turns; consumers reduce them
(e.g. to a speech/no-speech gate) client-side. `VAD_MODEL` (default
`silero_vad`) is informational; `VAD_USE_ONNX=true` runs the bundled ONNX graph
instead of the Torch JIT model. In the full stack, requests go through the
router's `/vad` pass-through (master-key gated); the `vad-only` shape is reached
directly at `http://vad-only:8000/vad` with no Bearer auth. Smoke-test (full
stack):

```bash
curl -fsS -X POST http://localhost:${ROUTER_HOST_PORT:-9000}/vad \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F 'file=@/path/to/recording.mp3'
```

Smoke-test (vad-only, no auth):

```bash
curl -fsS -X POST http://localhost:${VAD_HOST_PORT:-8006}/vad \
  -F 'file=@/path/to/recording.mp3'
```
