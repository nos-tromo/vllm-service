# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A pure infrastructure repo: a Docker Compose stack that fronts several vLLM
backends with a single LiteLLM Proxy router. There is **no application source
code** — everything is configuration. The Docker assets live under `docker/`
(`docker/compose.yaml`, `docker/compose.override.yaml`, `docker/Dockerfile.vllm`,
`docker/Dockerfile.gliner`, `docker/litellm.config.yaml`) plus `.env` and
`.dockerignore` at the repo root.

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
make build             # build images for the active service set
make up                # core: router, chat, embed, rerank, ner
make up PROFILE=media  # add translate + audio
```

`make up` layers `docker/compose.override.yaml` so the router is published on
the host for dev. The base `docker/compose.yaml` is the production shape and
publishes no host ports.

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

There is no test suite or linter.

## Architecture

### Routing model

`router` (LiteLLM Proxy, port 4000 inside, published on
`${ROUTER_HOST_PORT:-9000}` on the host by `docker/compose.override.yaml`
when `make up` is used) is the **only** entry point. Clients always send to
the router and select a backend by the `model` field in the request body —
there is no path-based dispatch. `docker/litellm.config.yaml` maps each
`model_name` (read from env vars at startup) to an upstream `api_base`
(`http://chat:8000/v1`, `http://embed:8000/v1`, etc.).

LiteLLM natively exposes `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/translations`, and
`/v1/models`. vLLM-specific paths (`/v1/rerank`, `/pooling`, `/tokenize`) and
GLiNER's `/gliner` are forwarded by `pass_through_endpoints` in
`docker/litellm.config.yaml`. `/gliner` is the only pass-through whose upstream
is *not* a vLLM container — it goes to the `ner` service (Ray Serve).

### Backends

Most backends share the same `docker/Dockerfile.vllm` image, launched with a
different model and per-service env-driven flags:

- `chat` — general LLM (`TEXT_MODEL`)
- `embed` — embeddings, `--runner pooling --convert embed` (`EMBED_MODEL`)
- `rerank` — reranker, `--runner pooling` (`RERANK_MODEL`)
- `translate` *(profile: media)* — TranslateGemma fork (`TRANSLATE_MODEL`)
- `audio` *(profile: media)* — Whisper (`WHISPER_MODEL`)

`ner` is the one exception — it uses **`docker/Dockerfile.gliner`** (pytorch
base + `gliner[serve]`) and runs Ray Serve, not vLLM. GLiNER's span-matching head
isn't a stock HF classification head and the DeBERTa-v2/v3 disentangled
attention used by the v2.5 checkpoints isn't natively supported by vLLM,
so vLLM-native serving is not viable. The endpoint is `POST /gliner` with
body `{text, labels, threshold}` and is reached via the router's
`/gliner` pass-through (not `/v1/...`).

- `ner` — GLiNER zero-shot NER via Ray Serve (`NER_MODEL`, default
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
`chat → embed → rerank → ner → audio → translate → router`. This is intentional —
backends compete for GPU memory at startup, so they are brought up one at a time.
Healthchecks hit `http://localhost:8000/health` (vLLM backends),
`http://localhost:8000/-/healthz` (the `ner` Ray Serve container), and
`/health/liveliness` (router). Allow ~120s `start_period` before treating a
backend as unhealthy.

### Configuration surface

All tuning is done via `.env` (see `.env.example`). Per-service env vars follow
the pattern `<SERVICE>_<KNOB>` (e.g. `CHAT_GPU_MEMORY_UTILIZATION`,
`TRANSLATE_MAX_MODEL_LEN`, `EMBED_HF_OVERRIDES`, `NER_ENABLE_FLASHDEBERTA`). The
`chat`, `translate`, and `ner` entrypoints use a shell builder pattern (`set --
<cmd> …` then conditional `set -- "$@" --flag`) so optional flags are only
passed when the corresponding env var is set — when adding a new optional flag,
follow that same pattern rather than hard-coding it in `command:`. The `ner`
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
`model` field. Switching `NER_MODEL` and restarting `ner` still works.

### TranslateGemma quirk

`translate` runs `Infomaniak-AI/vllm-translategemma-4b-it` (a vLLM-compatible
repackaging — the stock `google/translategemma-*` cannot be served by a vanilla
OpenAI client). Clients must encode source/target/text inline:

```
<<<source>>>{iso_src}<<<target>>>{iso_tgt}<<<text>>>{text}
```

Trained for ~2K context — keep `TRANSLATE_MAX_MODEL_LEN=2048` unless you have a
specific reason to raise it.
