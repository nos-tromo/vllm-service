# Architecture

How the full CUDA stack is put together: which containers run, how requests
reach them, and which Docker networks carry the traffic. For the endpoint
surface and auth model see [api-reference.md](api-reference.md); for the
CPU-only standalone shapes see
[deployment-shapes.md](deployment-shapes.md).

## Internal services

Twelve containers, plus a one-shot `volume-permissions` init job. None are behind
a Compose profile — the full stack runs all of them.

`router` is the only container apps talk to. The other eleven are workers.

| Service | Serving runtime | What it serves | Reached through |
|---|---|---|---|
| `router` | LiteLLM Proxy | dispatch, auth, metrics, audit log | it *is* the entry point |
| `chat` | vLLM | chat completions | `model` field |
| `embed` | vLLM | dense embeddings | `model` field |
| `embed-sparse` | vLLM | sparse/lexical weights + tokenization | `/pooling`, `/tokenize` |
| `rerank` | vLLM | reranking | `model` field |
| `asr` | vLLM | Whisper transcription | `model` field |
| `gliner` | Ray Serve | GLiNER named-entity recognition | `/gliner` |
| `clip` | FastAPI (uvicorn) | CLIP image + text tower | `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension` |
| `diarize` | FastAPI (uvicorn) | pyannote speaker diarization | `/diarize` |
| `vad` | FastAPI (uvicorn) | Silero voice-activity detection | `/vad` |
| `faces` | FastAPI (uvicorn) | YuNet face detection + SFace per-video grouping (CPU) | `/faces/detect`, `/faces/group` |
| `topics` | FastAPI (uvicorn) | topic-model fit over caller-supplied vectors (CPU) | `/topics/fit` |

Despite the repo's name, **only five of the eleven workers run on vLLM.** vLLM
serves models it has a runner for — generation, pooling, transcription. The
other six do not: `gliner` runs GLiNER's own Ray Serve app, `clip`, `diarize`
and `vad` are small FastAPI servers around the upstream libraries, `faces` is a
small FastAPI server running two ONNX models through OpenCV, and `topics` is a
small FastAPI server around a topic-model fit that loads no model at all.
They are first-class members of the stack, not adapters bolted on: the router
fronts them the same way, and they join the same networks.

`topics` is stateless CPU compute with no weights, so it sits outside the GPU
startup chain: the router waits only for it to *start*, never for it to be
healthy, which keeps a failed startup self-check from taking inference down.

`faces` is CPU-only too and also sits outside the GPU chain (it waits only for
`volume-permissions`), and the router likewise waits only for it to *start*:
with either ONNX file missing from the cache — say, on a host upgraded without
fetching them — `faces` does not start, and only `/faces/*` fails; the rest of
inference stays up. It returns labels, never the features it groups by — see
[api-reference.md](api-reference.md#faces).

`embed` and `embed-sparse` are the **same image and the same model**
(`BAAI/bge-m3`) run twice with different pooler tasks. vLLM binds one pooling
task per server and rejects per-request switching, so the dense
`/v1/embeddings` contract stays on `embed` while the sparse `token_classify`
contract (`/pooling`, `/tokenize`) moved to `embed-sparse`.

### Two dispatch mechanisms, not one

Routing is declared in `docker/litellm.config.yaml`, and it works two ways —
which of the two applies is the "Reached through" column above:

- **By `model` field** (`model_list`) — `chat`, `embed`, `rerank` and `asr`
  are OpenAI-compatible, so a client picks one purely by the model id it
  sends. There is no path-based dispatch *among these four*: they all answer
  on the standard OpenAI routes.
- **By path** (`pass_through_endpoints`) — the other seven speak their own
  non-OpenAI contracts, so the router forwards fixed paths to them verbatim.
  A `model` field means nothing here.

Both go through the router and both are master-key gated. The full endpoint
surface is in [api-reference.md](api-reference.md).

### The audit callback

The router writes the transfer record of the data-protection audit log: one
`ai_call` event per model call, with the requested model, the deployment and
revision that served it, its size, duration and outcome, and the calling
app's `X-Request-ID` (see [audit.md](audit.md)). It is a LiteLLM callback,
`docker/router/router_audit.py`, mounted read-only into the pinned image at
`/opt/router` together with the vendored audit library and put on
`PYTHONPATH`; `litellm_settings.callbacks` registers it and
`LITELLM_WORKER_STARTUP_HOOKS` opens each worker's audit chain at startup.
Its events go to the container's stdout, which carries nothing else at
runtime; every log line goes to stderr. Each deployment's `model_info.id` in
`docker/litellm.config.yaml` names the compose service the event records as
the recipient.

### Prefix-cache isolation

`chat` caches prompt prefixes, and a cache hit shortens the time to the first
token, so a caller who can time requests could test whether another caller's
prompt is cached. vLLM shares cached blocks only between requests with the same
`cache_salt`. A second router callback, `docker/router/cache_salt.py`
(`cache_salt.handler`), sets that salt on every completion call and replaces
any salt the client sent. The salt depends on the caller's partition:

| Request carries | Partition |
|---|---|
| an Open WebUI user token that verifies against `AUDIT_OPENWEBUI_JWT_SECRET` | that user |
| `X-Cache-Scope: <64 lowercase hex>`, the per-user pseudonym docint, Nextext and translator send | that scope |
| any other `X-OpenWebUI-*` header | all Open WebUI users together |
| none of these | every other caller together |

The rows are checked in order. The salt is an HMAC-SHA256 of the partition
under `OPENAI_API_KEY`, so it cannot be guessed, is the same in every router
worker, and passes vLLM's salt check (64 hex characters). The pseudonym is not
logged or recorded; it only selects the partition. Each app derives it from
the user name under a key that exists only in that app's process, so the
router cannot map it back to a user. An app restart therefore starts that
app's users with a cold cache.

Only `completion` and `text_completion` calls are salted. vLLM does not cache
prefixes for the embedding, reranking and Whisper models, and pass-through
bodies are forwarded unchanged. Some sharing remains:

- An app sends no scope for work done for no user (CLI runs, maintenance
  jobs). That work shares the last partition.
- Without `AUDIT_OPENWEBUI_JWT_SECRET`, all Open WebUI users share one
  partition, and a caller that reaches `chat` directly on `inference-net`
  bypasses the router and its salt.

Keep vLLM's default `--prefix-caching-hash-algo` (`sha256`), and leave
`--enable-prompt-tokens-details` unset: it would report cache hits in
`usage`, without any timing. `CHAT_ENABLE_PREFIX_CACHING=false` turns caching
off completely ([configuration.md](configuration.md#prefix-caching)).

### The gliner watchdog

The gliner container is supervised by `scripts/gliner_watchdog.sh` (installed as
`/usr/local/bin/gliner-watchdog`): it probes `POST /gliner` and self-exits after
`NER_WATCHDOG_FAILURES` consecutive failures so `restart: unless-stopped` recreates
the container, recovering from the Ray Serve rank-consistency wedge
(ray-project/ray#63862). The Docker healthcheck uses the same functional probe;
set `NER_WATCHDOG_ENABLED=false` to disable. Knobs: `NER_WATCHDOG_*` in `.env.example`.
The wedge itself is fixed upstream in Ray >= 2.57.0 (ray-project/ray#64636; both
gliner Dockerfiles floor `ray[serve]>=2.57`), but the watchdog stays as
defense-in-depth: Ray's memory monitor can still kill the ServeController/replica
under host RAM pressure and leave gliner unresponsive (see
`NER_RAY_MEMORY_THRESHOLD`).

## Networking

- `vllm-net` is private to this compose project and carries traffic between the
  router and the worker containers.
- `inference-net` is an external shared Docker network used for cross-project
  service discovery and reverse-proxy access.
- `router` joins `inference-net` with the `vllm-router` alias so existing
  consumers do not need to change their `OPENAI_API_BASE`; it remains the
  only app-facing entry point. `obs-plane` also scrapes the router's own
  `/metrics` there (LiteLLM's Prometheus callback, enabled in
  `docker/litellm.config.yaml`) for routing-layer telemetry the backends
  cannot report — per-model request counts, end-to-end latency including
  routing, and failures that never reached a backend.
- All eleven workers (`chat`, `embed`, `embed-sparse`, `rerank`, `asr`,
  `gliner`, `clip`, `diarize`, `vad`, `faces`, `topics`) also join
  `inference-net` (no additional alias), so `obs-plane` can reach them by
  service name — apps should still go through `router`, never call a backend
  directly. Not all eleven have something to scrape: the five vLLM workers serve
  `/metrics`, while the FastAPI wrappers (`clip`, `diarize`, `vad`, `faces`,
  `topics`) expose only their contract routes and `/health`.
- Both metrics surfaces are unauthenticated on `inference-net`, which is the
  trust boundary for scraping; app traffic through the router is still
  master-key gated.
