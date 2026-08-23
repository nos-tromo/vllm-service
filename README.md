# Standalone vLLM Service

This repository runs the standalone routed vLLM deployment used by [docint](https://github.com/nos-tromo/docint) and
other consumers. It is pure infrastructure — Docker Compose, LiteLLM, vLLM and
a handful of small FastAPI servers; no application source.

## The routed-endpoint model

The stack exposes **one** routed HTTP endpoint fronted by
[LiteLLM Proxy](https://docs.litellm.ai/docs/proxy), which dispatches to a
backend two ways. The OpenAI-compatible backends are selected by the `model`
field in the request body, not by path — they all answer on the standard
OpenAI routes (`/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`,
`/v1/audio/transcriptions`, …). The backends that speak their own contracts
are reached by path instead, forwarded verbatim (`/pooling`, `/tokenize`,
`/gliner`, `/clip/*`, `/diarize`, `/vad`). Every route
is gated by the router's master key, so clients send
`Authorization: Bearer $OPENAI_API_KEY`.

The full endpoint list, the two auth exceptions, and request/response bodies
are in [api-reference.md](docs/api-reference.md).

## Deployment shapes

| Shape | Compose project | Serves |
|---|---|---|
| Full stack (CUDA) | `docker/compose.yaml` | everything, behind the LiteLLM router |
| NER-only | `docker/compose.gliner-only.yaml` | `/gliner` |
| Rerank-only | `docker/compose.rerank-only.yaml` | `/rerank` |
| CLIP-only | `docker/compose.clip-only.yaml` | `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension` |
| Embed-only | `docker/compose.embed-only.yaml` | `/v1/embeddings`, `/pooling`, `/tokenize` |
| Diarize-only | `docker/compose.diarize-only.yaml` | `/diarize` |
| ASR-only | `docker/compose.asr-only.yaml` | `/v1/audio/transcriptions`, `/v1/audio/translations` |
| VAD-only | `docker/compose.vad-only.yaml` | `/vad` |

The seven `-only` shapes are single-container CPU deployments for hosts that
cannot run the CUDA stack (Mac dev boxes, ROCm or CPU-only Linux running Ollama
for chat/embed). They have no router and no auth, expose the same contracts as
the full stack, and can be co-deployed on one host. Runbook:
[deployment-shapes.md](docs/deployment-shapes.md).

## Prerequisites

- Docker with Compose v2.
- The external `inference-net` Docker network and the `huggingface-cache`
  volume (`make network` and `make volumes` create them).
- A `.env` file — copy `.env.example` and set the model IDs and API key.

## Quick start

The Docker assets live under `docker/`: a base `compose.yaml`, a
`compose.override.yaml` dev overlay, the Dockerfiles, and
`litellm.config.yaml`. The `Makefile` is the entry point — it points Compose
at `docker/compose.yaml`, since a bare `docker compose` from the repo root no
longer finds the compose file.

1. Copy `.env.example` to `.env` and set the model IDs, API key, and any
   GPU-placement settings. `make up-dev` publishes the router on host port
   `9000`; if that port is already in use, set `ROUTER_HOST_PORT` in `.env`
   to another free port such as `9001`.
2. Ensure the external `huggingface-cache` Docker volume exists.
3. Initialize the shared proxy network and persistent model cache:

   ```bash
   make network   # create the external inference-net
   make volumes   # create the huggingface-cache Docker volume
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

5. Point a consuming app at the router.

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

Switching a model is an `.env` edit plus a restart — see
[configuration.md](docs/configuration.md#updating-the-model-catalog).

## Operating

Run `make help` for the full target list: the full-stack lifecycle
(`network` / `volumes` / `build` / `up` / `up-dev` / `dev` / `stop` / `down` /
`logs`), the same set per `-only` shape, the `bundle` targets, and the local
lint gate (`make verify`).

## Documentation

Reference material lives in [docs/](docs/README.md).

- [architecture.md](docs/architecture.md) — the containers behind the router,
  the gliner watchdog, and the Docker networks.
- [deployment-shapes.md](docs/deployment-shapes.md) — bringing up any of the
  seven CPU-only standalone shapes.
- [configuration.md](docs/configuration.md) — the env knobs that need
  explaining, and how to switch models.
- [api-reference.md](docs/api-reference.md) — endpoint surface, auth, and every
  request/response body.
- [airgap-bundles.md](docs/airgap-bundles.md) — producing and loading offline
  image bundles.
- [development.md](docs/development.md) — the lint regime for the Python
  servers.

## Pointers

- Consumers in the federation: [docint](https://github.com/nos-tromo/docint),
  [Nextext](https://github.com/nos-tromo/Nextext),
  [chorus](https://github.com/nos-tromo/chorus),
  [translator](https://github.com/nos-tromo/translator),
  [open-webui-service](https://github.com/nos-tromo/open-webui-service).
- Issues: <https://github.com/nos-tromo/vllm-service/issues>
- Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
