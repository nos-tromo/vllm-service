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

- `/v1/rerank`
- `/pooling`
- `/tokenize`

Internally it runs:

- `router` (LiteLLM Proxy)
- `chat`
- `embed`
- `rerank`

The following services are optional and only started with `--profile media`:

- `translate`
- `audio`

Model-to-backend routing is declared in `litellm.config.yaml`. Clients select a
backend purely by the `model` field they send; there is no path-based dispatch.

## Usage

1. Copy `.env.example` to `.env` and set the model IDs, API key, and any
   GPU-placement settings. If host port `9000` is already in use, set
   `ROUTER_HOST_PORT` in `.env` to another free port such as `9001`.
2. Ensure the external `huggingface-cache` Docker volume exists.
3. Initialize the shared proxy network and persistent model cache:

   ```bash
   docker network create inference-net
   docker volume create huggingface-cache
   ```

4. Start the core stack:

   ```bash
   docker compose up --build
   ```

   To also start the `translate` and `audio` services, add the `media` profile:

   ```bash
   docker compose --profile media up --build
   ```

5. Point third-party app at the router.

If the consuming app is on the same shared Docker network, use the router
alias directly:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://vllm-router:9000/v1
   OPENAI_API_KEY=<token>
   ```

If the consuming app is outside that network, use a host or reverse-proxy URL:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://<host>:${ROUTER_HOST_PORT:-9000}/v1
   OPENAI_API_KEY=<token>
   ```

## Offline image bundles

For airgapped hosts, customer deployments, or any environment without
Docker Hub access, `make bundle` produces a versioned `.tar.gz` pair you
can ship alongside `docker-compose.yml`, `litellm.config.yaml`, and `.env`.

### Producing the bundle

On a build host with internet:

```bash
make bundle           # core only (chat, embed, rerank)
make bundle-media     # core + media (translate, audio)
```

This computes `VLLM_SERVICE_VERSION` as `YYYY-MM-DD-<short-sha>` (override by
exporting it before invocation), builds the locally-buildable services with
that version tag, pulls the externally-hosted images (LiteLLM Proxy), then
writes two gzipped tarballs in the cwd:

| File | Contents |
|---|---|
| `vllm-service-built-<profile>-<version>.tar.gz` | Locally-built `vllm-service-{chat,embed,rerank,...}` images. |
| `vllm-service-pulled-<profile>-<version>.tar.gz` | Externally-hosted images (LiteLLM router); re-tagged so the `name:tag@digest` references in `docker-compose.yml` resolve after `docker load`. |

The compose file references the version through
`image: vllm-service-<svc>:${VLLM_SERVICE_VERSION:-latest}`, so it falls
back to `:latest` for normal dev workflows and uses the pinned tag whenever
the variable is set.

### Loading and running the bundle

Ship the two tarballs along with the matching `docker-compose.yml`,
`litellm.config.yaml`, and a `.env`. Then on the target host:

```bash
docker load -i vllm-service-built-core-<version>.tar.gz
docker load -i vllm-service-pulled-core-<version>.tar.gz
export VLLM_SERVICE_VERSION=<version>
docker compose up --no-build -d
```

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
- Only the `router` service joins `inference-net`; `chat`, `translate`, `embed`,
  `rerank`, and `audio` stay on the private network.
- The `router` service keeps its `vllm-router` alias on `inference-net` so
  existing consumers do not need to change their `OPENAI_API_BASE`.

## Updating the model catalog

`litellm.config.yaml` is model-agnostic: all model names are read at startup
from the environment variables `TEXT_MODEL`, `TRANSLATE_MODEL`, `EMBED_MODEL`,
and `WHISPER_MODEL`. To switch a model, update the relevant variable in `.env`
and restart the stack. No changes to `litellm.config.yaml` are required.

Clients must use the exact model ID set in `.env` as the `model` field in
their requests (e.g. `"model": "BAAI/bge-m3"`). The `/v1/models` endpoint
returns the currently active IDs.


## Calling the translate service

The translate service runs
[`Infomaniak-AI/vllm-translategemma-4b-it`](https://huggingface.co/Infomaniak-AI/vllm-translategemma-4b-it),
a vLLM-compatible repackaging of Google's TranslateGemma 4B. Unlike a general
chat model, it expects the source language, target language, and text to be
encoded in the message content using a delimiter format:

```
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

The translate service is only started when the `media` profile is active:

```bash
docker compose --profile media up
```

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

The audio service is only started when the `media` profile is active:

```bash
docker compose --profile media up
```
