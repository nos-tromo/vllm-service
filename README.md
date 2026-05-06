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

Internally it runs:

- `router` (LiteLLM Proxy)
- `chat`
- `embed`
- `rerank`

The following services are optional and only started with `--profile media`:

- `audio`
- `translate`

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
   OPENAI_API_BASE=http://vllm-router:4000/v1
   OPENAI_API_KEY=<token>
   ```

If the consuming app is outside that network, use a host or reverse-proxy URL:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://<host>:${ROUTER_HOST_PORT:-9000}/v1
   OPENAI_API_KEY=<token>
   ```

## Networking

- `vllm-net` is private to this compose project and carries traffic between the
  router and the worker containers.
- `inference-net` is an external shared Docker network used for cross-project
  service discovery and reverse-proxy access.
- Only the `router` service joins `inference-net`; `chat`, `embed`,
  `rerank`, `audio`, and `translate` stay on the private network.
- The `router` service keeps its `vllm-router` alias on `inference-net` so
  existing consumers do not need to change their `OPENAI_API_BASE`.

## Airgapped / offline deployment

When the target server has no internet access the `uv pip install` step in
`Dockerfile.vllm` fails because it cannot reach PyPI.  The Dockerfile supports
two strategies, described below.  **The wheels approach (strategy B) is
confirmed working and is the recommended path.**

### Strategy A — BuildKit cache (if the server was previously online)

If the server ran a successful build using a proxy in the past, the Docker
BuildKit cache may already contain the required packages.  Try building without
`OFFLINE_BUILD` first:

```bash
docker compose up --build --pull never
```

If the BuildKit cache is warm the build succeeds without network access.  If it
fails with a PyPI connection error, fall back to strategy B.

### Strategy B — pre-downloaded wheels (recommended)

#### 1 — Download wheels (on a connected machine)

```bash
./scripts/download-wheels.sh
```

This launches the vllm base image, resolves exactly which packages are missing
from the base image (`vllm[audio]` extras, `orjson`, `conch-triton-kernels`,
and any `transformers` upgrade), and downloads those wheels into `wheels/`.
The `wheels/*.whl` files are git-ignored; ship them separately.

#### 2 — Bundle for transfer

```bash
# Update the git bundle
git bundle create vllm-service.bundle --all

# Archive the wheels
tar czf wheels.tar.gz wheels/
```

Transfer both `vllm-service.bundle` and `wheels.tar.gz` to the airgapped
server.

#### 3 — Build on the airgapped server

```bash
git clone vllm-service.bundle vllm-service
cd vllm-service
tar xzf ../wheels.tar.gz

OFFLINE_BUILD=1 docker compose up --build --pull never
```

Setting `OFFLINE_BUILD=1` tells the Dockerfile to install from `wheels/`
using `--no-index --find-links` instead of reaching out to PyPI.  The variable
can also be set permanently in `.env`.

## Updating the model catalog

`litellm.config.yaml` is model-agnostic: all model names are read at startup
from the environment variables `TEXT_MODEL`, `EMBED_MODEL`, `RERANK_MODEL`,
`TRANSLATE_MODEL`, and `WHISPER_MODEL`. To switch a model, update the relevant
variable in `.env` and restart the stack. No changes to `litellm.config.yaml`
are required.

Clients must use the exact model ID set in `.env` as the `model` field in
their requests (e.g. `"model": "BAAI/bge-m3"`). The `/v1/models` endpoint
returns the currently active IDs.

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

The translate service is only started when the `media` profile is active:

```bash
docker compose --profile media up
```
