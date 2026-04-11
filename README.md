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
- `translate`
- `embed`
- `rerank`
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

4. Start the stack:

   ```bash
   docker compose up --build
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

The public list of model names is declared in `litellm.config.yaml` under
`model_list`. Each entry maps a client-visible `model_name` to a vLLM backend
`api_base`. When you override `TEXT_MODEL`, `TRANSLATE_MODEL`, `EMBED_MODEL`,
`RERANK_MODEL`, or `WHISPER_MODEL` in `.env`, also update the matching
`model_name` (and the `model:` field inside `litellm_params`) in
`litellm.config.yaml` so `/v1/models` discovery and client calls keep working.
