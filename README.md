# Standalone vLLM Service

This repository runs the standalone routed vLLM deployment used by Docint and
other consumers.

## Purpose

The stack exposes one routed HTTP endpoint. The router fronts these paths:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/models`
- `/v1/rerank`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/pooling`
- `/tokenize`

Internally it runs:

- `router`
- `chat`
- `embed`
- `rerank`
- `audio`

## Usage

1. Copy `.env.example` to `.env` and set the model IDs, API key, and any
   GPU-placement settings.
2. Ensure the external `huggingface-cache` Docker volume exists.
3. Run `docker_setup.sh` to initialize the persistent model cache.
4. Create the shared proxy network once:

   ```bash
   docker network create proxy-net
   ```

5. Start the stack:

   ```bash
   docker compose up --build
   ```

6. Point third-party app at the router.

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
   OPENAI_API_BASE=http://<host>:9000/v1
   OPENAI_API_KEY=<token>
   ```

## Networking

- `vllm-net` is private to this compose project and carries traffic between the
  router and the worker containers.
- `proxy-net` is an external shared Docker network used for cross-project
  service discovery and reverse-proxy access.
- Only the `router` service joins `proxy-net`; `chat`, `embed`, `rerank`, and
  `audio` stay on the private network.
