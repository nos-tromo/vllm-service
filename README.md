# Standalone vLLM Service

This directory is a scaffold for a vLLM deployment.

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

1. Copy this directory into its own repository.
2. Copy `.env.example` to `.env` and set the model IDs, API key, and any
   GPU-placement settings.
3. Ensure the external `huggingface-cache` Docker volume exists.
4. Run `docker_setup.sh` to initialize the persistent model cache.
5. Start the stack:

   ```bash
   docker compose up --build
   ```

6. Point third-party app at the router:

   ```bash
   INFERENCE_PROVIDER=vllm
   OPENAI_API_BASE=http://<host>:9000/v1
   OPENAI_API_KEY=<token>
   ```
