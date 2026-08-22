# Architecture

How the full CUDA stack is put together: which containers run, how requests
reach them, and which Docker networks carry the traffic. For the endpoint
surface and auth model see [api-reference.md](api-reference.md); for the
CPU-only standalone shapes see
[deployment-shapes.md](deployment-shapes.md).

## Internal services

Internally it runs:

- `router` (LiteLLM Proxy)
- `chat`
- `embed`
- `rerank`
- `gliner` (GLiNER, served via Ray Serve rather than vLLM)

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

- `clip` (CLIP image+text tower, served via FastAPI rather than vLLM)
- `asr` (Whisper ASR, served via vLLM)
- `diarize` (pyannote speaker diarization, served via FastAPI rather than vLLM)
- `vad` (Silero voice activity detection, served via FastAPI rather than vLLM)

Model-to-backend routing is declared in `docker/litellm.config.yaml`. Clients
select a backend purely by the `model` field they send; there is no path-based
dispatch.

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
- `chat`, `embed`, `rerank`, `gliner`, `clip`, `asr`, `diarize`, and `vad`
  also join `inference-net` (no additional alias), so `obs-plane` can scrape
  their metrics endpoints by service name — apps should still go through
  `router`, never call a backend directly.
- Both metrics surfaces are unauthenticated on `inference-net`, which is the
  trust boundary for scraping; app traffic through the router is still
  master-key gated.
