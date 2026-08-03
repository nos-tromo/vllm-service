# FP8 KV cache quantization for the chat backend — design

Date: 2026-08-03
Status: approved

## Goal

Let a host halve the chat backend's KV cache memory footprint (≈2× the
context/concurrency headroom at the same `--gpu-memory-utilization`) by
opting into vLLM's FP8 quantized KV cache
(<https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/>).
The full stack runs GPU-memory-tight; the chat backend's KV cache is the
dominant elastic allocation.

## Scope

Chat backend only. `embed` and `rerank` are pooling runners (one forward
pass, no autoregressive KV cache growth) and `asr` (Whisper) decodes short
fixed windows — KV cache is not their pressure point. The non-vLLM services
(gliner, clip, diarize, vad) have no vLLM KV cache at all. No other shape
(`*-only` composes) runs a vLLM chat container, so only
`docker/compose.yaml` changes.

## Design

Two new opt-in env knobs, wired through the existing shell-builder pattern
in the chat service's `command:` in `docker/compose.yaml`:

- `CHAT_KV_CACHE_DTYPE` — when set, appends
  `--kv-cache-dtype "$CHAT_KV_CACHE_DTYPE"`. Passed through verbatim;
  vLLM validates the value (`fp8`, `fp8_e4m3`, `fp8_e5m2`). Unset →
  flag not passed → vLLM default `auto` → byte-identical current
  behavior.
- `CHAT_CALCULATE_KV_SCALES` — when `true`, appends
  `--calculate-kv-scales`. On-the-fly K/V scale calibration; the
  accuracy mitigation for fp8/e4m3, whose default scale of 1.0 is the
  known quality footgun. Independent knob, but only has effect together
  with an fp8 KV cache dtype (documented, not enforced — vLLM owns
  validation).

`.env.example` gains two commented lines in the Chat section explaining:
`fp8` resolves to `e4m3` on CUDA; `e5m2` trades precision for range and
does not use scales; `--enable-prefix-caching` (this stack's default)
composes fine with fp8 KV cache on the pinned vLLM; for maximum accuracy
use an llm-compressor-calibrated checkpoint via `TEXT_MODEL` (a model
choice, not an infra knob).

CLAUDE.md's configuration-surface section needs no structural change —
the knobs follow the documented `<SERVICE>_<KNOB>` pattern; a brief
mention alongside the other chat flags is enough.

Deliberately out of scope (YAGNI): `--kv-cache-dtype-skip-layers` (only
matters for sliding-window/hybrid models; add when a model needs it),
same knobs on other backends, changing any defaults.

## Error handling

Startup-time failures are vLLM's own and land in `docker compose logs
chat`: an invalid dtype string fails argument parsing; a GPU without FP8
support (pre-Ada for e4m3 kernels on some backends) fails at engine init.
The compose restart policy (`on-failure`, 5 attempts) applies as usual.
No new failure modes are introduced when the knobs are unset.

## Testing

No test suite exists for compose wiring; verification is manual, matching
repo practice:

1. `docker compose --env-file .env -f docker/compose.yaml config` renders
   with and without the new vars set.
2. With `CHAT_KV_CACHE_DTYPE=fp8` (and optionally
   `CHAT_CALCULATE_KV_SCALES=true`) in `.env`, `make dev` brings chat up
   healthy; startup log shows the fp8 KV cache dtype and a larger
   available-KV-cache token count than before.
3. Smoke `POST /v1/chat/completions` through the router returns a
   coherent response.
