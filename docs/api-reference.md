# API reference

## Endpoint surface

The stack exposes one routed HTTP endpoint fronted by [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy),
which dispatches two ways — see
[architecture.md](architecture.md#two-dispatch-mechanisms-not-one).

**By the `model` field**, on the routes LiteLLM serves natively:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/rerank`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/v1/models`

Rerank belongs on this list, not among the pass-throughs: LiteLLM's own
rerank route takes precedence over `pass_through_endpoints`, so the backend is
declared as a `model_list` entry (`RERANK_MODEL` -> `http://rerank:8000`, via
the `hosted_vllm` provider) and picked by the request's `model` field like any
other — see `docker/litellm.config.yaml:39-48`.

**By path**, for the backends whose contracts are not OpenAI-shaped.
`pass_through_endpoints` (`docker/litellm.config.yaml:95-153`) forwards the
request body verbatim; a `model` field means nothing on these:

| Path | Backend |
|---|---|
| `/pooling`, `/tokenize` | `embed-sparse` — sparse/lexical weights + tokenization |
| `/gliner` | `gliner` — zero-shot NER (Ray Serve) |
| `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension` | `clip` — CLIP image+text tower |
| `/diarize` | `diarize` — pyannote speaker diarization |
| `/vad` | `vad` — Silero voice activity detection |

## Authentication

Every route above — pass-throughs included — is gated by the router's master
key, so clients must send `Authorization: Bearer $OPENAI_API_KEY`. (The
standalone `-only` shapes have no router and no auth.) The
two exceptions are the operational endpoints `/health/liveliness` and
`/metrics`, both deliberately served unauthenticated on `inference-net` for
obs-plane — see [architecture.md](architecture.md#networking).

## Two addresses for the same contract

Every capability below can be reached two ways, and the request and response
bodies are identical either way:

- **Through the full stack** — at the LiteLLM router, with
  `Authorization: Bearer $OPENAI_API_KEY`. In-network that is
  `http://vllm-router:4000` — the alias and the container port
  (`docker/compose.yaml`, router `--port 4000`). `ROUTER_HOST_PORT` (default
  `9000`) is the *host* publish port `make up-dev` adds
  (`docker/compose.override.yaml:12`); it is not reachable under the alias,
  and the production shape publishes nothing at all.
- **Through a CPU-only standalone shape** — directly on that shape's
  container, port 8000 on `inference-net`, with no `Authorization` header.
  See [deployment-shapes.md](deployment-shapes.md).

Each section below gives both addresses. Consumers switch between them by
changing only the base URL (and dropping or adding the header).

## Embeddings

In the full stack these three routes land on **two** backends: `/v1/embeddings`
is dispatched by `model` field to the vLLM `embed` service, while `/pooling`
and `/tokenize` are pass-throughs to `embed-sparse` — the same image and the
same `BAAI/bge-m3`, run a second time with the `token_classify` pooler task
because vLLM binds one pooling task per server
(`docker/litellm.config.yaml:96-105`). The `embed-only` shape collapses the
pair back into one container, serving all three from one loaded model — dense
via `/v1/embeddings`, sparse via `/pooling` with `task: "token_classify"` — so
a consumer points both its embedding base and its sparse base at the same
container.

```bash
curl http://embed-only:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "what is RAG?"}'

curl http://embed-only:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"task": "token_classify", "input": ["what is RAG?"]}'

curl http://embed-only:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "what is RAG?"}'
```

Response shape for `/v1/embeddings` (OpenAI-compatible; the request's
`model` field is ignored, and the configured model is echoed back):

```json
{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.01, -0.02, 0.03]}],
 "model": "BAAI/bge-m3", "usage": {"prompt_tokens": 6, "total_tokens": 6}}
```

Response shape for `/pooling` (one per-token weight list per input, aligned
to that input's own non-padding token ids):

```json
{"model": "BAAI/bge-m3", "data": [{"index": 0, "data": [0.0, 0.31, 0.0, 0.42]}]}
```

Any `/pooling` `task` other than `token_classify` is rejected with HTTP
400 — this server implements no other pooling task.

`/v1/embeddings` is OpenAI-compatible; the request's `model` field is ignored
and the configured model is echoed back. Batch sizing is bounded server-side —
see [configuration.md](configuration.md#embedding-batch-budget).

## Rerank

The full stack reaches the vLLM `rerank` backend on the router's own
`/v1/rerank` route, selected by the request's `model` field (`RERANK_MODEL`);
the `rerank-only` shape exposes the same Jina-shape body at
`http://rerank-only:8000/rerank` (`src/rerank_server.py:96`). The path and the
`Authorization` header are the only differences.

```bash
curl http://rerank-only:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is RAG?",
    "documents": [
      "Retrieval-augmented generation grounds an LLM in retrieved context.",
      "The cafeteria menu changes daily.",
      "RAG combines a retriever and a generator model."
    ],
    "top_n": 2
  }'
```

Response:

```json
{
  "id": "rerank-...",
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {"index": 0, "relevance_score": 0.96},
    {"index": 2, "relevance_score": 0.91}
  ]
}
```

## CLIP

The full stack's router passes `/clip/embed_image`, `/clip/embed_text` and
`/clip/dimension` through to the `clip` backend; the `clip-only` shape exposes
the same contract at `http://clip-only:8000`.

```bash
# text tower
curl http://clip-only:8000/clip/embed_text \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'

# image tower (multipart)
curl http://clip-only:8000/clip/embed_image \
  -F "file=@/path/to/img.jpg"

# dimension probe (for Qdrant collection compat checks)
curl http://clip-only:8000/clip/dimension
```

Response shape for both embed endpoints:

```json
{"embedding": [0.012, -0.034, ...], "dimension": 512}
```

## Speech-to-text (ASR)

The `asr` service runs Whisper via vLLM and exposes OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/translations` endpoints.

```bash
curl http://vllm-router:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F model="$WHISPER_MODEL" \
  -F file="@recording.mp3"
```

The maximum accepted file size defaults to 200 MB and can be raised with
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` in `.env`.

The `asr-only` shape exposes the same contract on CPU at
`http://asr-only:8000/v1/audio/transcriptions` with no Bearer auth (CPU
openai-whisper instead of vLLM):

```bash
curl http://asr-only:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "response_format=verbose_json"
```

Response (`verbose_json`):

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.34,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "...", "no_speech_prob": 0.01}
  ]
}
```

No `Authorization` header is required: the container has no built-in
Bearer-token gate (same posture as the other `-only` shapes). The `model` form
field is accepted but ignored — the server always uses the model it loaded at
boot.

## Diarization

The `diarize` service runs the pipeline named by `DIARIZE_MODEL` — by default
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
— behind FastAPI. Like `gliner` and `clip` it is **not** vLLM and
does **not** expose OpenAI-compatible routes — it is reached through the
router's `/diarize` pass-through. The uploaded file may be any container
ffmpeg can decode (wav, mp3, m4a, mp4, ...); it is resampled to 16 kHz
mono server-side.

```bash
curl http://vllm-router:4000/diarize \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@recording.mp3" \
  -F "max_speakers=4"
```

Response:

```json
{
  "segments": [
    {"start": 0.51, "end": 4.32, "speaker": "SPEAKER_00"},
    {"start": 4.80, "end": 9.11, "speaker": "SPEAKER_01"}
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

`num_speakers` (exact count) is mutually exclusive with the
`min_speakers`/`max_speakers` bounds; sending both returns 400. Times are
absolute seconds — consumers (Nextext) assign speakers to their ASR
segments by maximum overlap client-side.

**VAD gating (full stack: on by default).** Before responding, the service
crops the pipeline's turns to the Silero speech timeline fetched from the
stack's `vad` service (`DIARIZE_VAD_URL`, set to `http://vad:8000` by the
full-stack compose), dropping the music/noise the diarizer over-detects as
speech — measured −35% false alarm / −12.5% DER at the tuned
`DIARIZE_VAD_THRESHOLD=0.4` / `DIARIZE_VAD_PAD_MS=100`
(`eval/reports/2026-07-11-false-alarm-vad-gating.md`). Fail-open: an
unreachable `vad` logs a warning and returns ungated turns; the response
shape never changes. `DIARIZE_VAD_GATE=off` disables it. In `diarize-only`
the compose hardcodes the URL to `http://vad-only:8000` (a shared `.env`'s
full-stack `http://vad:8000` would not resolve in that shape) — co-deploy
`vad-only` on `inference-net` and gating engages there too; without it the
gate fails open, and `DIARIZE_VAD_GATE=off` silences it. Consumers that
gate client-side (Nextext's `NEXTEXT_DIARIZE_VAD_GATE`) should disable
their gate once this is live — double-gating is harmless but wasteful.

The pipeline weights are gated on the Hugging Face Hub — see
[deployment-shapes.md](deployment-shapes.md#diarize-only) for the one-time
setup that populates the shared `huggingface-cache` volume.

The `diarize-only` shape exposes the same contract at
`http://diarize-only:8000/diarize` with no Bearer auth.

## VAD

The `vad` service runs [Silero VAD](https://github.com/snakers4/silero-vad)
behind FastAPI. Like `gliner`, `clip`, and `diarize` it is **not** vLLM and
does **not** expose OpenAI-compatible routes — it is reached through the
router's `/vad` pass-through. The uploaded file may be any container ffmpeg can
decode; it is resampled to 16 kHz mono server-side.

```bash
curl http://vllm-router:4000/vad \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@recording.mp3"
```

Response:

```json
{
  "segments": [
    {"start": 0.51, "end": 4.32},
    {"start": 4.80, "end": 9.11}
  ],
  "has_speech": true,
  "sampling_rate": 16000
}
```

Times are absolute seconds; the service returns raw speech turns, leaving the
speech / no-speech reduction to the consumer. Optional `threshold`,
`min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, and
`max_speech_duration_s` form fields tune Silero; omitted ones use its defaults.
The Silero weights ship inside the `silero-vad` package, so no download or HF
token is needed.

The `vad-only` shape exposes the same contract at `http://vad-only:8000/vad`
with no Bearer auth.

## NER

The `gliner` service runs [GLiNER](https://github.com/urchade/GLiNER), a
zero-shot Named Entity Recognition model, behind Ray Serve. Unlike the
other backends it is **not** vLLM and does **not** expose OpenAI-compatible
routes — its request/response shape is GLiNER-native, and it is reached
through the router's `/gliner` pass-through:

```bash
curl http://vllm-router:4000/gliner \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at Acme Corp in Berlin.",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5
  }'
```

Response:

```json
{
  "entities": [
    {"start": 0,  "end": 5,  "text": "Alice",     "label": "person",       "score": 0.97},
    {"start": 15, "end": 24, "text": "Acme Corp", "label": "organization", "score": 0.92},
    {"start": 28, "end": 34, "text": "Berlin",    "label": "location",     "score": 0.95}
  ]
}
```

`labels` is the candidate set for this single request — GLiNER is
zero-shot, so labels can change request to request without retraining.

The default model is `gliner-community/gliner_large-v2.5` on CUDA. For
CPU-only hosts, set both `NER_MODEL=gliner-community/gliner_medium-v2.5`
and `NER_DEVICE=cpu` in `.env`. See `.env.example` for the full list of
`NER_*` knobs (dtype, batch size, FlashDeBERTa, sequence packing, etc.).

The `gliner-only` shape exposes the same contract at
`http://gliner-only:8000/gliner` with no Bearer auth.
