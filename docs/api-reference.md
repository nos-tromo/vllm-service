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
declared as a `model_list` entry (`RERANK_MODEL` → `http://rerank:8000`, via
the `hosted_vllm` provider) and picked by the request's `model` field like any
other — see `docker/litellm.config.yaml:39-48`.

**By path**, for the backends whose contracts are not OpenAI-shaped.
`pass_through_endpoints` (`docker/litellm.config.yaml:95-165`) forwards the
request body verbatim; a `model` field means nothing on these:

| Path | Backend |
|---|---|
| `/pooling`, `/tokenize` | `embed-sparse` — sparse/lexical weights + tokenization |
| `/gliner` | `gliner` — zero-shot NER (Ray Serve) |
| `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension` | `clip` — CLIP image+text tower |
| `/diarize` | `diarize` — pyannote speaker diarization |
| `/vad` | `vad` — Silero voice activity detection |
| `/topics/fit` | `topics` — topic-model fit over caller-supplied vectors (CPU) |

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
gate fails open, and `DIARIZE_VAD_GATE=off` silences it. Gating is
server-side only now: Nextext dropped its client-side knob, so consumers get
pre-gated turns with nothing to configure on their end.

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

## Topics

The `topics` service fits a topic model to vectors the caller already holds. The
caller posts the vectors and the matching texts and gets back one topic label per
input and a keyword list per topic. The fit is
[BERTopic](https://maartengr.github.io/BERTopic/) (UMAP → HDBSCAN → c-TF-IDF)
with no embedding model, so nothing is embedded, downloaded or stored: the
service keeps no state between requests. Like `gliner`, `clip`, `diarize` and
`vad` it is **not** vLLM and does **not** expose OpenAI-compatible routes — it is
reached through the router's `/topics/fit` pass-through. Both addresses take the
same request and return the same response:

- **Full stack** — `POST http://vllm-router:4000/topics/fit` with
  `Authorization: Bearer $OPENAI_API_KEY`.
- **`topics-only` shape** — `POST http://topics-only:8000/topics/fit`, no Bearer
  auth.

### Request

The body is `multipart/form-data`, and every part must be a **file part** (sent
with a filename, as `curl -F name=@file` does):

| Part | Filename | Content |
|---|---|---|
| `meta` | `meta.json` | The settings below; every field is required |
| `texts` | `texts.json` | A JSON array of `n` strings — the c-TF-IDF input |
| `vectors` | `vectors.f32` | `n × dim` little-endian float32, row-major |

| `meta` field | Meaning | Bounds |
|---|---|---|
| `n` | Number of points: the length of `texts` and the row count of `vectors` | ≥ 1 |
| `dim` | Vector dimension | 1 to 8192 |
| `seed` | UMAP's random state | 0 to 2³² − 1 |
| `min_topic_size` | HDBSCAN's `min_cluster_size`; HDBSCAN's `min_samples` is the same value capped at 100 | ≥ 2 |
| `nr_topics` | Topic-reduction target, or `0` for none; BERTopic counts the outlier class as one of the topics | `0` or ≥ 2 |
| `top_n_words` | Keywords kept per topic | 1 to 30 |
| `min_df` | The fewest texts a word must occur in to become a keyword | ≥ 1 |
| `stop_words` | Words never used as keywords; they are matched against the lowercased tokens, so send them lowercase | at most 10000, each at most 64 characters |

```bash
curl http://vllm-router:4000/topics/fit \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "meta=@meta.json;type=application/json" \
  -F "texts=@texts.json;type=application/json" \
  -F "vectors=@vectors.f32;type=application/octet-stream"
```

with, for example, this `meta.json`:

```json
{"n": 20000, "dim": 1024, "seed": 42, "min_topic_size": 20, "nr_topics": 0,
 "top_n_words": 10, "min_df": 2, "stop_words": []}
```

### Response

```json
{
  "labels": [0, -1, 2, ...],
  "outliers": 123,
  "topics": [{"id": 0, "size": 410, "keywords": [["word", 0.031], ...]}, ...],
  "versions": {"bertopic": "0.17.4", "umap-learn": "0.5.12", "hdbscan": "0.8.44", ...},
  "runtime": {"machine": "x86_64", "python": "3.11.x", "threads": {"OPENBLAS_NUM_THREADS": "1", ...}},
  "timings": {"fit_s": 12.3}
}
```

- `labels[i]` is the topic of input `i`, in the order sent. `-1` marks an
  outlier, and `outliers` counts them.
- `topics` lists the ids ≥ 0 in ascending **id** order. Ids are not ordered by
  size, so a consumer that wants the largest topic first sorts on `size` itself.
- A topic's `size` counts every input row that carries its id, copies of a
  duplicated vector included, so `sum(size) + outliers == n`.
- `keywords` are `[word, score]` pairs, best first: the c-TF-IDF score rounded to
  6 decimals, highest first, and words with equal scores in code-point order of
  the word.
- `versions` names the installed versions of the fit's libraries, `runtime` the
  CPU architecture, the Python version and the thread pins, and `timings.fit_s`
  the wall time of the fit itself, in seconds.

### Fit behaviour

- **Minimum input.** A fit needs at least `max(16, 2·min_topic_size)` points: UMAP
  needs more points than its 15 neighbours, and HDBSCAN room for two topics.
- **Exact duplicates are clustered once.** The fit runs on the distinct vectors in
  first-occurrence order (`-0.0` and `0.0` are the same value), each paired with
  its first occurrence's text, and copies inherit their representative's label.
  The texts of the copies are not used, for keywords or for `min_df`, while
  `size` counts every row. `min_topic_size` counts distinct vectors, so copies of
  one vector never add up to a topic. If the points meet the minimum above but
  the distinct vectors do not, the answer is a `200`, not a `422`: every label is
  `-1`, `topics` is empty, and UMAP and HDBSCAN do not run.
- **Texts are used as sent.** They are not stripped to ASCII, so non-Latin scripts
  and diacritics reach the keywords unchanged. Words are scikit-learn tokens:
  lowercased, two or more word characters. The vocabulary is fitted on the
  individual texts, so `min_df` counts texts rather than topics, and is fixed
  before clustering. When fewer than two words survive `stop_words` and `min_df`,
  every topic's `keywords` is empty; the clustering is unchanged.
- **One fit at a time.** A fit request that arrives while another fit is being
  received or run gets a 503 with `Retry-After: 30` before any of its body is
  read; requests are not queued.
- **Duration.** A fit takes seconds to minutes. Measured on synthetic data at 1024
  dimensions on one Apple Silicon core: about 90 s for 100k points with six-word
  texts, and 57 s for 50k and 180 s for 100k with texts of about 2,000 characters.
  It grows faster than `n`, and the shape of the data moves it as well: one
  512-dimensional set of 50k points took 244 s. The first fit with 4096 or more
  distinct vectors in a process also compiles pynndescent's numba code once,
  adding about 6–7 s to that request.
  Clients need a read timeout to match; the router's pass-through timeout is 4 h.

### Errors

| Status | When |
|---|---|
| `400` | A malformed `meta.json` (including a `dim` above 8192 or a stop word over 64 characters), or a multipart body that names no boundary |
| `408` | An admitted fit's body has not arrived within 300 s; the connection is closed |
| `413` | A body over the request ceiling (below), declared in `Content-Length` or counted as it arrives; `n > TOPICS_MAX_POINTS`, `n·dim > TOPICS_MAX_POINTS·1024`, a `meta.json` over 1 MiB, or a `texts` part over `min(n·16384 + 65536, 536870912)` bytes. Each is decided from the body's length, `meta.json` and the parts' sizes alone, before a large part is read into memory |
| `422` | A missing part (including a body that is not multipart), a `texts` part that is not a JSON array of strings, `len(texts) != n`, a vector byte count other than `n·dim·4`, non-finite values, or fewer points than the minimum, `max(16, 2·min_topic_size)` |
| `503` + `Retry-After` | Another fit is being received or run (one at a time; requests are not queued). Answered before any of the body is read |
| `500` | The fit itself failed; the body is a fixed message and quotes nothing from the request |

An error body is `{"detail": ...}`. A 422's `detail` is a string from the service
(`texts holds 19 items, expected n = 20`), or FastAPI's list form for a missing
part (`[{"type": "missing", "loc": ["body", "vectors"], "msg": "Field required", ...}]`),
so a client has to handle both. A multipart body with no boundary is a 400 from
Starlette (`Missing boundary in multipart.`).

A part sent without a filename reaches the server as a form field, not a file.
Above 1 MiB Starlette refuses it with a 400 (`Part exceeded maximum size of
1024KB.`); below that it is a 422 in FastAPI's list form
(`"msg": "Value error, Expected UploadFile, received: <class 'str'>"`).

The upload is bounded before it is parsed. A 503, or a 413 for a declared
`Content-Length` over the request ceiling, is answered before any of the body is
read: a client that waits for `100 Continue` is never asked for its body, and what
a client sends anyway is read and dropped, not stored. The ceiling is `meta.json`'s
1 MiB, plus the `texts` limit at `TOPICS_MAX_POINTS` points, plus
`TOPICS_MAX_POINTS·1024·4` bytes of vectors, plus 64 KiB of multipart framing:
1,152,385,024 bytes at the default. An admitted fit holds the one slot from the
first byte of its body until its response is sent; the body is counted as it
arrives, chunked or not, and cut off with a 413 past the ceiling or a 408 after
300 s. Through the full stack's router this spares the service but not the router,
which takes every multipart body in full before it forwards it. The disk a body
takes, the router's share, and the memory a fit needs for a given
`TOPICS_MAX_POINTS`, are in
[configuration.md](configuration.md#topic-model-fit-limits).

A request whose fit outgrows the container's memory limit (`TOPICS_MEM_LIMIT`)
gets none of these answers. The kernel kills the topics container, so a direct
call sees the connection close with no response, and a call through the router
gets a 500 from LiteLLM, `{"error": {"message": "Server disconnected", ...}}`,
not the service's `{"detail": ...}`. The container then restarts and passes its
self-check again before it answers.

### Determinism

The same input bytes, settings, image and CPU class give the same labels and
keywords. The service never reorders or samples the input, so its order is the
caller's. A fixed `seed` makes UMAP single-threaded (`n_jobs=1`), HDBSCAN
computes core distances on one thread, and the image pins the BLAS and numba
thread pools to 1 (`runtime.threads` reports the four variables), so a result
does not depend on the host's core count. numba runs its `workqueue` threading
layer because the image has no libgomp; that is harmless with the pins at 1.

Checked: identical output with the container pinned to 1, 2, 4 and 8 cores
(`--cpuset-cpus`), each with the thread variables at 1, 2, 4 and 8. Across
architectures the output can differ: an arm64 image and an emulated amd64 one
agree on 2,000 points of 64 dimensions, but on 6,000 points of 1024 dimensions,
which take the approximate-neighbour path, they found 105 and 106 topics. That,
and the x86_64 CPU models left unchecked, is why the guarantee is scoped to a CPU
class. The server also fits a synthetic set twice at start and refuses to serve
if the two results differ, so `GET /health`
(`{"status": "ok", "versions": {...}}`, the container's healthcheck target)
answers only once that check has passed.
