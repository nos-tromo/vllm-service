# Configuration

Everything is env-driven: `.env` is the source of truth, `.env.example` ships
the annotated full set, and `docker/litellm.config.yaml` reads model names from
the environment at startup. This document covers the settings that need more
than a one-line comment. Per-shape defaults for the CPU-only containers live in
[deployment-shapes.md](deployment-shapes.md).

## Master key

`OPENAI_API_KEY` is the router's master key: every client sends it as its
Bearer token, and the router passes it on to the vLLM backends, whose
`--api-key` it is too. The apps (docint, Nextext, translator,
open-webui-service) hold the same value, and deploy's `make env` copies it from
this repo's `.env` to theirs. `.env.example` leaves it empty, and compose
refuses the full stack without one:

```
required variable OPENAI_API_KEY is missing a value: OPENAI_API_KEY must be set in .env (the router master key; the apps use the same value)
```

Compose checks the key where it delivers it, in the router's `environment:`,
so the check and the router read the same value and the router never boots
without a master key. Every command that loads `docker/compose.yaml` stops
this way, as data-plane's does for `QDRANT_API_KEY`, so set the key before the
first `make build` or `make bundle` (any value will do on a build host; images
do not carry it); `make down` and deploy's `make ps` need it too. The CPU-only
shapes run no router and do not read it.

## Shell precedence

Except for the settings in the table below, every setting reaches the
containers from env files only, so a value exported in the shell that runs
`make up` does not apply: `.env` is the one source. The router reads
`docker/router/defaults.env` first, which holds its model names (`TEXT_MODEL`,
`EMBED_MODEL`, `RERANK_MODEL`, `WHISPER_MODEL`, at the defaults the backends
start with) and the HF flags, then `.env`, which wins for every key it sets.
Router and backends therefore agree on every model name the router routes by,
whatever the shell exports. The topics image sets the HF flags itself, and
`.env` overrides them there too.

The exceptions are the settings compose itself fills in with `${KEY:-default}`
(the master key with `${KEY:?…}`): for those, an exported value beats `.env`,
and set in neither, a key takes the compose default, the value `.env.example`
shows:

| Compose file | Settings filled in by compose |
|---|---|
| `compose.yaml` | the router's `OPENAI_API_KEY`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` (every service but router and topics), `PYTORCH_CUDA_ALLOC_CONF`, `CLIP_MODEL`, `CLIP_DEVICE`, `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `VLLM_MAX_AUDIO_DECODE_DURATION_S`, `DIARIZE_MODEL`, `DIARIZE_DEVICE`, `DIARIZE_VAD_URL`, `DIARIZE_VAD_GATE`, `DIARIZE_VAD_THRESHOLD`, `DIARIZE_VAD_PAD_MS`, `DIARIZE_VAD_TIMEOUT`, `VAD_MODEL`, `VAD_USE_ONNX`, `NER_RAY_MEMORY_THRESHOLD`, `TOPICS_MAX_POINTS`, `TOPICS_MEM_LIMIT`, `FACE_DETECT_MODEL`, `FACE_DETECT_MODEL_FILE`, `FACE_EMBED_MODEL`, `FACE_EMBED_MODEL_FILE`, `FACE_GROUP_THRESHOLD`, `FACE_JOIN_THRESHOLD` |
| the CPU-only shapes | the HF flags (all but topics-only), `WHISPER_MODEL` (asr-only), `CLIP_MODEL` (clip-only), `EMBED_MODEL`, `EMBED_MAX_LENGTH`, `EMBED_MAX_BATCH_TOKENS` (embed-only), `RERANK_MODEL`, `RERANK_USE_FP16` (rerank-only), `DIARIZE_MODEL` (diarize-only), `VAD_MODEL`, `VAD_USE_ONNX` (vad-only), `NER_SHM_SIZE` (gliner-only), `TOPICS_MAX_POINTS`, `TOPICS_MEM_LIMIT` (topics-only), `FACE_DETECT_MODEL`, `FACE_DETECT_MODEL_FILE`, `FACE_EMBED_MODEL`, `FACE_EMBED_MODEL_FILE`, `FACE_GROUP_THRESHOLD`, `FACE_JOIN_THRESHOLD` (faces-only) |
| every file | `INFERENCE_NETWORK`, `EXTRA_NO_PROXY`, and the dev overlays' `*_HOST_PORT` |

`OPENAI_API_KEY` is the one router setting a shell export overrides: compose
delivers it to the router through `environment:`, where its check reads it,
while the vLLM backends take theirs from `.env`. An exported key that differs
from `.env` therefore fails closed, with 401s from the backends on every call;
it never opens the router. Set the key in `.env`, and do not export it in the
shell that runs `make up`.
`env | cut -d= -f1 | grep -xE 'OPENAI_API_KEY|[A-Z_]+_MODEL|HF_HUB_OFFLINE'`
lists, by name only, the exports that could matter.

## Tool calling

The `chat` backend starts with auto tool choice on, using the tool-call
parser for the default `TEXT_MODEL` (`Qwen/Qwen3.8-27B-FP8`):

```bash
# Default. false turns tool calling off.
CHAT_ENABLE_AUTO_TOOL_CHOICE=true
# Default. An empty value passes no parser, which also turns tool calling off.
CHAT_TOOL_CALL_PARSER=qwen3_xml
```

Keep comments on their own line, as above: after an empty value Compose keeps
a trailing `# …` as the value, and as a parser name that stops `chat` at
startup.

Without these flags the pinned vLLM release rejects a request that sends
`tools` with HTTP 400 unless its `tool_choice` is `"none"`: `"auto"` (the
default once `tools` is set), `"required"` and a named function all need them.

The parser has to match the model's tool-call format. The default model's
chat template writes calls as `<tool_call><function=…><parameter=…>` XML,
which is what `qwen3_xml` reads (`qwen3_coder` is another name for the same
parser in the pinned release). Switching `TEXT_MODEL` to another family
therefore means switching `CHAT_TOOL_CALL_PARSER` with it; vLLM's
tool-calling documentation lists the parser for each family. A parser that
does not match does not stop the server from starting — it just does not
recognise the model's tool calls. A name vLLM does not register (a typo, the
reasoning parser's name, one from newer upstream documentation) does stop
`chat` at startup, and its log lists the valid names. A `.env` that sets only
`CHAT_TOOL_CALL_PARSER` now takes effect, and its name is checked: it used to
be ignored while the toggle defaulted to off.

## Embedding batch budget

Applies to the `embed-only` shape (`src/embed_server.py`).

Both batch routes bound how much work one forward pass does. Because a
batch pads to its longest text, a single long chunk inflates the whole
batch — 64 short texts alongside one 4k-token chunk pad to ~262k tokens.
`/v1/embeddings` and `/pooling` therefore split their input into
sub-batches of at most `EMBED_MAX_BATCH_TOKENS` padded tokens (rows ×
longest row) and concatenate the results, so a client's batch size does
not decide the cost of a pass. Responses are unaffected: one entry per
input, in input order. A single text over the budget still gets its own
pass rather than being dropped — `EMBED_MAX_LENGTH` truncation is the
only cap that drops content.

Override defaults via `.env` — only `EMBED_*` knobs apply in this shape:

```bash
EMBED_MODEL=BAAI/bge-m3        # default
EMBED_MAX_LENGTH=8192          # default
EMBED_MAX_BATCH_TOKENS=16384   # default; padded tokens per forward pass
# EMBED_HOST_PORT=8007         # host publish port for dev
```

## Diarization speaker granularity

Applies to both the full-stack `diarize` service and the `diarize-only` shape.
Override defaults via `.env`:

```bash
DIARIZE_MODEL=pyannote/speaker-diarization-community-1   # default (pyannote.audio 4.x; gated)
DIARIZE_DEVICE=cuda                              # full stack; diarize-only pins cpu
# DIARIZE_HOST_PORT=8004                          # host publish port for dev
# DIARIZE_FB=0.4                                  # community-1 clustering: LOWER → MORE speakers; 0.4 validated best
# DIARIZE_FA=0.07                                 # community-1 PLDA companion weight (leave at stock 0.07)
# DIARIZE_CLUSTERING_THRESHOLD / DIARIZE_SEG_MIN_DURATION_OFF   # further overrides (threshold inert for community-1)
```

`DIARIZE_FB`/`DIARIZE_FA` are community-1's speaker-granularity knobs (unset →
stock defaults, Fb 0.8 / Fa 0.07). `Fb=0.4` is the deploy value — best
speaker-attribution accuracy and turn precision on real labeled clips
(`eval/reports/2026-07-14-fb-realdata-validation.md`; the benchmark sweep's
provisional 0.2 over-splits real content — see
`eval/reports/2026-07-14-fa-fb-sweep.md`). Leave `Fa` at stock — both
directions measured worse. Validate on labelled clips with `eval/` (the
transcript metric + `--fb` sweep) before deploying a different value; an
unparseable value warns and is ignored rather than crashing startup.

## Topic-model fit limits

Applies to both the full-stack `topics` service and the `topics-only` shape. A fit
holds its whole request in memory for as long as it runs, and the request arrives
on disk first, so a few settings decide what one request may cost. Override
defaults via `.env`:

```bash
TOPICS_MAX_POINTS=150000   # default; the most points one request may carry
TOPICS_MEM_LIMIT=7g        # default; the container's memory limit (compose mem_limit)
# TOPICS_HOST_PORT=8008    # host publish port for dev (topics-only)
# TOPICS_WARMUP=true       # image default, not in .env.example; false skips the startup self-check
```

`TOPICS_MAX_POINTS` bounds `n`: above it a request is refused with a 413, decided
from `meta.json` alone. It also caps `n × dim` at `TOPICS_MAX_POINTS × 1024` (a 413
too), so a wider vector means fewer points, not a larger peak. And it sets the
largest body a request may send: `meta.json`'s 1 MiB, plus the `texts` limit at that
many points, plus `TOPICS_MAX_POINTS × 1024 × 4` bytes of vectors, plus 64 KiB of
multipart framing, which is 1,152,385,024 bytes at the default. A value that is not
a positive integer stops the server at startup.

**Memory.** A fit's peak grows with the number of points and with what each one
carries. Measured on the real server, natively on one Apple Silicon core, with
synthetic requests (1024-dimensional vectors, texts of about 2,000 characters,
`min_topic_size` 100, 10% exact duplicates, about a third of the points outliers),
one request peaked at 2.3 GB with 50,000 points and 3.9 GB with 100,000, of which
the idle server is about 0.5 GB. Extrapolated linearly, the default of 150,000
points peaks at about 5.4 GB. What drives it, measured at 50,000 points:

- **Points:** each added about 31 KB between 50,000 and 100,000 at those settings,
  and almost every part of the fit scales with them.
- **Dimensions:** 512 instead of 1024 saved 0.44 GB, about four times the vector
  bytes left out, because the vectors are held several times over: the request
  bytes, the distinct rows the fit runs on, and a transient while duplicates are
  found. Since `n × dim` is capped, a wider vector means fewer points.
- **Text length:** texts of 200 instead of 2,000 characters saved 0.24 GB, about
  2.7 times the text left out. Each text is tokenized for the vocabulary, and again,
  joined with the rest of its topic, for c-TF-IDF. The `texts` part may hold up to
  512 MiB, so longer texts can cost well beyond these figures.
- **Duplicates:** roughly neutral at 10%. Copies are not fitted, but the distinct
  rows are copied out of the request; the same 50,000 points with no duplicates
  peaked 0.18 GB higher.
- **`min_topic_size`:** bounded. HDBSCAN's `min_samples`, half of it, is capped at 100, so its
  neighbour table stays under `n × 101 × 16` bytes, about 0.24 GB at 150,000
  points.

`TOPICS_MAX_POINTS` therefore does not bound memory on its own. `TOPICS_MEM_LIMIT`
sets compose's `mem_limit` on both topics services: the default, `7g`, is the
extrapolated 150,000-point peak plus 25%, rounded up to a whole GiB. Checked on
Linux in the `topics-only` container, a 150,000-point request of the same shape
peaked at 4.3 GB resident, 5.1 GB for the container with its page cache, and used
no swap. A request that outgrows the limit kills only the topics container, never
another service: a direct client sees the connection close with no response, a
client of the router gets a 500 from LiteLLM (`Server disconnected`), and the
container restarts and runs its self-check again before it serves. With swap on
the host, Docker lets the container use as much swap again before the kill. In the
full stack the fit shares the host with chat, embed and gliner; raise either
setting only as far as the host's headroom allows.

**Disk.** Starlette spools each uploaded part to a temporary file once it passes
1 MiB, in `/tmp`, the disk-backed `media-tmp` volume (shared with the router, `asr`,
`diarize` and `vad` in the full stack). The service lets one body in at a time and
bounds it before it is parsed. A fit request that arrives while another is being
received or run gets its 503, and a declared length over the ceiling above gets its
413, before any of the body is read; what such a client sends anyway is read and
dropped, never stored. An admitted body is counted as it arrives, chunked or not,
and cut off with a 413 past the ceiling or a 408 after 300 s, so a stalled client
cannot hold the slot. At the default limit the `vectors` part alone is up to about
600 MB (150000 × 1024 × 4 bytes).

The router is not bounded that way. LiteLLM v1.97.0 reads every multipart body in
full before it checks the key, on every route: it spools the parts onto the same
volume, then reads every file part into memory and re-encodes the multipart body
with a new boundary for the backend (the part names, filenames and content types
survive). In the full stack a request in flight therefore occupies `media-tmp`
twice, and at the caps about 1.1 GB of router memory (the vectors plus a 512 MiB
`texts` part), including a request the service then refuses with a 503 or a 413.
LiteLLM's own request-size limit (`max_request_size_mb`) takes effect only with an
enterprise license; a router-wide limit is tracked in
[issue #158](https://github.com/nos-tromo/vllm-service/issues/158).

**Startup.** The server runs a self-check before it listens: it compiles the numba
code, fits a synthetic set twice, and stops if the two results differ. `/health`
therefore passes only about 11 to 15 s after the container starts, as measured (the
compose `start_period` is 180 s, for slower hosts), and a failing check makes the
container restart instead of serve. The first fit with 4096 or more distinct
vectors compiles pynndescent's numba code once more, about 6–7 s on that request.
numba caches what it compiles in `NUMBA_CACHE_DIR`, `/run/numba`, a 64 MB tmpfs
of the container's own: numba unpickles the cache's index files, so the cache stays
off the shared `media-tmp`. The tmpfs starts empty, so every start compiles afresh.
`TOPICS_WARMUP` defaults to `true` in the image; `false`, `0`, `no` or `off` skip
the check. Skipping it moves the compile onto the first request and leaves nothing
showing that fits repeat on the host, so it is meant for tests, not for a
deployment.

**Threads.** A fit runs on one core. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` are set to 1, which keeps a result
independent of the host's core count. The image sets them, and both compose files
set them again under `environment:`, because a value in the shared `.env` would
override the image's (`env_file` takes precedence over the image) but cannot
override `environment:`. Every response reports them under `runtime.threads`.

The request and response bodies, and the errors these limits produce, are in
[api-reference.md](api-reference.md#topics).

## Faces

Applies to both the full-stack `faces` service and the `faces-only` shape.
Override defaults via `.env`:

```bash
FACE_DETECT_MODEL=opencv/face_detection_yunet               # default; the detection model's repository
FACE_DETECT_MODEL_FILE=face_detection_yunet_2023mar.onnx     # default; the file in it
FACE_EMBED_MODEL=opencv/face_recognition_sface              # default; the feature model's repository
FACE_EMBED_MODEL_FILE=face_recognition_sface_2021dec.onnx    # default; the file in it
# FACE_DETECT_MODEL_REVISION=<commit>
# FACE_EMBED_MODEL_REVISION=<commit>
# FACE_GROUP_THRESHOLD=0.42             # lowest mean cosine at which two groups merge
# FACES_HOST_PORT=8009                  # host publish port for dev (faces-only)
```

**Files.** Each repository holds the model in several variants (fp32, int8,
int8bq), so the file is configured beside the repository; the defaults are the
fp32 files. Both come from the `huggingface-cache` volume, offline whatever
`HF_HUB_OFFLINE` says (a `0` left in `.env` from a gated first fetch does not
make the service download); only the two files and each repository's `LICENSE` belong there (the repositories also
hold `example_outputs/` with photographs of people). YuNet is MIT-licensed and
SFace Apache-2.0, both from the OpenCV Zoo. InsightFace's SCRFD/ArcFace models
are not used because their weights are licensed for non-commercial research
only.

Fetch only those files, with the deploy repo's `scripts/fetch-model.sh` and its
repeatable `--file` option (given before the model id); each run prints
`Revision:`, the commit for that model's revision variable:

```bash
./fetch-model.sh --file face_detection_yunet_2023mar.onnx --file LICENSE opencv/face_detection_yunet      # → FACE_DETECT_MODEL_REVISION
./fetch-model.sh --file face_recognition_sface_2021dec.onnx --file LICENSE opencv/face_recognition_sface  # → FACE_EMBED_MODEL_REVISION
```

A whole-repository fetch would bring along the repositories' `example_outputs/`,
photographs of people, which have no place on a deployment host. More in
[deployment-shapes.md](deployment-shapes.md#faces-only).

**Upgrading.** A host upgraded to the release that adds `faces` needs the two
files in `huggingface-cache` before the service can start. Without them only
`faces` is down: the router waits for it to start, not to be healthy, so the
rest of inference comes up and only `/faces/detect` and `/faces/group` fail.

**Revisions.** One backend, two models: each loads at its own revision
variable, and the router records each per route (see
[Model revisions](#model-revisions)).

**Grouping threshold.** `FACE_GROUP_THRESHOLD` is the lowest average cosine
between two groups of tracks at which `/faces/group` still merges them. The
default `0.42` is stricter than OpenCV's 0.363 for a single same-person pair,
because merging two people is worse than splitting one: raise it to split more,
lower it to merge more. A value that is not a number in `[0, 1]` stops the
server at startup. Tracks that were on screen together are never merged,
whatever the threshold. The same threshold decides whether a weak track (one
the client sends with `"anchor": false`, without a good face) joins a person;
weak tracks never found or link people
([api-reference.md](api-reference.md#post-facesgroup)).

**Limits.** A request may carry at most 8 images (detect) or 500 tracks of at
most 3 faces and 3600 sample indices each, 100,000 sample indices in all
(group), in a body of at most 32 MiB, which the service reads into memory — so
`/tmp` is a 64 MB tmpfs, not `media-tmp`. Each image is decoded, downscaled to
1280 px on its long side and run before the next is decoded. An image that
decodes to more than 3840 × 2160 pixels is undecodable: `OPENCV_IO_MAX_IMAGE_PIXELS`
(set in the image, and by the server before OpenCV loads) makes OpenCV refuse
the canvas before allocating it, so a small, highly compressed file cannot
expand into gigabytes. A group crop over 512 px on a side counts as undecodable
too. At most four `/faces/*` requests are in flight at once, from before the
body is read until the response is built; a fifth gets `503` with
`Retry-After: 5` at once, before any of its body is read, so at most four
bodies and their decoded images are in memory. An admitted body must arrive
within 30 s, with no single read stalling for more than 10 s; otherwise the
request gets `408` and its connection is closed, so a slow client cannot keep a
slot. Decoding and the checks run
outside the model lock, which is held only for the detector and recognizer
calls, one call at a time: a request that waits more than 30 s for it, or runs
past its 120 s deadline, gets the same `503`. `/health` runs
on the event loop, so a busy worker pool cannot starve the probe. The 32 MiB
limit is the faces service's own: in the full stack the router enforces no body
limit at all, and reads the whole JSON body into memory, whatever its size,
before it forwards it, so a body the service then refuses with a 413 has
already been held by the router in full
([issue #158](https://github.com/nos-tromo/vllm-service/issues/158)).

The request and response bodies are in [api-reference.md](api-reference.md#faces).

## Audit log

The router writes one data-protection audit event per model call
([audit.md](audit.md)). Two settings go in `.env`:

```bash
# Default: empty, every call type is audited. Otherwise a comma-separated list of
# completion, text_completion, embedding, rerank, transcription, pass_through_endpoint, ...
# An unknown name stops the router.
AUDIT_ROUTER_CALL_TYPES=
# The secret open-webui-service signs its user token with; unset, no Open WebUI user is named.
AUDIT_OPENWEBUI_JWT_SECRET=
```

Leave a call type out only if every app that calls it records those calls
itself: the router then keeps no record of them. Compose sets the rest
(`AUDIT_OUTPUT`, `AUDIT_SERVICE_VERSION`, the callback's `PYTHONPATH` and
startup hook) in the router's `environment:`, where a shared `.env` cannot
change them, and pins `LITELLM_LOG=WARNING`, because LiteLLM's debug logging
prints request bodies.

## Updating the model catalog

`docker/litellm.config.yaml` is model-agnostic: all model names are read at
startup from the environment variables `TEXT_MODEL`, `EMBED_MODEL`,
`RERANK_MODEL`, and `WHISPER_MODEL`. The router falls back to
`docker/router/defaults.env`, the defaults its backends start with, so a name
left unset in `.env` still matches the model the backend serves. To switch a
model, update the relevant variable in `.env` and restart the stack. No changes to
`docker/litellm.config.yaml` are required. Switching `TEXT_MODEL` to another
model family also needs the matching `CHAT_TOOL_CALL_PARSER`: a mismatch does
not fail loudly, the model's tool calls just go unrecognised (see
[Tool calling](#tool-calling)).

Clients must use the exact model ID set in `.env` as the `model` field in
their requests (e.g. `"model": "BAAI/bge-m3"`). The `/v1/models` endpoint
returns the currently active IDs.

When you switch a model, update its revision with it (see
[Model revisions](#model-revisions)): `TEXT_MODEL_REVISION`,
`EMBED_MODEL_REVISION`, `RERANK_MODEL_REVISION` and `WHISPER_MODEL_REVISION`
name the commit the backend loads, and the audit log reports it for every call
to that deployment. A revision left from the previous model stops the backend
at startup, since the new model has no such commit.

`NER_MODEL`, `CLIP_MODEL`, `DIARIZE_MODEL`, `VAD_MODEL`, `FACE_DETECT_MODEL`
and `FACE_EMBED_MODEL` are the exceptions: their servers have no OpenAI-shaped
endpoints, so they are not in `model_list` and do not appear in `/v1/models`.
Switching them still works by updating the variable in `.env` (with
`CLIP_MODEL_REVISION`, `DIARIZE_MODEL_REVISION` or the `FACE_*_MODEL_REVISION`
for those that load by revision, and `FACE_*_MODEL_FILE` for the faces models)
and restarting the matching service (`gliner`, `clip`, `diarize`, `vad`,
`faces`).

## Model revisions

The services below load the commit their revision variable names, so the
weights a service serves are the ones recorded with the release. Unset or
empty, a service gets no revision and serves whatever the cache's
`refs/main` points at, as before.

| Variable | Services | Passed as |
|---|---|---|
| `TEXT_MODEL_REVISION` | `chat` | vLLM `--revision` and `--tokenizer-revision` |
| `EMBED_MODEL_REVISION` | `embed`, `embed-sparse`, `embed-only` | the vLLM flags; in `embed-only`, `revision=` for the encoder, its tokenizer and `sparse_linear.pt` |
| `RERANK_MODEL_REVISION` | `rerank`, `rerank-only` | the vLLM flags; in `rerank-only`, `revision=` for the model and its tokenizer |
| `WHISPER_MODEL_REVISION` | `asr` | the vLLM flags |
| `CLIP_MODEL_REVISION` | `clip`, `clip-only` | `revision=` for the model and its processor |
| `DIARIZE_MODEL_REVISION` | `diarize`, `diarize-only` | pyannote's `revision=` |
| `FACE_DETECT_MODEL_REVISION` | `faces`, `faces-only` | `hf_hub_download`'s `revision=` for `FACE_DETECT_MODEL_FILE` |
| `FACE_EMBED_MODEL_REVISION` | `faces`, `faces-only` | `hf_hub_download`'s `revision=` for `FACE_EMBED_MODEL_FILE` |

The router records each as `ai.revision` in the audit events of the calls it
forwards: the first four through `model_info.revision` in
`docker/litellm.config.yaml`, those of the pass-through backends
(`embed-sparse`, `clip`, `diarize`) through `PASS_THROUGH_REVISIONS` in
`docker/router/router_audit.py`, and the two of `faces`, which serves a model
per route, through `PASS_THROUGH_ROUTE_REVISIONS` there (`/faces/detect` →
`FACE_DETECT_MODEL_REVISION`, `/faces/group` → `FACE_EMBED_MODEL_REVISION`),
which the router checks before the backend's entry. Router and backends read them from the same
`.env`, so the revision in an audit event is the one the deployment loaded.

The value is the full 40-character commit id: what deploy's `fetch-model.sh`
prints as `Revision:`, and the name of the model's `snapshots/<commit>`
directory in the cache. With `HF_HUB_OFFLINE=1` (the default), vLLM turns the
model id into that snapshot directory, and the Python servers read their files
from it. A cache that does not hold the commit stops the service at startup
rather than serving other weights. A branch or tag name works only if the cache
has a `refs/` entry for it. A model id that is a local path has no revision:
vLLM and transformers ignore it, and pyannote refuses to start.

Not pinned, and why:

- `gliner`, `gliner-only`: `gliner.serve` has no revision option and loads
  the model id at the cache's `refs/main`. Its loader takes a revision, but the
  server does not pass one, and the v2.5 checkpoints take their encoder config
  from a second repo (`microsoft/deberta-v3-large`) that a revision of
  `NER_MODEL` would not cover.
- `vad`, `vad-only`: the weights ship inside the `silero-vad` package, so the
  image fixes them.
- `topics`, `topics-only`: there are no weights.
- `asr-only`: openai-whisper downloads its checkpoints from its own CDN and
  checks each against a SHA-256 built into the installed package, so the image
  fixes them; a Hub commit does not apply.

Some parts of a pinned service still load at their own repo's default
revision:

- a pyannote pipeline whose config names other repos. The 3.1 pipeline loads
  `pyannote/segmentation-3.0` and its embedding model that way; the default
  community-1 keeps its parts in its own repo, so one revision covers them.
- a draft model in `CHAT_SPECULATIVE_CONFIG`: give it a `"revision"` key in
  that JSON. An MTP drafter reads the chat model's own snapshot while
  `HF_HUB_OFFLINE=1`; with downloads on, it needs the key too.

Keep the comment off the variable's line in `.env`: after an empty value,
Compose reads a trailing `# …` as the value, and that value would be passed
as the revision.
