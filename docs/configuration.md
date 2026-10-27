# Configuration

Everything is env-driven: `.env` is the source of truth, `.env.example` ships
the annotated full set, and `docker/litellm.config.yaml` reads model names from
the environment at startup. This document covers the settings that need more
than a one-line comment. Per-shape defaults for the CPU-only containers live in
[deployment-shapes.md](deployment-shapes.md).

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
DIARIZE_DEVICE=cpu                               # default in diarize-only
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

## Updating the model catalog

`docker/litellm.config.yaml` is model-agnostic: all model names are read at
startup from the environment variables `TEXT_MODEL`, `EMBED_MODEL`,
`RERANK_MODEL`, and `WHISPER_MODEL`. To switch a model,
update the relevant variable in `.env` and restart the stack. No changes to
`docker/litellm.config.yaml` are required. Switching `TEXT_MODEL` to another
model family also needs the matching `CHAT_TOOL_CALL_PARSER`: a mismatch does
not fail loudly, the model's tool calls just go unrecognised (see
[Tool calling](#tool-calling)).

Clients must use the exact model ID set in `.env` as the `model` field in
their requests (e.g. `"model": "BAAI/bge-m3"`). The `/v1/models` endpoint
returns the currently active IDs.

`NER_MODEL`, `CLIP_MODEL`, `DIARIZE_MODEL`, and `VAD_MODEL` are the exceptions:
their servers have no OpenAI-shaped endpoints, so they are not in `model_list`
and do not appear in `/v1/models`. Switching them still works by updating the
variable in `.env` and restarting the matching service (`gliner`, `clip`,
`diarize`, `vad`).
