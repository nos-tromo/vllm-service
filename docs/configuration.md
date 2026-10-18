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
