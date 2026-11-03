# Audit log

The router is where data passes from the apps to the models, so it writes
the transfer record of the federation's data-protection audit log (§76
BDSG): one `ai_call` event for every model call that goes through it. The
event says which model and revision received the data, through which
deployment, how large and how long the call was, how it ended, and which of
the calling app's requests it belongs to. What was sent and for whom, the
calling app records in its own events; the router records no prompts,
inputs, outputs, documents or keys.

The store, its retention, access and the chain verifier live in obs-plane:
[obs-plane `docs/audit.md`](https://github.com/nos-tromo/obs-plane/blob/main/docs/audit.md).

The callback is `docker/router/router_audit.py`, mounted into the pinned
LiteLLM image with the federation's audit library (`audit_event.py`,
`audit_runtime.py`, vendored verbatim from `nos-tromo/.github`
`configs/audit/`; never edit those copies).

## What an event records

Every router worker process writes one hash-chained stream (`service`
`vllm-service`): `lifecycle` `start`, one `admin` `config.audit` event with
the operating mode (`call_types`, `openwebui_jwt`), a heartbeat a minute,
and `shutdown` when the worker stops. Each model call adds one event:

| Field | Value |
|---|---|
| `action` | `ai_call` |
| `operation` | `router.<call type>`: LiteLLM's call type without its async prefix, see [Routes](#routes) |
| `outcome` | `ok`, or `error` for a call that failed after it was sent |
| `actor` | `{"user": null, "via": "service"}`: every caller shares the master key, so the router cannot tell them apart; an Open WebUI user only with a valid signed token, see [Open WebUI's users](#open-webuis-users) |
| `request_id` | the caller's `X-Request-ID`, which the apps forward from the gateway; unset when absent or malformed |
| `recipient` | `model:<deployment>`, the compose service that received the data: `chat`, `embed`, `rerank`, `asr`, or the pass-through backend (`embed-sparse`, `gliner`, `clip`, `diarize`, `vad`, `topics`) |
| `ai.model_alias` | the model the caller requested; the request path for a pass-through route |
| `ai.model_id` | the deployment that served the call (`model_info.id` in `docker/litellm.config.yaml`) |
| `ai.revision` | the deployment's `<MODEL>_REVISION`, when set |
| `ai.input_tokens`, `ai.output_tokens` | as the response reports them; output tokens for text generation only |
| `ai.output_sha256` | SHA-256 of the text of a chat or text completion with a single choice, or of a transcription. A stream hashes its text deltas in order, which gives the same value. An app's `sha256_text(output)` matches when the app hashes the text as the model returned it |
| `criteria.route` | the request path |
| `criteria.status` | the HTTP status of the call |
| `criteria.duration_ms` | from the router's pre-call hook to the event |
| `criteria.stream` | whether the response was streamed |
| `criteria.attempts` | how many times the router sent the call to a deployment, retries included (not on pass-through routes) |
| `criteria.batch`, `criteria.documents` | the number of embedding inputs, of rerank documents |
| `criteria.error` | the error's class name, for `outcome: "error"`, or `client_disconnected` |
| `criteria.caller`, `criteria.forwarded_identity`, `criteria.chat_id` | how a forwarded user identity was handled, see below |

There is no `purpose`: the apps never send it to the router, they record it
with their own events. There is no `target`: the router does not know which
people the data is about.

When writes fail and recover, the next event written is `admin`
`audit.recovered` with `unwritten_events`: how many calls that had sent data
were left without their event in between.

## Routes

| Route | Audited as |
|---|---|
| `/v1/chat/completions`, `/chat/completions` | `router.completion` |
| `/v1/completions` | `router.text_completion` |
| `/v1/embeddings` | `router.embedding` |
| `/v1/rerank`, `/rerank`, `/v2/rerank` | `router.rerank` |
| `/v1/audio/transcriptions` | `router.transcription` |
| `/pooling`, `/tokenize`, `/gliner`, `/clip/embed_image`, `/clip/embed_text`, `/clip/dimension`, `/diarize`, `/vad`, `/topics/fit` | `router.pass_through_endpoint` |
| any other route LiteLLM serves that reaches a model (e.g. `/v1/responses`, `/v1/messages`, which LiteLLM translates for the chat model) | `router.<call type>` (`responses`, `anthropic_messages`, ...) |

Not audited:

- `/health/liveliness` (the healthcheck), `/metrics`, `/v1/models`: no data
  reaches a model.
- Requests without a valid key: LiteLLM refuses them before the pre-call
  hook, and auditing them would let anyone on the networks flood the trail.
- Calls that never reach a deployment (an unknown model, or refused by the
  audit gate below): nothing was transferred.
- The router's own calls to a deployment that no client request caused
  (LiteLLM health checks): they carry no client data.

## Fail closed

Each event is written in the last hook LiteLLM runs before it returns the
result. Where that hook can still turn the result into an error, a failed
write does so. Where it cannot, the failed write closes a gate: the pre-call
hooks then refuse every audited call, retries included, before anything
reaches a model, until a write succeeds again; the first event written after
that is `audit.recovered`.

| Path | Event written | When the write fails |
|---|---|---|
| Chat and text completions, embeddings (not streamed) | `async_post_call_success_hook`, before the response is sent | the client gets `503 audit log unavailable` and no result |
| Rerank, transcription | `async_post_call_success_deployment_hook`, inside the router's call to the deployment (these handlers run no later hook) | `503`, no result; the router's retries are refused before they are sent |
| Streamed completions | when the model's stream ends: text chunks go out as they arrive, the chunk with the finish reason and the usage chunk wait for the event | the client has the text streamed so far and then an error event instead of the final chunks and `[DONE]` |
| Pass-through routes | `async_post_call_response_headers_hook`, after the backend answered, before the response is sent | **not fail-closed for this response**: LiteLLM ignores what this hook raises, so the response goes out; the gate refuses the next audited call |
| A call that failed after it was sent (backend error, timeout, client gone) | `async_post_call_failure_hook`, or the stream's end | the client gets its error anyway; the gate closes |
| A call failed because its event could not be written | not written again | counted in the next `audit.recovered` |
| The chain itself | the worker startup hook | the worker does not start |

## Open WebUI's users

Open WebUI v0.12.4 can name its user on every call it makes to the router
(chat, RAG embeddings, speech-to-text). Its plain `X-OpenWebUI-User-*`
headers are not trusted: any holder of the master key can send them. With a
shared secret, Open WebUI sends one HS256-signed token instead, which the
router checks. To enable it, set in open-webui-service:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=true
FORWARD_USER_INFO_HEADER_JWT_SECRET=<a long random secret>
```

and the same secret here as `AUDIT_OPENWEBUI_JWT_SECRET`. A valid token
(issuer `open-webui`, unexpired, ±30 s clock skew) gives
`actor: {"user": "<email>", "via": "service"}` and
`criteria: {"caller": "open-webui", "forwarded_identity": "signed"}`, plus
Open WebUI's chat id as `criteria.chat_id`. The email is the account key
Open WebUI takes from the gateway's `X-Auth-Email`; the display name is not
used, because users can change it. Any other token, or a token while no
secret is set here, gives `forwarded_identity: "invalid"`; plain headers
give `"unsigned"`; both leave the user unnamed. The token is a bearer token:
whoever sees it on `inference-net` can reuse it until it expires
(`FORWARD_USER_INFO_HEADER_JWT_EXPIRES_SECONDS`, default 300).

## Volume

There is one event per call, not per item: a batch of 64 embedding inputs
is one event. Bulk ingestion still produces many, and every event is kept
for the audit store's retention. `AUDIT_ROUTER_CALL_TYPES` limits auditing
to the listed call types. A type left out gets no transfer record from the
router and is not held back by the gate either, so leave a type out only if
every app that calls it records those calls itself.

## Where the lines go

`AUDIT_OUTPUT=container` (set in `docker/compose.yaml`) writes to the
router container's main stdout under a lock every writer in the container
takes. The lock is on a router-private tmpfs (`/run/audit`): the router's
`/tmp` is the `media-tmp` spool it shares with `asr`, `diarize`, `vad` and
`topics`. stdout carries nothing else at runtime: uvicorn's access log goes
to stderr (`docker/router/uvicorn-log-config.json`), and `LITELLM_LOG` is
pinned to `WARNING`, since debug logging prints request bodies. LiteLLM
still prints its banner and the log-config notice to stdout once at
startup.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `AUDIT_ROUTER_CALL_TYPES` | empty: all | Comma-separated call types to audit (`completion`, `text_completion`, `embedding`, `rerank`, `transcription`, `pass_through_endpoint`, ...). An unknown name stops the router. |
| `AUDIT_OPENWEBUI_JWT_SECRET` | unset | The secret Open WebUI signs its user token with. Unset: no Open WebUI user is named. |
| `TEXT_MODEL_REVISION`, `EMBED_MODEL_REVISION`, `RERANK_MODEL_REVISION`, `WHISPER_MODEL_REVISION` | unset | Recorded as `ai.revision` (via `model_info.revision`). Recorded, not enforced: the backends still load whatever the cache resolves the model id to. |
| `AUDIT_OUTPUT` | `container` (compose) | `stdout` writes to the process's own stdout instead. |
| `AUDIT_SERVICE_VERSION` | `VLLM_SERVICE_VERSION` (compose) | The `service_version` of the events; a release tag loses its `v`. |

The first two go in `.env`; compose sets the others. The apps' purpose and
command-line settings (`AUDIT_REQUIRE_PURPOSE`, `AUDIT_CASE_REF`,
`AUDIT_REASON`, `AUDIT_CLI_USER`) do not apply: no purpose reaches the
router, and it has no command-line operations.

The callback is mounted from `docker/router/`, so a host needs that
directory next to `docker/litellm.config.yaml`; compose stops if it is
missing.

## Tests

```bash
uv run --group eval pytest eval/tests/test_router_audit.py
```

The tests stub LiteLLM and call the hooks in the order the pinned release
calls them; each ends by verifying the chain it wrote.
