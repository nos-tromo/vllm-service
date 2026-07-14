# Diarize backend VAD gating — design

**Date:** 2026-07-14
**Status:** approved

## Problem

False alarm — the diarizer marking music/noise as a speaker — is the dominant
model-independent error in `/diarize` output (16.1 s of 26.6 s total error on
the clean VoxConverse subset; the client-reported *"music/sung intro scored as
a speaker"* defect). It is unaffected by model choice or Fa/Fb tuning.
`eval/reports/2026-07-11-false-alarm-vad-gating.md` measured the fix: crop
diarization turns to a Silero VAD speech timeline — **−35% false alarm,
−12.5% DER** at threshold 0.4 / pad 100 ms.

Nextext already applies this gate client-side (`NEXTEXT_DIARIZE_VAD_GATE`,
default on, same tuned params). This change moves the gate into the diarize
service itself so **every** `/diarize` consumer benefits, not just Nextext.

## Decision summary

| Fork | Decision |
|---|---|
| Where | Backend, in `src/diarize_server.py` |
| VAD access | HTTP call to the stack's `vad` service (single Silero owner; no new model dep in the diarize images) |
| Default | **On in the full stack** (compose sets `DIARIZE_VAD_URL=http://vad:8000`); **off when `DIARIZE_VAD_URL` is unset** (code default) — diarize-only is unchanged unless an operator points it at a co-deployed `vad-only` |
| Failure mode | Fail-open: `/vad` error/timeout → return ungated turns, log a warning |
| Per-request override | None (YAGNI) |

## Behavior

On `POST /diarize`, after the pyannote pipeline returns turns:

1. If gating is disabled (no `DIARIZE_VAD_URL`, or `DIARIZE_VAD_GATE=false`),
   return turns as today — byte-identical behavior.
2. Otherwise POST the **original uploaded bytes** (multipart `file`, plus
   `threshold` and `speech_pad_ms` form fields) to `<DIARIZE_VAD_URL>/vad`.
   The vad service decodes via ffmpeg itself, so no re-encode is needed.
3. Crop every diarization turn to the returned speech timeline: intersect each
   turn with the speech intervals; a turn may split into several sub-turns;
   sub-turns of zero (or negative) length are dropped; the response `speakers`
   list is rebuilt from the surviving turns.
4. On any `/vad` failure (connection error, non-200, timeout, malformed body):
   log one warning and return the **ungated** turns (fail-open — a degraded
   gate must not take diarization down).

The `/diarize` response contract is unchanged (same JSON shape); only the set
of segments changes. `GET /health` is untouched and does not call `/vad`.

## Configuration

New optional env vars, `<SERVICE>_<KNOB>` pattern, parsed with the server's
existing warn-and-ignore-on-unparseable convention:

- `DIARIZE_VAD_URL` — root URL of the vad service. **Unset → gating off.**
  The full-stack `docker/compose.yaml` sets `http://vad:8000` on the `diarize`
  service, making gating the full-stack default.
- `DIARIZE_VAD_GATE` — kill switch (default `true` when the URL is set); set
  `false` to disable without unsetting the URL.
- `DIARIZE_VAD_THRESHOLD` — Silero threshold forwarded to `/vad`; default
  `0.4` (Silero's stock 0.5 over-cuts real speech — measured).
- `DIARIZE_VAD_PAD_MS` — `speech_pad_ms` forwarded to `/vad`; default `100`.
- `DIARIZE_VAD_TIMEOUT` — per-request timeout in seconds; default `30`.

## Code shape

- **`src/diarize_gate.py` (new, pure):** interval logic only —
  `crop_turns_to_speech(turns: list[tuple[float, float, str]], speech:
  list[tuple[float, float]]) -> list[tuple[float, float, str]]` (sorted-merge
  intersection). No torch, no HTTP, no env — unit-testable in the torch-free
  `eval` group.
- **`src/diarize_server.py`:** env parsing (same optional-float pattern as the
  Fa/Fb overrides), the `/vad` HTTP call (`requests`, already present
  transitively via `huggingface_hub`), fail-open wrapping, and rebuilding the
  response from the cropped turns.
- **`docker/compose.yaml`:** `DIARIZE_VAD_URL=http://vad:8000` (+ pass-through
  of the tuning vars) on the `diarize` service. No `depends_on`/healthcheck
  changes: the gate is a request-time call, and full-stack traffic cannot
  arrive before `vad` is healthy (the router's health chain starts after it).
- **Docs:** `README.md` (diarization section: gating behavior, env table,
  diarize-only pairing with `vad-only`), `CLAUDE.md`, `.env.example`.

## Testing

- Unit tests (`eval/tests/test_diarize_gate.py`, torch-free): crop semantics —
  turn fully inside speech (unchanged), fully outside (dropped), straddling
  (trimmed), spanning a gap (split in two), empty speech timeline (all turns
  dropped), empty turns (no-op), unsorted input.
- Server-level: env parsing (unset URL → gating skipped; `DIARIZE_VAD_GATE=
  false` honored; bad float warns and falls back) and fail-open (mocked
  `requests` raising → ungated turns returned) — mock-based, no live vad.
- Manual smoke (full stack): `curl /diarize` on a music-intro clip with and
  without `DIARIZE_VAD_GATE`, confirming the intro turns disappear when gated.

## Rollout

1. Ship this change; full-stack deployments gate by default.
2. Cross-repo follow-up (not in this change): set
   `NEXTEXT_DIARIZE_VAD_GATE=off` in Nextext deployments to stop
   double-gating. Double-gating is safe (the second crop is a near no-op with
   identical params), just wasteful — one extra `/vad` round-trip per job.
3. Validation follow-up (optional): re-run the transcript-metric scoring on
   the real labeled clips with the backend gate on, mirroring the Fb
   validation.

## Out of scope

- Per-request gate override fields on `/diarize`.
- In-process Silero in the diarize images (rejected: duplicate model
  ownership; the HTTP seam was chosen).
- Any change to the `vad` service or its `/vad` contract.
