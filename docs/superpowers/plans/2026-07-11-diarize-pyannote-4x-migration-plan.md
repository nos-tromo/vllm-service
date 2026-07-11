# Diarize server → pyannote.audio 4.x migration (to deploy community-1) — plan (scaffold)

**Date:** 2026-07-11
**Status:** Draft / scaffold — flagged decisions need sign-off before execution.
**Depends on:** vllm-service#55 (the diarize refactor + eval harness + version-tolerant `build_pipeline`). Branch it off `main` **after #55 merges** (this scaffold sits on `feature/diarize-pyannote-4x`, stacked on #55).

## Goal

Move the `/diarize` backend from pyannote.audio 3.x to **4.x** so it can run
`pyannote/speaker-diarization-community-1`, the measured **−27% DER / −35%
speaker-confusion** win on hard, many-speaker content (the client's merged-voices
defect). See `eval/reports/2026-07-11-model-comparison.md`.

## Why this is a migration, not an env flip

community-1 passes a `plda` kwarg 3.x's `SpeakerDiarization` rejects → it requires
pyannote.audio **4.x**, which is a breaking change vs the image's `<4` pin. The
#55 eval harness ran the whole pipeline on 4.x (4.0.7) + community-1, so the
breaking changes below are **empirically established**, not guessed.

## What pyannote.audio 4.x requires (established by the #55 eval harness)

1. **`pyannote.audio>=4,<5`** — pulls **`torchcodec`** (4.x moved audio decode to
   torchcodec) + newer `torch`/`torchaudio`. (On the eval dev box: torch 2.13 /
   torchaudio 2.11 / torchcodec 0.14 resolved cleanly.)
2. **Auth kwarg renamed** `use_auth_token=` → `token=`. **Already handled** —
   `src/diarize_pipeline.py::build_pipeline` (from #55) tries `token=` then falls
   back to `use_auth_token=`, so it works on both majors. The server calls
   `build_pipeline()`, so it inherits this for free.
3. **The pipeline returns a `DiarizeOutput` wrapper, not a bare `Annotation`.**
   `src/diarize_server.py:134,140` does `annotation = pipeline(...)` then
   `annotation.itertracks(...)` — this **breaks on 4.x** (`DiarizeOutput` has no
   `itertracks`). Must extract `.speaker_diarization` (the standard,
   overlap-allowing Annotation, matching 3.x semantics) first. `eval/run.py`
   already does `getattr(result, "speaker_diarization", result)`.
4. **`huggingface_hub<1.0` pin no longer needed** — 4.x uses `token=`, so the
   Dockerfile's `huggingface_hub>=0.13.0,<1.0` cap should be relaxed/removed.
5. **community-1 is gated** (cc-by-4.0): needs `HF_TOKEN` + accepted license, and
   **airgap bundling** of its weights (like the existing gated 3.1 procedure).
6. **`diarize_compat` revisit:** the torchaudio-symbol shims (`AudioMetaData`,
   `list_audio_backends`, `info`) are `hasattr`-guarded no-ops and likely
   irrelevant on 4.x/torchcodec — keep (harmless) or drop. The
   `weights_only`/`add_safe_globals` checkpoint allowlist may need **new entries**
   for community-1's checkpoint classes — verify a `weights_only=True` load.

## Decisions to make (sign-off needed)

- **D1 — default model.** Flip `DIARIZE_MODEL` default to `community-1`, or keep
  `3.1` as the image default and select community-1 per-deployment via `.env`?
  *Recommendation:* keep `3.1` as the built-in default (no gated download forced
  at build), ship the image **4.x-capable**, and set
  `DIARIZE_MODEL=pyannote/speaker-diarization-community-1` in the deployment
  `.env` once its weights are bundled. Lower blast radius; community-1 is opt-in
  per host.
- **D2 — diarization output flavor.** Use `.speaker_diarization` (standard,
  overlap-allowing — matches 3.x, Nextext aligns by overlap) vs
  `.exclusive_speaker_diarization` (one speaker at a time). *Recommendation:*
  standard, to preserve current downstream behavior.
- **D3 — 3.1-on-4.x parity.** The eval showed 3.1 scoring 0.103 on 4.x vs 0.098
  on 3.x (within noise). Accept 3.1-on-4.x as the fallback, or gate the migration
  on community-1 being the default? Tied to D1.

## Migration components

### `docker/Dockerfile.diarize.cuda` + `.cpu`
- `pyannote.audio>=3.3.2,<4` → `>=4,<5`; add `torchcodec`; ensure a
  4.x-compatible `torch`/`torchaudio` on the pinned CUDA base image.
- Relax/remove the `huggingface_hub<1.0` pin.
- Update the in-`RUN` **build smoke** for the 4.x import chain (it imports
  `pyannote.audio.pipelines`; add a torchcodec import check).
- Revisit the `diarize_compat` COPY + shim relevance (see item 6).

### `src/diarize_server.py`
- Extract the Annotation from the 4.x output before serializing:
  `result = pipeline(...); annotation = getattr(result, "speaker_diarization", result)`
  (cross-version: 3.x/fake return the Annotation directly). Everything else
  (the `itertracks` serialization, the request contract) is unchanged.

### `src/diarize_compat.py`
- Confirm the torchaudio shims are harmless/removable on 4.x; update the
  checkpoint-globals allowlist if community-1's `weights_only` load needs it.

### Airgap / bundle
- Bundle community-1 gated weights via the existing HF-token → download → bundle
  flow; update the download runbook + `.env.example` (`DIARIZE_MODEL` note).
- `make bundle`: re-check image size (torchcodec + 4.x) and that the community-1
  weights ship if it's the default (D1).

### Docs
- `CLAUDE.md`, `README.md`, `.env.example`: 4.x/torchcodec note, the gated
  community-1 download, and the `DIARIZE_MODEL` guidance.

## Task checklist (ordered; infra migration — verify at each step, not TDD-shaped)

1. **Dockerfile 4.x pins** (pyannote 4.x + torchcodec + torch/torchaudio + hub) +
   updated build smoke + `diarize_compat` revisit → **build the cpu image; the
   build smoke passes** (full pyannote 4.x import chain).
2. **`diarize_server` `DiarizeOutput` extraction** (D2) → **server smoke:** load
   3.1 on 4.x, `POST /diarize` a short multi-speaker clip, assert non-empty
   chronological segments (mirror the current response contract).
3. **community-1 access** (gated) + **D1** → load community-1 through the server,
   verify a real `/diarize` response.
4. **Airgap bundling** of community-1 weights + runbook.
5. **Acceptance = re-run the #55 eval harness against the deployed 4.x image**
   (community-1) on GPU → confirm the DER win holds end-to-end (the dev harness
   proved the pipeline; this proves the *served* path), and that 3.1-on-4.x is
   within noise of its 3.x number (D3).
6. **Docs** (above).

## Risks

- **torchcodec on the CUDA base** — C-extension + ffmpeg linkage + cu-version
  match; the highest-uncertainty step. Validate the image build early.
- **Version matrix** — torch/torchaudio/torchcodec/pyannote.audio mutual
  compatibility on the pinned CUDA base image.
- **Gated community-1 airgap bundling** — same class as the existing 3.1 gated
  flow, but a new (larger) model.
- **Behavior parity** — 3.1-on-4.x vs 3.x (small, measured); overlap semantics of
  the standard output (Nextext aligns by overlap, so compatible).

## Rollback

The migration is a Dockerfile + a small `diarize_server` change. Keep the last
3.x image tag as a fallback; revert the two files to roll back. `build_pipeline`'s
auth-tolerance and the eval harness are version-agnostic and stay either way.

## Out of scope

- Nextext-side changes — none needed (it aligns diarize turns by overlap; the
  standard output is the same shape). The VAD-gating (Nextext#91) is orthogonal
  and already deployable on the current stack.
- Fine-tuning community-1's `Fa`/`Fb` PLDA params (threshold is inert — see the
  model-comparison report); a separate optimization once the migration lands.
