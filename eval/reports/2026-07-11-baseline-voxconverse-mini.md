# Baseline — stock pyannote/speaker-diarization-3.1 (VoxConverse-mini)

**Date:** 2026-07-11
**Config:** `baseline-3.1` — stock `pyannote/speaker-diarization-3.1`, default hyperparameters, **no speaker bounds** (full auto-detect), CPU.
**Data:** the 8 shortest VoxConverse `dev` recordings (a fast smoke subset, **not** the full dev set): `hqyok tucrg tfvyr qrzjk qpylu szsyz gwtwd fxgvy` — 5.3 min total, 1–4 speakers/file (3 single-speaker, 5 multi-speaker).
**Scoring:** `pyannote.metrics` DER, 0.25 s collar, full-file UEM.

## Result

| config | DER | confusion | false alarm | missed | speaker-count MAE |
|---|---|---|---|---|---|
| baseline-3.1 | **0.098** | 5.9 s | 16.1 s | 4.6 s | 0.75 |

Sanity check: 3.1's published VoxConverse-dev DER is ~0.113, so 0.098 on this (shorter, easier) subset is the right ballpark — the harness scores correctly end-to-end against the real gated model.

## Read

- **False alarm dominates** (16.1 s of the 26.6 s total error) — pyannote marking speech/turns where the reference has none (or over-extending).
- **Speaker-count MAE 0.75** — auto-detect is off by ~0.75 speakers/file on this small set, the count instability a `min_speakers` floor is meant to stabilize.
- This subset is clean broadcast/YouTube speech — it does **not** contain the client's hard cases (background music, sung vocals, several similar male voices). Those need representative files added before the number reflects the reported failures.

## How to reproduce

```bash
uv sync --group eval --group eval-run           # torch + pyannote.audio (dev box)
# stage the subset (audio via HTTP range from the VoxConverse zip; RTTMs from GitHub; full-file UEM)
uv run --group eval python -m eval.prepare_data \
  --voxconverse-audio-dir "$PWD/data/voxmini/audio" \
  --voxconverse-rttm "$PWD/data/voxmini/refs.rttm" \
  --voxconverse-uem  "$PWD/data/voxmini/refs.uem" \
  --out data/database.yml
DIARIZE_DEVICE=cpu uv run --group eval --group eval-run python -m eval.sweep \
  --database data/database.yml --out-root data/runs \
  --configs data/configs-baseline.json --report data/report.md   # [{"label":"baseline-3.1","device":"cpu"}]
```

## Next (the tuning follow-up, separate spec/plan)

Measure against this 0.098 baseline: **(1)** swap to `pyannote/speaker-diarization-community-1`; **(2)** sweep `clustering_threshold`; **(3)** add a `min_speakers` floor. Add client-representative files (music / similar voices) so the number reflects the reported defects, then wire the winning config as server env knobs.
