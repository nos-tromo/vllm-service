# 3.1 vs community-1 — measured comparison

**Date:** 2026-07-11
**Configs:** stock `pyannote/speaker-diarization-3.1` vs `pyannote/speaker-diarization-community-1`, default hyperparameters, **no speaker bounds**, CPU. Both run on **pyannote.audio 4.0.7** (community-1 requires 4.x — see caveat).
**Scoring:** `pyannote.metrics` DER, 0.25 s collar, full-file UEM, VoxConverse `dev`.

## Two subsets, opposite verdicts

**Easy subset** — the 8 *shortest* dev files (5.3 min, mostly 1–3 speakers, clean broadcast):

| config | DER | confusion | false alarm | missed | count MAE |
|---|---|---|---|---|---|
| 3.1 | **0.103** | 7.3 | 16.1 | 4.6 | 0.50 |
| community-1 | 0.112 | 9.9 | 16.1 | 4.6 | 0.50 |

**Hard subset** — 6 dev files selected for difficulty (14.9 min, **3–12 speakers/file**, some overlap):

| config | DER | confusion | false alarm | missed | count MAE |
|---|---|---|---|---|---|
| **community-1** | **0.101** | **59.1** | 12.8 | 12.6 | **1.50** |
| 3.1 | 0.138 | 90.3 | 12.8 | 12.6 | 2.33 |

## Read

- **community-1 wins where it matters.** On the hard, many-speaker subset it cuts **DER −27%** (0.138→0.101), **speaker confusion −35%** (90.3→59.1 s), and **speaker-count error −36%** (MAE 2.33→1.50). Those are exactly the client-reported defects: *similar/male voices merged* (confusion) and *wrong speaker count*. On easy clean speech the two are ~equal (community-1 marginally worse), so a small/easy eval set hides the win — the hard set is the one that matters.
- **False alarm and miss are model-independent.** They are *byte-identical* across both models on *both* subsets (easy 16.1/4.6; hard 12.8/12.6). Swapping the model does nothing for them. This is the segmentation/VAD-driven error — the *"music/sung intro scored as a speaker"* complaint lives here — and it is a **separate lever** (VAD guard, segmentation `min_duration_off`/onset-offset), not a clustering/embedding one.
- **Stack consistency:** 3.1 scored 0.103 on pyannote 4.x vs 0.098 on 3.x (same easy subset) — within noise, so the cross-stack comparison is sound.

## Caveat — deploying community-1 is a pyannote 4.x migration

community-1 passes a `plda` kwarg that pyannote.audio **3.x's** `SpeakerDiarization` does not accept; it needs **4.x**. 4.x is a breaking change vs the production diarize server's `<4` pin (renamed `use_auth_token`→`token`, torchcodec decode). So adopting community-1 means migrating `vllm-service`'s diarize image to pyannote.audio 4.x, not just flipping `DIARIZE_MODEL`. The eval harness's `build_pipeline`/`run_diarization` are now version-tolerant (work on both majors); the `eval-run` group is pinned to 4.x so the harness can evaluate 4.x-era models.

## Clustering-threshold tuning is a dead end for community-1 (measured)

Both models **under-count** speakers on the hard set (reference 50; 3.1 finds 36, community-1 finds 41) — classic under-clustering. The obvious fix is to lower the clustering threshold (more clusters). It doesn't work for community-1:

| community-1 config (hard subset) | DER | confusion | count MAE |
|---|---|---|---|
| threshold 0.4 | 0.101 | 59.1 | 1.50 |
| threshold 0.5 | 0.101 | 59.1 | 1.50 |
| threshold 0.6 (default) | 0.101 | 59.1 | 1.50 |

Byte-identical. Verified it's not a harness bug: `instantiate` **does** apply (pyannote reports the new threshold), but community-1's output is unchanged even at extremes — one file gives the same speaker count at threshold 0.05, 0.5, and 0.95. **community-1's PLDA clustering ignores `clustering.threshold`;** its knobs are the PLDA calibration params `Fa`/`Fb` (defaults 0.07 / 0.8), which the harness's override seam does not yet expose. (The threshold seam still works for 3.1's agglomerative clustering.)

## Recommended next steps

1. **Adopt community-1** for the client's content — measured −27% DER on the hard set. Plan the diarize-server pyannote 4.x migration.
2. **To tune community-1's clustering, expose `Fa`/`Fb`** in the harness override (threshold is inert — above); or force speaker count with `min_speakers`. Count-MAE 1.50 says there's still room, but not via threshold.
3. **The biggest remaining lever is false alarm** (the model-independent 12–16 s): tighten the VAD guard / segmentation params. This is the largest error and the likely source of the *music-as-speaker* defect — and it helps **both** models.
4. Broaden the eval set with **client-representative clips** (actual music / sung intros / similar-male-voice recordings) before locking the production config.
