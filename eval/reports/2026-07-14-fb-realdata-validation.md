# Fb validation on real labeled clips — 0.4, not 0.2, is the deploy value

**Date:** 2026-07-14
**Context:** the Fa/Fb benchmark sweep (same date) found `Fb=0.2` optimal on a
hard VoxConverse subset and flagged its over-split risk for validation on real
content. This is that validation.
**Data:** the 5 operator-labeled clips in `~/nextext/diarization_testdata`
(German/Arabic/English social-media and trailer content, 2–14 true
speakers/clip, ~8 min total), each labeled via the transcript flow
(`true_speaker` corrected in Nextext `transcript.csv`s produced at fb 0.4/0.6/0.8
— 7 labeled references in all, two clips labeled twice under different runs).
**Method:** the diarize pipeline (community-1, CPU, `fa=0.07`) was re-run
locally at `fb ∈ {0.8, 0.4, 0.2}` on the raw media; turns were assigned to each
reference's Whisper segments by maximum overlap (Nextext's client-side
alignment) and scored with `eval/transcript_metric.py` (collar-free speaker
accuracy + turn-boundary F1). The local fb=0.8/0.4 runs reproduce the
corresponding Nextext runs' speaker counts on every clip, so the local chain is
faithful.

## Results (micro-averaged over the 7 references)

| local fb | seg_acc | dur_acc | turn_P | turn_R | turn_F1 |
|---|---|---|---|---|---|
| 0.8 (stock) | 0.589 | **0.812** | 0.737 | 0.723 | **0.730** |
| **0.4** | **0.612** | 0.792 | **0.742** | 0.713 | 0.727 |
| 0.2 | **0.612** | 0.789 | 0.702 | 0.723 | 0.712 |

Per-clip speaker counts (true → found at 0.8 / 0.4 / 0.2):

| clip | true | fb 0.8 | fb 0.4 | fb 0.2 |
|---|---|---|---|---|
| Sheep Detectives trailer | 13–14 | 4 ✗ | **13 ✓** | 18 ✗ (over-split) |
| Razzia DMG | 6 | 6 ✓ | 6 ✓ | 6 ✓ |
| wennichduwäre | 3 | 2 ✗ | **3 ✓** | 4 (3 after alignment) |
| example_ar_1 | 3 | 2 ✗ | 2 ✗ | **3 ✓** |
| 50 Cent interview | 2 | 2 ✓ | 2 ✓ | 2 ✓ |

## Read

- **`Fb=0.4` is the real-content sweet spot.** It ties 0.2 on segment
  accuracy (0.612, vs 0.589 stock), keeps the best turn precision (0.742),
  and lands the speaker count exactly on 4 of 5 clips — including the
  hard 13-speaker trailer that stock collapses to 4.
- **`Fb=0.2` over-splits real content.** It finds 18 speakers where 13–14
  exist and its turn precision drops to 0.702 — the failure mode the
  benchmark sweep's `sqkup` file predicted. Its benchmark win does not
  transfer; the benchmark optimum sits below the real-content knee.
- **Stock 0.8's headline numbers are inflated by its failure.** Its dur_acc
  0.812 and F1 0.730 edge the others only because collapsing the trailer to
  4 speakers happens to score adequately under optimal relabelling while
  being qualitatively useless (10 speakers simply don't exist in its output).
  On every clip where the count is contested, 0.4 beats it.
- Residual: `example_ar_1` needs fb<0.4 to split its 3rd speaker, and the
  trailer's seg_acc stays ~0.46 at every fb (its errors are
  embedding/segmentation-limited, echoing the benchmark's `kbkon`). Fb
  cannot fix those; they are VAD-gating / segmentation territory.

## Deployed default

**`DIARIZE_FB=0.4`** (`DIARIZE_FA` unset → stock 0.07), superseding the
benchmark sweep's provisional 0.2. Benchmark cost is nil-to-small (DER 0.162
vs 0.151, count MAE 1.50 vs 1.33 on the hard subset); real-content attribution
accuracy and turn precision are what operators read.

## How to reproduce

Labeled data: `~/nextext/diarization_testdata` (`files/` media,
`transcripts/` labeled CSVs). Re-run + align + score:
see the sweep report for the pipeline seam; alignment is max-overlap of
pipeline turns onto each labeled CSV's segments, scored with

```bash
uv run --group eval python -m eval.transcript_metric <labels>/*.label.tsv
```

(macOS note: NFC-normalize filenames when matching clips — the filesystem
stores `ä` decomposed.)
