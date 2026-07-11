# False alarm — root cause + VAD-gating fix (measured)

**Date:** 2026-07-11
**Context:** false alarm (system marks non-speech as a speaker) is DER's dominant term on clean content (53% of the easy-subset error) and is **byte-identical across 3.1 and community-1** — a model-independent, segmentation-driven error, and the likely home of the client's *"music/sung intro scored as a speaker"* defect.

## Root cause: pyannote over-detects speech in non-speech (music/noise)

Per-file FA on the easy subset (community-1 hypotheses vs reference): the 13.2 s of raw FA is **concentrated in one file — `tucrg` (10.3 s, 78%)**; every other file has ≤0.9 s (boundary jitter). `tucrg` is 26 s with only **4.5 s of reference speech** (21+ s non-speech), yet pyannote marks **14.8 s** as speech — ~10 s hallucinated across 5 scattered chunks. pyannote's *speaker*-segmentation model is permissive on music/noise; that over-detection is the FA, and it is model-independent (3.1 and community-1 produce identical FA).

## Fix: gate the diarization by Silero VAD

Silero VAD (the stack's `/vad` service) rejects music/noise far better than pyannote's segmentation. Gating = intersect the diarization output with Silero's speech timeline (drop turns outside VAD speech). Measured on the easy subset (community-1 hypotheses, re-scored):

| Silero gate | DER | false alarm | missed |
|---|---|---|---|
| *ungated baseline* | 0.112 | 16.1 | 4.6 |
| threshold 0.5, pad 30 ms (defaults) | 0.107 | 8.5 | 10.7 |
| **threshold 0.4, pad 100 ms** | **0.098** | 10.5 | 6.2 |
| threshold 0.3, pad 150 ms | 0.098 | 11.6 | 5.1 |
| threshold 0.2, pad 300 ms | 0.099 | 12.5 | 4.6 |

- **Best: −12.5% DER** (0.112→0.098), **FA −35%** (16.1→10.5), for a small miss cost (+1.6 s).
- **Default Silero (0.5) over-cuts** real speech (miss 4.6→10.7) — a lower threshold (0.3–0.4) + padding (100–150 ms) is the sweet spot: it removes the gross music FA without trimming clean-speech edges.
- The win is **model-independent** — FA is identical for 3.1/community-1, so VAD-gating stacks on either.

## Recommendation

**Add Silero-VAD gating to the diarization path** (threshold ≈ 0.35, `speech_pad_ms` ≈ 100), reusing the existing `/vad` service. It is the complement to the model swap:

| Error | Dominant on | Lever |
|---|---|---|
| **speaker confusion** | hard, many-speaker content | model → **community-1** (−35% confusion) |
| **false alarm** | clean/music content | **VAD-gating** (−35% FA) — the *music-as-speaker* fix |

Together they attack both dominant error sources. Clustering threshold is not a lever (inert for community-1; see the model-comparison report).

**Where to implement** (a design choice for the follow-up):
- **Client-side (Nextext)** — it already calls `/vad` as a pre-Whisper guard and holds the diarization turns; intersecting turns with the VAD timeline in `nextext/core/diarization.py` is the least-invasive option and needs no backend change.
- **Backend (`diarize_server`)** — gate inside the service so every consumer benefits; heavier (the thin pyannote proxy gains a VAD dependency).

**Caveats:** measured on 8 clean VoxConverse files (FA-dominant), with `tucrg` as the music proxy. Validate the threshold/pad on **real client music/sung-intro clips** before locking it — that is the content this targets.
