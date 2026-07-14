# Fa/Fb sweep — community-1 PLDA clustering, measured optimum (hard subset)

**Date:** 2026-07-14
**Context:** the model-comparison report (2026-07-11) showed community-1's
`clustering.threshold` is inert and named `Fa`/`Fb` as its only effective
speaker-granularity knobs (stock 0.07 / 0.8). This is the measured sweep that
work called for, run through the `--fa`/`--fb` seam this branch added.
**Config base:** `pyannote/speaker-diarization-community-1`, pyannote.audio
4.0.7, CPU, no speaker bounds.
**Data:** a re-staged 6-file VoxConverse `dev` hard subset (the 2026-07-11
subset was not preserved, so absolute numbers are not comparable across
reports): `wewoz xmfzh sqkup qjgpl tlprc kbkon` — 13.7 min, **6–12 reference
speakers/file**.
**Scoring:** `pyannote.metrics` DER, 0.25 s collar, full-file UEM
(`eval/sweep.py`).

## Results (three rounds, 13 configs, ranked)

| config | DER | confusion (s) | false alarm | missed | count MAE |
|---|---|---|---|---|---|
| **fb-0.2** | **0.151** | **91.9** | 14.1 | 11.2 | **1.33** |
| fb-0.5 | 0.155 | 94.8 | 14.1 | 11.2 | 2.00 |
| fb-0.3 | 0.156 | 95.4 | 14.1 | 11.2 | 1.33 |
| fb-0.6 | 0.157 | 96.3 | 14.1 | 11.2 | 2.17 |
| fb-0.4 | 0.162 | 99.9 | 14.1 | 11.2 | 1.50 |
| c1-default (fb 0.8, fa 0.07) | 0.162 | 100.5 | 14.1 | 11.2 | 2.33 |
| fb-0.15 | 0.167 | 104.2 | 14.1 | 11.2 | 2.17 |
| fb-1.0 | 0.169 | 105.4 | 14.1 | 11.2 | 2.50 |
| fa-0.14 fb-0.5 | 0.182 | 115.4 | 14.1 | 11.2 | 2.33 |
| fb-0.1 | 0.197 | 126.9 | 14.1 | 11.2 | 2.83 |
| fa-0.14 fb-0.3 | 0.198 | 128.1 | 14.1 | 11.2 | 2.67 |
| fa-0.03 fb-0.3 | 0.233 | 155.1 | 14.1 | 11.2 | 4.17 |
| fa-0.03 fb-0.5 | 0.407 | 290.3 | 14.1 | 11.2 | 6.17 |

## Read

- **Winner: `Fb = 0.2`, `Fa` stock (0.07)** — DER −7% (0.162→0.151),
  confusion −9% (100.5→91.9 s), speaker-count MAE −43% (2.33→1.33) vs the
  stock config. The knee is sharp: 0.15 and 0.1 over-split and are *worse*
  than stock.
- **Do not touch `Fa`.** It is far more sensitive than `Fb` and both
  directions lose: doubling it (0.14) costs ~+0.03 DER; halving it (0.03)
  is catastrophic (DER 0.233–0.407, count MAE up to 6.17). Stock 0.07 is
  correct; `DIARIZE_FA` stays an escape hatch, not a tuning target.
- **FA/miss are byte-identical across all 13 configs** (14.1 / 11.2 s) —
  Fa/Fb move *only* clustering (confusion + speaker count), confirming the
  knob's advertised scope and that VAD-gating (2026-07-11 report) remains an
  independent, stackable lever.
- **Where the win lives (per-file, stock → fb-0.2):** the highest-speaker
  files improve dramatically — `wewoz` (12 spk) 0.210→0.043 with the count
  exactly right (8→12), `tlprc` (8 spk) 0.158→0.095 — while one file
  over-splits (`sqkup` 9 spk: 7→11 hyp, 0.177→0.361). Net strongly positive,
  but the variance means content dominated by ~small speaker counts gains
  less; `kbkon` (6 spk) finds only 2 speakers at *every* Fb — its ceiling is
  segmentation/embedding, not clustering.

## Deployed default

**Superseded by real-content validation (same date):** `Fb=0.2` over-splits
real clips; the deployed default is **`DIARIZE_FB=0.4`** — see
`eval/reports/2026-07-14-fb-realdata-validation.md`. This still resolves the
previously inconsistent, unmeasured values (README 0.5 / `.env.example` 0.8 /
dev `.env` 0.6), and `Fa` stays stock per this sweep.

## How to reproduce

```bash
uv sync --group eval --group eval-run
# stage the hard subset: RTTMs from github.com/joonson/voxconverse (dev/),
# the 6 wavs from voxconverse_dev_wav.zip, full-file UEM; then:
uv run --group eval python -m eval.prepare_data \
  --voxconverse-audio-dir "$PWD/data/voxhard/audio" \
  --voxconverse-rttm "$PWD/data/voxhard/dev.rttm" \
  --voxconverse-uem  "$PWD/data/voxhard/dev.uem" \
  --out data/database-hard.yml
uv run --group eval --group eval-run python -m eval.sweep \
  --database data/database-hard.yml \
  --protocol VoxConverse.SpeakerDiarization.Benchmark \
  --out-root data/runs --configs <sweep>.json --report <report>.md
# sweep JSON entries: {"label": "fb-0.2", "model_id":
#   "pyannote/speaker-diarization-community-1", "device": "cpu", "fb": 0.2}
```

## Next

- Validate `Fb=0.2` on **client-representative clips** via the transcript
  metric (`eval/transcript_metric.py`) before treating it as final — the
  `sqkup` over-split says low-Fb behavior on ~2-speaker content needs a check
  (the easy-subset/few-speaker case was not re-measured here).
- The **VAD-gating lever** (−35% FA, 2026-07-11) is still unimplemented and
  is independent of this tuning; it remains the largest untouched win.
