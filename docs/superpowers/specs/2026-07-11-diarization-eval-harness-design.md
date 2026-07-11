# Diarization evaluation harness — design

**Date:** 2026-07-11
**Status:** Approved
**Repo / branch:** `vllm-service` / `feature/diarization-eval-harness`

## Problem

The `/diarize` backend (`src/diarize_server.py`) runs a **stock
`pyannote/speaker-diarization-3.1` pipeline with default hyperparameters**,
called with **no speaker bounds** (full auto-detect). Client-reported precision
is insufficient: acoustically-similar voices are merged into one speaker,
sung/music intros collapse into the narrator's label, and real turns are missed.

The Nextext diarization upgrade (`feature/diarization-always-on-wordlevel`, PR
Nextext#91) added word-level alignment, but that layer is **downstream** of
pyannote — it attributes Whisper words to whichever speaker pyannote already
decided and can split a Whisper segment at a pyannote boundary, but it cannot
create a distinction pyannote never made. The `diarization-always-on-wordlevel`
design explicitly deferred *"server-side pyannote parameter tuning (segmentation
thresholds, clustering)"* with the note **"revisit only if precision is still
lacking after word-level alignment."** It is. We are revisiting.

**The blocker to any tuning is that there is no metric.** Today, every proposed
change (swap the model, lower the clustering threshold, set a speaker floor) can
only be judged by eyeballing one transcript. This document specifies an
**evaluation harness** that produces a reproducible Diarization Error Rate (DER)
for the `/diarize` backend against public benchmarks, so the subsequent tuning
work is measured rather than guessed.

## Goal

A dev-only harness that, for a given diarization configuration, reports **DER
(+ its components) and speaker-count error** over public benchmark audio, and
can **sweep** a small grid of configurations to compare them. Concretely, it
must let us answer: *"Does swapping to `community-1` / lowering the clustering
threshold / adding a `min_speakers` floor reduce DER on diverse audio, and by
how much?"*

## Non-goals

- **Not** the tuning itself. Flipping the server's default model, wiring
  `DIARIZE_CLUSTERING_THRESHOLD` / `DIARIZE_MIN_SPEAKERS` env knobs, and
  promoting a new default config is a **follow-up** (informed by this harness's
  numbers) with its own plan. This spec only makes the pipeline *parameterizable*
  and *measurable*.
- **Not** Nextext-side work: the word-level alignment and the `Speaker N`
  label-ordering fix ("3 after 5") are a separate track. DER is
  permutation-invariant, so label *names* are correctly outside its scope.
- **Not** shipped: the harness and its data cache are excluded from the Docker
  images and `make bundle`.
- **Not** CI-gated (initially). A regression gate is a possible later addition,
  called out under Future work.
- **Not** fine-tuning pyannote's segmentation/embedding models.

## Approach (decided)

- **Measure the backend, in-process.** DER scores the pyannote pipeline's output
  directly against reference RTTM. The harness runs the pipeline **in-process**
  (imports it, no HTTP), so a parameter sweep needs no container rebuild or
  redeploy.
- **One pipeline factory, shared by server and eval.** Extract the pipeline
  construction currently inline in `diarize_server.py` into
  `src/diarize_pipeline.py::build_pipeline(...)`, parameterized by model,
  device, and the tunable hyperparameters. `diarize_server.py` is refactored to
  call it — **behavior-preserving**: with no overrides it is byte-for-byte the
  same stock 3.1, no-bounds pipeline as today. This guarantees *eval == what we
  ship* and is exactly the seam the follow-up tuning needs.
- **Public benchmarks, freely licensed.** VoxConverse (in-the-wild) primary, AMI
  (clean meetings) secondary — both CC-BY-4.0, ungated, downloadable on a
  networked dev box.
- **Standard, comparable metrics.** `pyannote.metrics` DER with the conventional
  0.25 s collar, overlap included, scored within each dataset's UEM — so our
  baseline is directly comparable to pyannote's published model-card numbers and
  a wrong-config baseline is caught immediately.

## Datasets

| Dataset | Role | Source (canonical) | License | Notes |
|---|---|---|---|---|
| **VoxConverse** (`dev`, 216 files) | Primary | RTTM: `github.com/joonson/voxconverse`; audio: Oxford VGG release | CC-BY-4.0 | Multi-speaker YouTube — talk shows, debates, news, **with music/noise/overlap**. Closest public proxy to real "audio/video in the wild"; stresses the exact failure modes reported. |
| **AMI** (`test`, standard partition) | Secondary | `github.com/pyannote/AMI-diarization-setup` (audio pointers + RTTM + UEM; use its standard partition and audio condition, not an ad-hoc pick) | CC-BY-4.0 | 4–5 speaker meetings; the standard clean multi-speaker reference point; lets us sanity-check against published DER. |

- **English only, and that is fine.** pyannote diarization is
  **language-agnostic** (acoustic segmentation + speaker embeddings, no ASR), so
  a config tuned here transfers to the German/other-language audio the toolkit
  ingests. A multilingual check (CALLHOME) is deferred — it is gated and
  CC-BY-NC (non-commercial).
- **Canonical sources, not the HF `diarizers-community` parquet.** The HF
  mirrors are chunked for *fine-tuning* pyannote segmentation; whole-recording
  DER wants the original per-file audio + RTTM + UEM. `prepare_data.py`
  normalizes canonical releases into the local cache.
- **UEM matters.** Both benchmarks define an Un-partitioned Evaluation Map (the
  scored regions). DER is computed within the UEM; ignoring it produces
  non-comparable numbers.
- **Loading mechanism: `pyannote.database` protocols.** Rather than re-implement
  RTTM/UEM plumbing, register the corpora via a `database.yml` and load them
  through the standard `pyannote.database` protocols
  (`AMI.SpeakerDiarization.*`, `VoxConverse.SpeakerDiarization.*`), which yield
  each file's audio, reference `Annotation`, and UEM uniformly and compose
  natively with `pyannote.metrics`. `prepare_data.py` reduces to fetching the raw
  corpora and emitting that `database.yml`.

## Metrics

Computed with `pyannote.metrics.diarization.DiarizationErrorRate`:

- **DER (primary):** collar **0.25 s**, overlap **included** (`skip_overlap=False`),
  within the dataset UEM. Matches pyannote model-card convention.
- **DER components:** **missed detection**, **false alarm**, **speaker
  confusion** — reported separately. These map onto the reported defects:
  *confusion* = merged voices, *false alarm* = music-as-speaker, *miss* = dropped
  turns.
- **Secondary DER:** collar **0 s** (stricter, boundary-sensitive) for reference.
- **Speaker-count error:** per file `|#hyp_speakers − #ref_speakers|`, reported
  as MAE and as a signed-bias mean (auto-detect's tendency to *under*-count is a
  named suspicion — the sign tells us).
- **Aggregation:** per-file rows plus per-dataset and overall summaries. DER is
  aggregated the correct way (Σ error durations / Σ reference durations, i.e.
  `pyannote.metrics`' accumulation), **not** a naïve mean of per-file DERs.

## Components

New/changed files (all under `vllm-service`):

```
src/diarize_pipeline.py        # NEW — shared pyannote pipeline factory
src/diarize_server.py          # CHANGED — call the factory (behavior-preserving)
eval/                          # NEW — dev-only harness (excluded from images/bundle)
  prepare_data.py              #   fetch + normalize VoxConverse/AMI → data cache
  run.py                       #   run a config over the eval set → hypothesis RTTM
  score.py                     #   ref vs hyp RTTM → DER + components + count error
  sweep.py                     #   iterate a config grid → comparison table
  configs.py                   #   dataclass describing one diarization config
  README.md                    #   usage + one-time data-prep instructions
  tests/                       #   pure-python unit tests (no GPU, no model download)
data/                          # NEW — local benchmark cache (gitignored, dev-only)
```

### `src/diarize_pipeline.py` — the shared factory

```python
def build_pipeline(
    *,
    model_id: str | None = None,       # default: DIARIZE_MODEL env or 3.1
    device: str | None = None,         # default: DIARIZE_DEVICE env or cuda
    clustering_threshold: float | None = None,   # override; None = pretrained default
    segmentation_min_duration_off: float | None = None,  # override; None = default
) -> Pipeline: ...
```

- Loads `Pipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN or None)`,
  keeps the existing `None`-guard (gated-repo misconfig crash-loops loudly), and
  `.to(device)`.
- **Override semantics (subtle — must be correct):** pyannote's
  `pipeline.instantiate(params)` **replaces the full parameter set**. To override
  only the clustering threshold we read the pipeline's instantiated defaults
  (`pipeline.parameters(instantiated=True)`), merge the override, and
  re-instantiate. **When no override is given, `instantiate` is not called at
  all** — preserving today's exact behavior.
- **Model-aware hyperparameters:** 3.1 and `community-1` expose different
  hyperparameter trees (`community-1`'s pipeline differs). The factory maps the
  logical knobs (clustering threshold, min-duration-off) to the loaded pipeline's
  actual parameter path and raises a clear error if a requested knob is absent
  for the loaded model, rather than silently no-op'ing.

### `src/diarize_server.py` — refactor (behavior-preserving)

Replace the inline `Pipeline.from_pretrained(...)` + `.to(...)` block (currently
around lines 59–80) with a single `build_pipeline()` call using env defaults and
no overrides. The `/diarize` request path, response shape, `num/min/max_speakers`
handling, and the serialization loop are unchanged.

### `eval/prepare_data.py`

One-time, run on a networked dev box. Downloads VoxConverse (`dev`) and AMI
(`test`) canonical releases, normalizes the audio to 16 kHz mono WAV, and emits a
`data/database.yml` registering both as `pyannote.database` protocols (with their
reference RTTM + UEM). Idempotent (skips already-materialized files).

### `eval/run.py`

Given a `DiarizeConfig` and a protocol name, builds the pipeline once via the
factory and iterates the protocol's files (via `pyannote.database`), running the
pipeline over each (respecting `num/min/max_speakers` from the config) and
writing per-file **hypothesis RTTM** into a run directory. Reuses
`diarize_server.py`'s ffmpeg decode (factored into a shared helper) so eval
decoding matches production.

### `eval/score.py`

Takes the protocol's reference `Annotation` + UEM (from `pyannote.database`) and
the run's hypothesis RTTM, computes the metrics above with `pyannote.metrics`,
and writes a report (`report.md` + `report.csv`): per-file rows and
per-dataset/overall summaries.

### `eval/sweep.py`

Runs `run.py` + `score.py` for each config in a small grid (e.g.
`model ∈ {3.1, community-1} × clustering_threshold ∈ {default, …} × min_speakers ∈ {none, 2}`)
and emits a single comparison table sorted by DER. Explicitly `log`s the grid so
no config is silently dropped.

### `eval/configs.py`

A `DiarizeConfig` dataclass: `model_id`, `device`, `clustering_threshold`,
`segmentation_min_duration_off`, `num_speakers`, `min_speakers`, `max_speakers`,
plus a human label. Serializable into the report so every number is traceable to
its exact config.

## Data flow

```
prepare_data.py ─► data/{audio 16k, rttm, uem} + data/database.yml   (one-time)
                                    │  (pyannote.database protocols)
DiarizeConfig ─► run.py ─(build_pipeline)─► hyp RTTM per file ─► score.py ─► report.md/csv
                                    │                                   ▲
                                    └───────── sweep.py orchestrates ───┘ (grid → comparison table)
```

## The tuning loop this unlocks (the follow-up, step b)

1. **Baseline:** stock `speaker-diarization-3.1`, no bounds → the number we start
   from (sanity-checked against the published ~model-card DER).
2. **Model swap:** `community-1` (gated; needs `HF_TOKEN` + accepted license) →
   re-score. Expected the single biggest drop.
3. **Sweep** `clustering_threshold × min_speakers` → lock the best config; the
   follow-up plan wires those as server env knobs and promotes the default.

## Error handling & edge cases

| Situation | Behavior |
|---|---|
| A recording fails to decode/diarize | Log, record an empty hypothesis for that file, continue the run (one bad file must not abort a sweep). Count skipped files in the report. |
| Gated model without `HF_TOKEN`/license | `build_pipeline` fails loudly (existing `None`-guard) with the actionable message. |
| Requested knob absent for the loaded model | Raise a clear error naming the knob and model (no silent no-op). |
| Missing UEM for a dataset | Score over full-file support and **mark the report** as "no UEM (numbers not directly comparable)". |
| Partial data cache | `prepare_data.py` is idempotent; `run.py` processes whatever files the registered protocol resolves and reports the file count. |

## Testing

Harness logic is unit-tested **without GPU or model downloads**:

- `score.py`: synthetic ref/hyp RTTM with a hand-computed DER (incl. a
  known-confusion and a known-miss case) → asserts DER + component split; a
  speaker-count-error case.
- `prepare_data.py`: RTTM/UEM normalization on a tiny fixture (parsing,
  16 kHz-mono audio + emitted `database.yml` shape).
- `run.py`/`sweep.py`: wiring tested with a **mock pipeline** (a fake returning a
  canned annotation) so no torch/pyannote is imported in the unit tests.
- `build_pipeline`: an override-merge unit test using a stub that records the
  params passed to `instantiate` — asserts *no* `instantiate` call when no
  override is given (the behavior-preserving guarantee), and a correct merged
  param set when one is.
- Gate: `pre-commit run --all-files` (ruff + pyrefly) green on the new modules.
  (vllm-service runs ruff + pyrefly via pre-commit; the pure-python harness tests
  run under `pytest` locally without the ML extras.)

## Packaging / airgap

- `eval/` and `data/` are **dev-only**: not `COPY`'d by `docker/Dockerfile.diarize.*`
  and not included in `make bundle`. `data/` is gitignored.
- `src/diarize_pipeline.py` **is** shipped — it is imported by
  `diarize_server.py`.
- Downloads happen only in `prepare_data.py` on a networked dev box; nothing in
  the harness fetches at serve time. The airgap invariant is unchanged.

## Future work (out of scope here)

- Wire `DIARIZE_CLUSTERING_THRESHOLD` / `DIARIZE_MIN_SPEAKERS` server env knobs
  and promote a tuned default (the step-b plan).
- Optional CI regression gate (fail if DER rises > X on a small fixed subset).
- Multilingual benchmark (CALLHOME) if language-specific validation is wanted.
- Music/non-speech pre-filtering ahead of diarization (feeding VAD-confirmed
  speech only), if the sung-intro class-collision persists after model/threshold
  tuning.
