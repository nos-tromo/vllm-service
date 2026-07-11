# Diarization eval harness

Dev-only harness that reports a reproducible Diarization Error Rate (DER —
plus its components and speaker-count error) for the `/diarize` backend
(`src/diarize_server.py`) against public benchmarks, and can sweep a small
grid of configurations to compare them.

**Not shipped.** `eval/` and `data/` are excluded from every Docker image and
from `make bundle` (see `.dockerignore`/`.gitignore`). Nothing here runs at
serve time; `src/diarize_pipeline.py` is the only module in this design that
*is* shipped, because `diarize_server.py` imports it directly.

Design: `docs/superpowers/specs/2026-07-11-diarization-eval-harness-design.md`.
Plan: `docs/superpowers/plans/2026-07-11-diarization-eval-harness.md`.

## Install

```bash
# Unit tests only (torch-free — pyannote.metrics/database/core are pure Python):
uv sync --group eval

# Real diarization runs (adds pyannote.audio + torch; needs a GPU in practice):
uv sync --group eval --group eval-run
```

`uv run --group eval pytest eval/tests/ -v` runs the whole suite, including
`eval/tests/test_smoke_e2e.py`, without the `eval-run` group and without a
GPU, model download, or network access.

## One-time data prep

`eval/prepare_data.py` does **not** download anything — it only registers
corpora you've already put on disk as a `pyannote.database` `data/database.yml`.
Fetching the corpora themselves is a manual step, done once on a networked dev
box, from their canonical sources:

### VoxConverse (primary)

- **Audio:** `https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip`
  (dev split; there's a matching `voxconverse_test_wav.zip` for the test split).
  Unzip so you end up with one `<uri>.wav` per recording, e.g.
  `data/voxconverse/audio/abjxc.wav`.
- **RTTM:** clone/download `https://github.com/joonson/voxconverse` — its
  `dev/` folder holds one `<uri>.rttm` per recording (use the `master` branch;
  its README notes a v0.3 fix to some test RTTMs). This harness's
  `prepare_data.py` wants **one combined RTTM per split**, so concatenate:

  ```bash
  cat voxconverse/dev/*.rttm > data/voxconverse/dev.rttm
  ```

- **UEM:** VoxConverse does not ship UEM files — score over each recording's
  full duration. Generate one line per file (`<uri> 1 0.000 <duration>`) from
  the audio you already downloaded, e.g.:

  ```bash
  uv run --group eval python -c "
  import glob, os, soundfile as sf
  with open('data/voxconverse/dev.uem', 'w') as out:
      for wav in sorted(glob.glob('data/voxconverse/audio/*.wav')):
          uri = os.path.splitext(os.path.basename(wav))[0]
          duration = sf.info(wav).duration
          out.write(f'{uri} 1 0.000 {duration:.3f}\n')
  "
  ```

### AMI (secondary)

Use the pyannote fork of the official setup:
`https://github.com/pyannote/AMI-diarization-setup` — it vendors `lists/`
(per-split meeting-id lists), `only_words/rttms/<split>/<uri>.rttm`, and
`uems/<split>/<uri>.uem` (one file per recording each), plus scripts
(`pyannote/download_ami.sh`) to fetch the actual AMI audio (headset mix) into
`amicorpus/`. Follow its README to download the audio, then land it under
`data/ami/audio/<uri>.wav`, and concatenate its per-file references into the
single combined RTTM/UEM this harness's `prepare_data.py` expects:

```bash
cat AMI-diarization-setup/only_words/rttms/test/*.rttm > data/ami/test.rttm
cat AMI-diarization-setup/uems/test/*.uem > data/ami/test.uem
```

Use the `test` split (standard partition) for the recorded baseline; `dev` is
available the same way for iteration.

### Register both as a `pyannote.database` registry

Each protocol's trio of flags (`--<name>-audio-dir`/`--<name>-rttm`/`--<name>-uem`)
must be given together or not at all, and **registering both VoxConverse and
AMI requires passing both trios in the same invocation** — each run overwrites
`--out` rather than merging into an existing registry:

```bash
uv run --group eval python -m eval.prepare_data \
  --voxconverse-audio-dir data/voxconverse/audio \
  --voxconverse-rttm data/voxconverse/dev.rttm \
  --voxconverse-uem data/voxconverse/dev.uem \
  --ami-audio-dir data/ami/audio \
  --ami-rttm data/ami/test.rttm \
  --ami-uem data/ami/test.uem \
  --out data/database.yml
```

This writes `data/database.yml` plus, alongside each RTTM, a derived
`<rttm>.uris.lst` (`pyannote.database`'s custom-protocol loader requires an
explicit list of recording ids per subset — `write_database_yml` derives it
from the RTTM automatically, so you never hand-maintain it). The registered
protocols are `VoxConverse.SpeakerDiarization.Benchmark` and
`AMI.SpeakerDiarization.Benchmark`, each with a single `test` subset.

## Hugging Face gated model

The stock pipeline (`pyannote/speaker-diarization-3.1`) and its segmentation
dependency are gated on the Hugging Face Hub. Before any real run:

1. Accept the conditions for `pyannote/speaker-diarization-3.1` **and** its
   segmentation model dependency on huggingface.co (same account).
2. `export HF_TOKEN=<your token>` in the environment `eval/run.py`/`eval/sweep.py`
   run in.

`build_pipeline` (`src/diarize_pipeline.py`) fails loudly with an actionable
message if `Pipeline.from_pretrained` returns `None` — that's this step
missing, not a bug.

## Smoke a 2-3 file subset first

Before burning GPU time (or quota) on the full ~216-file VoxConverse dev set,
validate the whole chain — decode → pipeline → RTTM → score — on a tiny
subset. Trim the combined RTTM/UEM down to 2-3 uris and point
`prepare_data.py` at a separate registry so the full run stays untouched:

```bash
mkdir -p data/voxconverse-smoke/audio
cp data/voxconverse/audio/{abjxc,afjiv}.wav data/voxconverse-smoke/audio/
grep -E '^SPEAKER (abjxc|afjiv) ' data/voxconverse/dev.rttm > data/voxconverse-smoke/dev.rttm
grep -E '^(abjxc|afjiv) '        data/voxconverse/dev.uem  > data/voxconverse-smoke/dev.uem

uv run --group eval python -m eval.prepare_data \
  --voxconverse-audio-dir data/voxconverse-smoke/audio \
  --voxconverse-rttm data/voxconverse-smoke/dev.rttm \
  --voxconverse-uem data/voxconverse-smoke/dev.uem \
  --out data/database-smoke.yml

uv sync --group eval --group eval-run   # first time only

uv run --group eval python -m eval.run \
  --database data/database-smoke.yml \
  --protocol VoxConverse.SpeakerDiarization.Benchmark \
  --out-dir .eval-runs/smoke \
  --label smoke-baseline
```

Confirm `.eval-runs/smoke/*.rttm` was written with plausible speaker turns
before moving on. This is also the fastest way to confirm
`build_pipeline().parameters(instantiated=True)`/`instantiate` still matches
the installed pyannote.audio version's parameter tree — see
`src/diarize_pipeline.py`'s `_resolve_param_overrides`.

## Running a config

`eval/run.py` only **writes hypothesis RTTMs** — it does not score them:

```bash
uv run --group eval python -m eval.run \
  --database data/database.yml \
  --protocol VoxConverse.SpeakerDiarization.Benchmark \
  --out-dir .eval-runs/baseline-3.1 \
  --label baseline-3.1
```

Config knobs (`--model`, `--device`, `--clustering-threshold`,
`--min-speakers`, `--max-speakers`, `--num-speakers`) default to `None`,
which means "pipeline/env default" — see `eval/configs.py::DiarizeConfig`.

## Running and scoring a sweep

`eval/sweep.py` loads the protocol once, then for each config builds a
pipeline, runs it, and scores it against the protocol's references/UEM —
**this is the entry point for a scored report, even for a single config.**
`--configs` points at a JSON file: a list of objects, each becoming one
`DiarizeConfig` (`label` is required; every other field is optional and
defaults to `None`):

```json
[
  {"label": "baseline-3.1"},
  {"label": "community-1", "model_id": "pyannote/speaker-diarization-community-1"},
  {"label": "threshold-0.55", "clustering_threshold": 0.55},
  {"label": "floor2", "min_speakers": 2}
]
```

```bash
uv run --group eval python -m eval.sweep \
  --database data/database.yml \
  --protocol VoxConverse.SpeakerDiarization.Benchmark \
  --out-root .eval-runs \
  --configs eval/configs/sweep.json \
  --report eval/reports/2026-07-11-baseline.md
```

Each config's hypotheses land under `.eval-runs/<label>/`; progress and each
config's resulting DER are logged as the sweep runs (via the `logging`
module) so a config is never silently dropped from the final table. Prints
the ranked Markdown table (best/lowest DER first) and, with `--report`, also
writes it to a file.

`--collar` (default `0.25`, matching pyannote's model-card convention) and
`--protocol` (default `VoxConverse.SpeakerDiarization.Benchmark`) can target
`AMI.SpeakerDiarization.Benchmark` instead, or a stricter `--collar 0`.

## Notes

- `data/`, `.eval-runs/`, and `eval/**/__pycache__/` are gitignored — never
  commit corpus audio or scratch run output.
- `eval/score.py` has no CLI of its own; `eval/sweep.py::run_sweep` is the
  supported way to turn hypotheses into a scored report (see
  `eval/score.py::RunReport` for the field-by-field DER/component breakdown
  if you need it programmatically).
- The real `build_pipeline` path (an actual pyannote model, `eval-run` group,
  a GPU) is only exercised by an actual run — `eval/tests/test_smoke_e2e.py`
  validates the prepare → load → run → score → sweep glue with a fake
  pipeline so that check runs in CI-safe conditions (torch-free, no network).
