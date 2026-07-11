# Diarization Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dev-only harness that reports reproducible DER (+ components + speaker-count error) for the `/diarize` backend against public benchmarks, plus the behavior-preserving pipeline factory it and the server share.

**Architecture:** Factor the pyannote pipeline construction and the ffmpeg decode out of `src/diarize_server.py` into reusable modules (`src/diarize_pipeline.py`, `src/diarize_audio.py`); the server keeps identical behavior. A new dev-only `eval/` package loads public corpora via `pyannote.database`, runs a configurable pipeline over them, and scores hypotheses with `pyannote.metrics`. Heavy `pyannote.audio` (torch) is isolated to actual diarization runs; every unit test is torch-free.

**Tech Stack:** Python 3.11+, pyannote.audio (>=3.3.2,<4), pyannote.metrics, pyannote.database, ffmpeg (subprocess), pytest, ruff, pyrefly, uv dependency-groups.

## Global Constraints

- **Python** `>=3.11`; **ruff** `line-length = 120`, lints `E,F,W,I,B,D,ANN,UP,RUF` (google docstring convention) — every new/modified function needs a **google-style docstring** and **type annotations**.
- **pyrefly** `preset = "strict"`, `ignore-missing-imports = ["*"]` — pyannote/torch resolve to `Any`; first-party code must still type-check.
- **Behavior-preserving server:** the refactored `src/diarize_server.py` must be functionally identical — same `/diarize` request/response, same `num/min/max_speakers` handling, same stock `pyannote/speaker-diarization-3.1` + no-bounds defaults, same gated-repo `None`-guard message.
- **Dev-only, not shipped:** `eval/` and `data/` are excluded from the Docker images and `make bundle`; nothing under `eval/` is imported by a shipped server module. **Airgap:** only `eval/prepare_data.py` fetches data, and only on a networked dev box — never at serve time.
- **Torch isolation:** unit tests import only the torch-free `eval` dependency group. `pyannote.audio` (the `eval-run` group) is required only for real diarization runs (Task 8).
- **Commits:** conventional-commit subjects; every commit message ends with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (omitted from the per-step `-m` examples below for brevity — append it to each).
- **Run commands from the repo root** of this worktree: `/Users/himarc/dev/nos-tromo/infra/.worktrees/vllm-svc-diarization-eval-harness`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | Declare `eval` (torch-free) + `eval-run` (pyannote.audio) dependency groups. |
| `.gitignore` (modify) | Ignore `data/` and eval caches. |
| `.dockerignore` (modify) | Belt-and-suspenders exclusion of `eval/` + `data/` from image build context. |
| `eval/configs.py` (create) | `DiarizeConfig` dataclass — one diarization configuration, serializable. |
| `eval/score.py` (create) | `pyannote.metrics` scoring → per-file + aggregate DER/components/count-error. |
| `eval/prepare_data.py` (create) | Fetch corpora + emit `data/database.yml` registering `pyannote.database` protocols. |
| `src/diarize_audio.py` (create) | `decode_audio(bytes) -> np.ndarray` — ffmpeg decode, moved verbatim from the server. |
| `src/diarize_pipeline.py` (create) | `build_pipeline(...)` factory + pure `_resolve_param_overrides(...)`. |
| `src/diarize_server.py` (modify) | Consume the two new `src/` modules; drop the inline construction. |
| `eval/run.py` (create) | Run one config over a protocol → per-file hypothesis RTTM. |
| `eval/sweep.py` (create) | Iterate a config grid → sorted comparison table. |
| `eval/tests/` (create) | Torch-free unit tests for the above. |
| `eval/README.md` (create) | Usage + one-time data-prep + baseline runbook. |

---

## Task 1: Dependency groups + ignore scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `.dockerignore`
- Create: `eval/__init__.py`, `eval/tests/__init__.py`

**Interfaces:**
- Produces: the `eval` (torch-free) and `eval-run` (heavy) uv dependency groups; an importable `eval` package. Downstream tasks run `uv run --group eval pytest ...`.

- [ ] **Step 1: Add the dependency groups.** Edit `pyproject.toml` `[dependency-groups]` (currently `dev = ["pre-commit", "pyrefly==1.1.1"]`) to add:

```toml
[dependency-groups]
dev = [
    "pre-commit",
    "pyrefly==1.1.1",
]
# Dev-only eval harness. `eval` is torch-free (pyannote.metrics/database/core are
# pure-python) so every unit test installs without torch. `eval-run` adds
# pyannote.audio (torch) and is needed only for real diarization runs (Task 8).
eval = [
    "pyannote.metrics>=3.2",
    "pyannote.database>=5.0",
    "pytest>=8",
    "soundfile>=0.12",
    "pyyaml>=6",
]
eval-run = [
    "pyannote.audio>=3.3.2,<4",
]
```

- [ ] **Step 2: Ignore data + caches.** Append to `.gitignore`:

```gitignore

# Dev-only diarization eval harness (see eval/README.md)
/data/
eval/**/__pycache__/
.eval-runs/
```

- [ ] **Step 3: Exclude from image build context.** Append to `.dockerignore`:

```dockerignore

# Dev-only eval harness — never part of any image
eval/
data/
```

- [ ] **Step 4: Create the package markers.**

`eval/__init__.py`:
```python
"""Dev-only diarization evaluation harness (not shipped in any image)."""
```

`eval/tests/__init__.py`:
```python
"""Unit tests for the diarization eval harness."""
```

- [ ] **Step 5: Install and verify the torch-free group.**

Run: `uv sync --group eval`
Then: `uv run --group eval python -c "import pyannote.metrics, pyannote.database, soundfile, yaml; print('eval group OK')"`
Expected: prints `eval group OK`, and no `torch` in the resolved env (`uv run --group eval python -c "import importlib.util,sys; print('torch' in sys.modules or importlib.util.find_spec('torch') is not None)"` prints `False`).

- [ ] **Step 6: Commit.**

```bash
git add pyproject.toml uv.lock .gitignore .dockerignore eval/__init__.py eval/tests/__init__.py
git commit -m "build(eval): add torch-free eval + eval-run dependency groups and ignores"
```

---

## Task 2: `DiarizeConfig` dataclass

**Files:**
- Create: `eval/configs.py`
- Test: `eval/tests/test_configs.py`

**Interfaces:**
- Produces: `DiarizeConfig` (frozen dataclass) with fields `model_id: str | None`, `device: str | None`, `clustering_threshold: float | None`, `segmentation_min_duration_off: float | None`, `num_speakers: int | None`, `min_speakers: int | None`, `max_speakers: int | None`, `label: str`; method `as_dict() -> dict[str, Any]`; property `pipeline_kwargs -> dict[str, int]` (only the non-None `num/min/max_speakers`, for passing to the pipeline call). Consumed by `run.py`, `sweep.py`, `score.py` reporting.

- [ ] **Step 1: Write the failing test.** `eval/tests/test_configs.py`:

```python
"""Tests for the DiarizeConfig value object."""

from eval.configs import DiarizeConfig


def test_defaults_are_all_none_and_label_required() -> None:
    """A bare config carries only its label; every knob defaults to None."""
    cfg = DiarizeConfig(label="baseline")
    assert cfg.label == "baseline"
    assert cfg.model_id is None
    assert cfg.clustering_threshold is None
    assert cfg.pipeline_kwargs == {}


def test_pipeline_kwargs_include_only_set_speaker_bounds() -> None:
    """pipeline_kwargs surfaces only the speaker bounds that were set."""
    cfg = DiarizeConfig(label="floor2", min_speakers=2)
    assert cfg.pipeline_kwargs == {"min_speakers": 2}


def test_as_dict_round_trips_all_fields() -> None:
    """as_dict is flat and includes every field for report traceability."""
    cfg = DiarizeConfig(label="c1", model_id="pyannote/speaker-diarization-community-1", clustering_threshold=0.6)
    d = cfg.as_dict()
    assert d["label"] == "c1"
    assert d["model_id"] == "pyannote/speaker-diarization-community-1"
    assert d["clustering_threshold"] == 0.6
    assert d["num_speakers"] is None
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run --group eval pytest eval/tests/test_configs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.configs'`.

- [ ] **Step 3: Write minimal implementation.** `eval/configs.py`:

```python
"""Value object describing one diarization configuration under evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DiarizeConfig:
    """One diarization configuration: model, device, hyperparameters, bounds.

    Attributes:
        label: Human-readable name, used as the report row key.
        model_id: pyannote pipeline id; None → the server's env/3.1 default.
        device: Torch device string; None → the server's env/cuda default.
        clustering_threshold: Clustering-threshold override; None → pretrained default.
        segmentation_min_duration_off: Segmentation override; None → pretrained default.
        num_speakers: Exact speaker count, if forced.
        min_speakers: Lower bound on the speaker count.
        max_speakers: Upper bound on the speaker count.
    """

    label: str
    model_id: str | None = None
    device: str | None = None
    clustering_threshold: float | None = None
    segmentation_min_duration_off: float | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    @property
    def pipeline_kwargs(self) -> dict[str, int]:
        """Return only the set speaker-bound kwargs for the pipeline call.

        Returns:
            Mapping of ``num_speakers``/``min_speakers``/``max_speakers`` to
            their values, omitting any that are None.
        """
        pairs = (
            ("num_speakers", self.num_speakers),
            ("min_speakers", self.min_speakers),
            ("max_speakers", self.max_speakers),
        )
        return {name: value for name, value in pairs if value is not None}

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dict of every field for report serialization.

        Returns:
            All dataclass fields as a plain dictionary.
        """
        return asdict(self)
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `uv run --group eval pytest eval/tests/test_configs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit.**

```bash
git add eval/configs.py eval/tests/test_configs.py
git commit -m "feat(eval): add DiarizeConfig value object"
```

---

## Task 3: `score.py` — DER + components + speaker-count error

**Files:**
- Create: `eval/score.py`
- Test: `eval/tests/test_score.py`

**Interfaces:**
- Consumes: `pyannote.core.Annotation`/`Segment`/`Timeline`, `pyannote.metrics.diarization.DiarizationErrorRate`.
- Produces:
  - `FileScore` dataclass: `uri: str`, `der: float`, `confusion: float`, `false_alarm: float`, `missed_detection: float`, `total: float`, `ref_speakers: int`, `hyp_speakers: int`, `speaker_count_error: int`.
  - `RunReport` dataclass: `files: list[FileScore]`, `overall_der: float`, `overall_confusion: float`, `overall_false_alarm: float`, `overall_missed_detection: float`, `speaker_count_mae: float`, `speaker_count_bias: float`; method `to_markdown() -> str` and `to_csv_rows() -> list[dict[str, Any]]`.
  - `score_run(items: list[tuple[str, Annotation, Annotation, Timeline | None]], *, collar: float = 0.25, skip_overlap: bool = False) -> RunReport`.

> **Note for the implementer:** confirm the exact `pyannote.metrics` detailed-component key strings at Step 2 — this plan uses `"confusion"`, `"missed detection"`, `"false alarm"`, `"total"`, which are the documented keys. If the installed version differs, adjust the `_COMPONENT_KEYS` mapping only. The tests use `collar=0` for exact hand-computed arithmetic.

- [ ] **Step 1: Write the failing test.** `eval/tests/test_score.py`:

```python
"""Tests for DER scoring (hand-computed cases, collar=0)."""

from pyannote.core import Annotation, Segment

from eval.score import score_run


def _ann(spans: list[tuple[float, float, str]]) -> Annotation:
    """Build an Annotation from (start, end, speaker) triples."""
    ann = Annotation()
    for start, end, spk in spans:
        ann[Segment(start, end)] = spk
    return ann


def test_missed_detection_half() -> None:
    """Ref speaks 0-10, hyp only 0-5 → 5s miss over 10s total → DER 0.5."""
    ref = _ann([(0.0, 10.0, "A")])
    hyp = _ann([(0.0, 5.0, "A")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    assert report.files[0].missed_detection == 5.0
    assert report.overall_der == 0.5


def test_confusion_is_permutation_invariant() -> None:
    """Two ref speakers, one hyp speaker: best mapping leaves 5s confusion → DER 0.5."""
    ref = _ann([(0.0, 5.0, "A"), (5.0, 10.0, "B")])
    hyp = _ann([(0.0, 10.0, "X")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    assert report.overall_der == 0.5
    assert report.files[0].confusion == 5.0


def test_speaker_count_error_and_bias() -> None:
    """Ref has 2 speakers, hyp has 1 → count error 1, bias -1 (under-count)."""
    ref = _ann([(0.0, 5.0, "A"), (5.0, 10.0, "B")])
    hyp = _ann([(0.0, 10.0, "X")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    assert report.files[0].ref_speakers == 2
    assert report.files[0].hyp_speakers == 1
    assert report.files[0].speaker_count_error == 1
    assert report.speaker_count_mae == 1.0
    assert report.speaker_count_bias == -1.0


def test_markdown_has_overall_row() -> None:
    """The rendered report includes an OVERALL summary line."""
    ref = _ann([(0.0, 10.0, "A")])
    hyp = _ann([(0.0, 10.0, "A")])
    report = score_run([("f1", ref, hyp, None)], collar=0.0)
    md = report.to_markdown()
    assert "OVERALL" in md
    assert "f1" in md
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run --group eval pytest eval/tests/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.score'`. (While here, in a Python REPL confirm the component keys: `from pyannote.metrics.diarization import DiarizationErrorRate` then inspect a `metric(ref, hyp, detailed=True)` dict.)

- [ ] **Step 3: Write minimal implementation.** `eval/score.py`:

```python
"""Score diarization hypotheses against references with pyannote.metrics.

DER is accumulated across files by a single metric instance (Σ error / Σ
reference), never by averaging per-file rates. Speaker-count error is computed
directly from the annotations' distinct labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyannote.core import Annotation, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

# Detailed-component keys emitted by DiarizationErrorRate(detailed=True).
_CONFUSION = "confusion"
_MISS = "missed detection"
_FALSE_ALARM = "false alarm"
_TOTAL = "total"


@dataclass(frozen=True)
class FileScore:
    """Per-file diarization scores (seconds for the duration components)."""

    uri: str
    der: float
    confusion: float
    false_alarm: float
    missed_detection: float
    total: float
    ref_speakers: int
    hyp_speakers: int
    speaker_count_error: int


@dataclass(frozen=True)
class RunReport:
    """Aggregate scores for a run over a set of files."""

    files: list[FileScore]
    overall_der: float
    overall_confusion: float
    overall_false_alarm: float
    overall_missed_detection: float
    speaker_count_mae: float
    speaker_count_bias: float

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Return one dict per file plus an OVERALL row for CSV export.

        Returns:
            A list of flat dictionaries suitable for ``csv.DictWriter``.
        """
        rows: list[dict[str, Any]] = [
            {
                "uri": f.uri,
                "der": round(f.der, 4),
                "confusion": round(f.confusion, 2),
                "false_alarm": round(f.false_alarm, 2),
                "missed_detection": round(f.missed_detection, 2),
                "ref_speakers": f.ref_speakers,
                "hyp_speakers": f.hyp_speakers,
                "speaker_count_error": f.speaker_count_error,
            }
            for f in self.files
        ]
        rows.append(
            {
                "uri": "OVERALL",
                "der": round(self.overall_der, 4),
                "confusion": round(self.overall_confusion, 2),
                "false_alarm": round(self.overall_false_alarm, 2),
                "missed_detection": round(self.overall_missed_detection, 2),
                "ref_speakers": "",
                "hyp_speakers": "",
                "speaker_count_error": round(self.speaker_count_mae, 3),
            }
        )
        return rows

    def to_markdown(self) -> str:
        """Render the report as a Markdown table.

        Returns:
            A Markdown string with a per-file row per file and an OVERALL row.
        """
        header = (
            "| uri | DER | conf | FA | miss | ref# | hyp# | count_err |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        body = "".join(
            f"| {f.uri} | {f.der:.3f} | {f.confusion:.1f} | {f.false_alarm:.1f} | "
            f"{f.missed_detection:.1f} | {f.ref_speakers} | {f.hyp_speakers} | {f.speaker_count_error} |\n"
            for f in self.files
        )
        overall = (
            f"| **OVERALL** | **{self.overall_der:.3f}** | {self.overall_confusion:.1f} | "
            f"{self.overall_false_alarm:.1f} | {self.overall_missed_detection:.1f} | | | "
            f"MAE {self.speaker_count_mae:.2f} / bias {self.speaker_count_bias:+.2f} |\n"
        )
        return header + body + overall


def _rate(numerator: float, denominator: float) -> float:
    """Safe ratio; 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def score_run(
    items: list[tuple[str, Annotation, Annotation, Timeline | None]],
    *,
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> RunReport:
    """Score reference/hypothesis pairs and aggregate DER + speaker-count error.

    Args:
        items: ``(uri, reference, hypothesis, uem)`` tuples; ``uem`` may be None.
        collar: Forgiveness collar in seconds around reference boundaries.
        skip_overlap: When True, exclude overlapped-speech regions from scoring.

    Returns:
        A RunReport with per-file scores and correctly-accumulated overall rates.
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    files: list[FileScore] = []
    count_errors: list[int] = []
    count_signed: list[int] = []
    for uri, reference, hypothesis, uem in items:
        components = metric(reference, hypothesis, uem=uem, detailed=True)
        total = float(components[_TOTAL])
        confusion = float(components[_CONFUSION])
        false_alarm = float(components[_FALSE_ALARM])
        missed = float(components[_MISS])
        ref_n = len(reference.labels())
        hyp_n = len(hypothesis.labels())
        files.append(
            FileScore(
                uri=uri,
                der=_rate(confusion + false_alarm + missed, total),
                confusion=confusion,
                false_alarm=false_alarm,
                missed_detection=missed,
                total=total,
                ref_speakers=ref_n,
                hyp_speakers=hyp_n,
                speaker_count_error=abs(ref_n - hyp_n),
            )
        )
        count_errors.append(abs(ref_n - hyp_n))
        count_signed.append(hyp_n - ref_n)

    return RunReport(
        files=files,
        overall_der=abs(metric),
        overall_confusion=float(metric[_CONFUSION]),
        overall_false_alarm=float(metric[_FALSE_ALARM]),
        overall_missed_detection=float(metric[_MISS]),
        speaker_count_mae=_rate(sum(count_errors), len(count_errors)),
        speaker_count_bias=_rate(sum(count_signed), len(count_signed)),
    )
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `uv run --group eval pytest eval/tests/test_score.py -v`
Expected: PASS (4 tests). If a component-key `KeyError` appears, fix the `_CONFUSION/_MISS/_FALSE_ALARM/_TOTAL` constants to the installed version's keys and re-run.

- [ ] **Step 5: Commit.**

```bash
git add eval/score.py eval/tests/test_score.py
git commit -m "feat(eval): DER + components + speaker-count scoring"
```

---

## Task 4: `prepare_data.py` — emit the `pyannote.database` registry

**Files:**
- Create: `eval/prepare_data.py`
- Test: `eval/tests/test_prepare_data.py`

**Interfaces:**
- Produces:
  - `ProtocolPaths` dataclass: `name: str`, `audio_dir: str`, `rttm: str`, `uem: str`.
  - `build_database_yml(protocols: list[ProtocolPaths]) -> str` — pure; returns the `database.yml` text registering each protocol under `Protocols:`.
  - `write_database_yml(protocols: list[ProtocolPaths], out_path: str) -> None`.
  - `main()` (CLI) — the documented fetch step (not unit-tested; performs network downloads).

> **Scope note:** the corpus *download* (VoxConverse audio + RTTM from the Oxford/joonson releases; AMI via `pyannote/AMI-diarization-setup`) is an integration action run once on a networked dev box — it is driven by `main()` and documented in the README, not unit-tested. This task unit-tests only the pure `database.yml` generation, which is the part that must be exactly right for `pyannote.database` to resolve the protocols.

- [ ] **Step 1: Write the failing test.** `eval/tests/test_prepare_data.py`:

```python
"""Tests for the pyannote.database registry generation."""

import yaml

from eval.prepare_data import ProtocolPaths, build_database_yml


def test_database_yml_registers_each_protocol() -> None:
    """Each ProtocolPaths becomes a resolvable Databases + Protocols entry."""
    protocols = [
        ProtocolPaths(name="VoxConverse", audio_dir="/data/voxconverse/audio", rttm="/data/voxconverse/dev.rttm", uem="/data/voxconverse/dev.uem"),
        ProtocolPaths(name="AMI", audio_dir="/data/ami/audio", rttm="/data/ami/test.rttm", uem="/data/ami/test.uem"),
    ]
    text = build_database_yml(protocols)
    parsed = yaml.safe_load(text)
    assert set(parsed["Protocols"]) == {"VoxConverse", "AMI"}
    vox = parsed["Protocols"]["VoxConverse"]["SpeakerDiarization"]["Benchmark"]
    assert vox["test"]["annotation"] == "/data/voxconverse/dev.rttm"
    assert vox["test"]["annotated"] == "/data/voxconverse/dev.uem"
    assert "/data/voxconverse/audio" in parsed["Databases"]["VoxConverse"][0]
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run --group eval pytest eval/tests/test_prepare_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.prepare_data'`.

- [ ] **Step 3: Write minimal implementation.** `eval/prepare_data.py`:

```python
"""Prepare public diarization benchmarks for the eval harness.

Downloads the canonical VoxConverse (dev) and AMI (test) releases and emits a
``pyannote.database`` ``database.yml`` that registers each as a
``<Name>.SpeakerDiarization.Benchmark`` protocol. Only ``main()`` touches the
network; it runs once on a networked dev box (never at serve time).
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ProtocolPaths:
    """Filesystem locations for one benchmark's audio and references.

    Attributes:
        name: Database/protocol name (e.g. ``VoxConverse``).
        audio_dir: Directory of the recordings (referenced as a glob).
        rttm: Path to the reference RTTM covering the split.
        uem: Path to the UEM defining the scored regions.
    """

    name: str
    audio_dir: str
    rttm: str
    uem: str


def build_database_yml(protocols: list[ProtocolPaths]) -> str:
    """Render a pyannote.database registry for the given protocols.

    Args:
        protocols: One entry per benchmark split to register.

    Returns:
        YAML text with ``Databases`` (audio globs) and ``Protocols``
        (a ``SpeakerDiarization.Benchmark`` protocol per entry, its ``test``
        subset pointing at the RTTM/UEM).
    """
    databases: dict[str, list[str]] = {}
    registry: dict[str, dict[str, object]] = {}
    for proto in protocols:
        databases[proto.name] = [f"{proto.audio_dir}/{{uri}}.wav"]
        registry[proto.name] = {
            "SpeakerDiarization": {
                "Benchmark": {
                    "test": {"annotation": proto.rttm, "annotated": proto.uem},
                }
            }
        }
    return yaml.safe_dump({"Databases": databases, "Protocols": registry}, sort_keys=True)


def write_database_yml(protocols: list[ProtocolPaths], out_path: str) -> None:
    """Write the rendered registry to ``out_path``.

    Args:
        protocols: Protocols to register.
        out_path: Destination path for ``database.yml``.
    """
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(build_database_yml(protocols))


def main() -> None:
    """Fetch the benchmark corpora and emit ``data/database.yml`` (networked).

    Downloads VoxConverse (dev) and AMI (test) into ``data/`` and writes the
    registry. Idempotent: existing files are left in place. Implemented as the
    documented one-time prep step; see ``eval/README.md`` for the exact source
    URLs and the manual steps for the gated AMI audio.
    """
    raise NotImplementedError(
        "Run the documented data-prep steps in eval/README.md, then call "
        "write_database_yml(...) with the resulting paths."
    )
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `uv run --group eval pytest eval/tests/test_prepare_data.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit.**

```bash
git add eval/prepare_data.py eval/tests/test_prepare_data.py
git commit -m "feat(eval): pyannote.database registry generation"
```

---

## Task 5: Factor server internals — `diarize_audio.py` + `diarize_pipeline.py` + refactor server

**Files:**
- Create: `src/diarize_audio.py`
- Create: `src/diarize_pipeline.py`
- Modify: `src/diarize_server.py` (imports + drop inline construction/decoder)
- Test: `eval/tests/test_diarize_pipeline.py`

**Interfaces:**
- Consumes: `eval` group only (tests). `diarize_compat` (existing module, imported lazily inside `build_pipeline`).
- Produces:
  - `src/diarize_audio.py`: `SAMPLE_RATE: int = 16_000`; `decode_audio(data: bytes) -> np.ndarray`.
  - `src/diarize_pipeline.py`: `build_pipeline(*, model_id=None, device=None, clustering_threshold=None, segmentation_min_duration_off=None) -> "Pipeline"`; pure `_resolve_param_overrides(defaults: dict[str, Any], *, clustering_threshold: float | None, segmentation_min_duration_off: float | None) -> dict[str, Any] | None`.

> **Testing boundary:** `_resolve_param_overrides` carries the behavior-preserving guarantee (no override → `None` → caller skips `instantiate()`), and is fully unit-tested torch-free. `build_pipeline`'s thin pyannote wiring (load / conditional instantiate / `.to(device)`) is validated by the real run in Task 8 — do **not** unit-test it (it would download a model). Confirm `Pipeline.parameters(instantiated=True)` returns the nested params dict in the installed version during Task 8; if that accessor differs, adjust only `build_pipeline`.

- [ ] **Step 1: Write the failing test.** `eval/tests/test_diarize_pipeline.py`:

```python
"""Tests for the pure hyperparameter-override resolver (torch-free)."""

import sys
from pathlib import Path

import pytest

# src/ is not a package; make its modules importable for the unit test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diarize_pipeline import _resolve_param_overrides  # noqa: E402


def test_no_overrides_returns_none() -> None:
    """No override requested → None, so the caller preserves pretrained defaults."""
    defaults = {"clustering": {"threshold": 0.7}, "segmentation": {"min_duration_off": 0.0}}
    assert _resolve_param_overrides(defaults, clustering_threshold=None, segmentation_min_duration_off=None) is None


def test_threshold_override_is_merged_without_mutating_defaults() -> None:
    """Overriding the threshold changes only that key and leaves defaults intact."""
    defaults = {"clustering": {"threshold": 0.7, "method": "centroid"}, "segmentation": {"min_duration_off": 0.0}}
    merged = _resolve_param_overrides(defaults, clustering_threshold=0.55, segmentation_min_duration_off=None)
    assert merged is not None
    assert merged["clustering"]["threshold"] == 0.55
    assert merged["clustering"]["method"] == "centroid"
    assert merged["segmentation"]["min_duration_off"] == 0.0
    assert defaults["clustering"]["threshold"] == 0.7  # unmutated


def test_missing_knob_raises() -> None:
    """Requesting a knob the loaded pipeline lacks is a loud error, not a no-op."""
    defaults = {"segmentation": {"min_duration_off": 0.0}}
    with pytest.raises(ValueError, match="clustering.threshold"):
        _resolve_param_overrides(defaults, clustering_threshold=0.5, segmentation_min_duration_off=None)
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run --group eval pytest eval/tests/test_diarize_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'diarize_pipeline'`.

- [ ] **Step 3a: Create `src/diarize_audio.py`** (move the decoder verbatim from the server, docstring preserved):

```python
"""Decode arbitrary media bytes to 16 kHz mono float32 PCM via ffmpeg.

Extracted from ``diarize_server`` so the eval harness decodes audio through the
exact same path the server uses (eval == production).
"""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np

SAMPLE_RATE = 16_000


def decode_audio(data: bytes) -> np.ndarray:
    """Decode media bytes to a 1-D float32 array at ``SAMPLE_RATE``.

    Applies the same s16le -> float32 / 32768 normalization that
    ``whisper.load_audio`` performs. The bytes are spooled to a temp file
    because MP4-family containers with a trailing moov atom cannot be demuxed
    from a non-seekable stdin pipe.

    Args:
        data: Raw bytes of any container/codec ffmpeg can decode.

    Returns:
        A 1-D float32 array of samples at ``SAMPLE_RATE``.

    Raises:
        ValueError: If ffmpeg fails or the payload holds no audio samples.
    """
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-threads", "0", "-i", tmp.name,
             "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "pipe:1"],
            capture_output=True,
        )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"ffmpeg could not decode payload: {tail}")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("decoded audio contains no samples")
    return audio
```

- [ ] **Step 3b: Create `src/diarize_pipeline.py`:**

```python
"""Build the pyannote diarization pipeline, optionally overriding hyperparameters.

Single home for pipeline construction, shared by the server and the eval
harness so both run byte-identical pipelines. pyannote (and the ``diarize_compat``
shim it needs) are imported lazily inside ``build_pipeline`` so importing this
module — e.g. for the pure ``_resolve_param_overrides`` unit tests — never pulls
in torch.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch-free at runtime; only for type-checkers
    from pyannote.audio import Pipeline

_DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


def _resolve_param_overrides(
    defaults: dict[str, Any],
    *,
    clustering_threshold: float | None,
    segmentation_min_duration_off: float | None,
) -> dict[str, Any] | None:
    """Merge requested hyperparameter overrides onto the pipeline's defaults.

    Args:
        defaults: The pipeline's instantiated parameter tree.
        clustering_threshold: New clustering threshold, or None to leave it.
        segmentation_min_duration_off: New segmentation value, or None.

    Returns:
        A new merged parameter dict, or None when no override was requested (the
        caller then skips ``instantiate`` entirely, preserving defaults exactly).

    Raises:
        ValueError: If an override targets a key absent from ``defaults``.
    """
    if clustering_threshold is None and segmentation_min_duration_off is None:
        return None
    merged = copy.deepcopy(defaults)
    if clustering_threshold is not None:
        if "threshold" not in merged.get("clustering", {}):
            raise ValueError(f"loaded pipeline has no clustering.threshold to override; keys={sorted(merged)}")
        merged["clustering"]["threshold"] = clustering_threshold
    if segmentation_min_duration_off is not None:
        if "min_duration_off" not in merged.get("segmentation", {}):
            raise ValueError(f"loaded pipeline has no segmentation.min_duration_off to override; keys={sorted(merged)}")
        merged["segmentation"]["min_duration_off"] = segmentation_min_duration_off
    return merged


def build_pipeline(
    *,
    model_id: str | None = None,
    device: str | None = None,
    clustering_threshold: float | None = None,
    segmentation_min_duration_off: float | None = None,
) -> "Pipeline":
    """Load the diarization pipeline, applying overrides only when given.

    With no overrides this is byte-for-byte the server's historical construction
    (stock model from ``DIARIZE_MODEL``, no ``instantiate`` call).

    Args:
        model_id: pyannote pipeline id; None → ``DIARIZE_MODEL`` env or 3.1.
        device: Torch device; None → ``DIARIZE_DEVICE`` env or ``cuda``.
        clustering_threshold: Clustering-threshold override, or None.
        segmentation_min_duration_off: Segmentation override, or None.

    Returns:
        The instantiated pyannote ``Pipeline`` moved to ``device``.

    Raises:
        RuntimeError: If ``from_pretrained`` returns None (gated-repo misconfig).
    """
    import diarize_compat  # noqa: F401 — restore torchaudio shims BEFORE importing pyannote
    import torch
    from pyannote.audio import Pipeline

    resolved_model = model_id or os.environ.get("DIARIZE_MODEL", _DEFAULT_MODEL)
    resolved_device = device or os.environ.get("DIARIZE_DEVICE", "cuda")
    pipeline = Pipeline.from_pretrained(resolved_model, use_auth_token=os.environ.get("HF_TOKEN") or None)
    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained({resolved_model!r}) returned None — gated-repo access "
            "missing? Accept the conditions for the model and its segmentation dependency "
            "on the Hugging Face Hub, then run once with HF_HUB_OFFLINE=0, "
            "TRANSFORMERS_OFFLINE=0, and HF_TOKEN set."
        )
    overrides = _resolve_param_overrides(
        pipeline.parameters(instantiated=True),
        clustering_threshold=clustering_threshold,
        segmentation_min_duration_off=segmentation_min_duration_off,
    )
    if overrides is not None:
        pipeline.instantiate(overrides)
    pipeline.to(torch.device(resolved_device))
    return pipeline
```

- [ ] **Step 3c: Refactor `src/diarize_server.py`.** Remove the `# isort: off … from pyannote.audio import Pipeline … # isort: on` block, the inline `pipeline = Pipeline.from_pretrained(...)` / `None`-guard / `pipeline.to(...)` block, and the module-level `_decode_audio` function. Replace with imports at the top of the module (after the existing stdlib/third-party imports):

```python
from diarize_audio import SAMPLE_RATE, decode_audio
from diarize_pipeline import build_pipeline
```

Delete the now-unused `import tempfile`, `import subprocess`, and the local `SAMPLE_RATE = 16_000` line (now imported). Keep `import torch` (used for the waveform tensor) and `import numpy as np`. Replace the construction block with:

```python
pipeline = build_pipeline()  # env defaults, no overrides — identical to before
_pipeline_lock = threading.Lock()
```

In `diarize(...)`, replace the `try: audio = _decode_audio(audio_bytes)` call with `decode_audio(audio_bytes)` (same name minus the leading underscore). Everything else in the route is unchanged.

- [ ] **Step 4: Run tests + lint + type-check.**

Run: `uv run --group eval pytest eval/tests/test_diarize_pipeline.py -v`
Expected: PASS (3 tests).
Run: `uv run --group dev pre-commit run ruff pyrefly --files src/diarize_audio.py src/diarize_pipeline.py src/diarize_server.py`
Expected: ruff + pyrefly PASS (pyannote resolves to Any under `ignore-missing-imports`). Fix any docstring/annotation findings before continuing.

- [ ] **Step 5: Commit.**

```bash
git add src/diarize_audio.py src/diarize_pipeline.py src/diarize_server.py eval/tests/test_diarize_pipeline.py
git commit -m "refactor(diarize): extract decode + pipeline factory; parameterize hyperparameters"
```

---

## Task 6: `run.py` — run a config over a protocol → hypothesis RTTM

**Files:**
- Create: `eval/run.py`
- Test: `eval/tests/test_run.py`

**Interfaces:**
- Consumes: `DiarizeConfig` (Task 2), `decode_audio` (Task 5), `build_pipeline` (Task 5, only in `main()`), `pyannote.core.Annotation`.
- Produces:
  - `run_diarization(pipeline: Any, files: list[tuple[str, str]], out_dir: str, config: DiarizeConfig) -> list[str]` — for each `(uri, audio_path)`, decode via `decode_audio`, call `pipeline({"waveform": ..., "sample_rate": SAMPLE_RATE}, **config.pipeline_kwargs)`, write `<out_dir>/<uri>.rttm`, return the written paths. `pipeline` is any callable returning a pyannote `Annotation` (injected, so tests need no torch).
  - `main()` — CLI wiring `build_pipeline(...)` + a `pyannote.database` protocol.

- [ ] **Step 1: Write the failing test.** `eval/tests/test_run.py`:

```python
"""Tests for run_diarization with an injected fake pipeline (torch-free)."""

import sys
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.configs import DiarizeConfig  # noqa: E402
from eval.run import run_diarization  # noqa: E402


class _FakePipeline:
    """Records call kwargs and returns a fixed two-speaker annotation."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, payload: dict, **kwargs: object) -> Annotation:
        self.calls.append(kwargs)
        ann = Annotation(uri=payload.get("uri"))
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"
        ann[Segment(1.0, 2.0)] = "SPEAKER_01"
        return ann


def test_writes_one_rttm_per_file_and_passes_bounds(tmp_path, monkeypatch) -> None:
    """Each file yields an RTTM; config speaker-bounds reach the pipeline call."""
    monkeypatch.setattr("eval.run.decode_audio", lambda data: np.zeros(16000, dtype=np.float32))  # bypass ffmpeg
    (tmp_path / "a.wav").write_bytes(b"x")
    fake = _FakePipeline()
    out = tmp_path / "hyp"
    written = run_diarization(fake, [("a", str(tmp_path / "a.wav"))], str(out), DiarizeConfig(label="floor2", min_speakers=2))
    assert written == [str(out / "a.rttm")]
    assert (out / "a.rttm").read_text().count("SPEAKER_") == 2
    assert fake.calls[0] == {"min_speakers": 2}
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run --group eval pytest eval/tests/test_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.run'`.

- [ ] **Step 3: Write minimal implementation.** `eval/run.py`:

```python
"""Run one diarization config over a set of audio files → hypothesis RTTM.

``run_diarization`` takes an already-built pipeline (any callable returning a
pyannote ``Annotation``) so it is testable without torch; ``main`` wires the real
``build_pipeline`` and a ``pyannote.database`` protocol.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

# src/ is not a package; import the shared server helpers from it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from diarize_audio import SAMPLE_RATE, decode_audio  # noqa: E402

from eval.configs import DiarizeConfig  # noqa: E402


def run_diarization(pipeline: Any, files: list[tuple[str, str]], out_dir: str, config: DiarizeConfig) -> list[str]:
    """Diarize each file and write one hypothesis RTTM per recording.

    Args:
        pipeline: Callable ``(payload, **bounds) -> Annotation`` (real or fake).
        files: ``(uri, audio_path)`` pairs to process.
        out_dir: Directory to write ``<uri>.rttm`` into (created if absent).
        config: The configuration; its ``pipeline_kwargs`` are passed through.

    Returns:
        The list of written RTTM paths, in input order.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for uri, audio_path in files:
        with open(audio_path, "rb") as handle:
            audio = decode_audio(handle.read())
        waveform = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        annotation = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": uri}, **config.pipeline_kwargs)
        out_path = os.path.join(out_dir, f"{uri}.rttm")
        with open(out_path, "w", encoding="utf-8") as rttm:
            annotation.write_rttm(rttm)
        written.append(out_path)
    return written


def main() -> None:
    """CLI: build the pipeline for a config and run it over a database protocol.

    Loads ``data/database.yml`` (see ``eval/prepare_data.py``), resolves the
    requested ``<Name>.SpeakerDiarization.Benchmark`` protocol's ``test`` files,
    builds the pipeline via ``build_pipeline`` and writes hypotheses. See
    ``eval/README.md`` for arguments.
    """
    raise NotImplementedError("Wire argparse + pyannote.database in Task 8's runbook; see eval/README.md.")
```

> **Note:** `pipeline` passes torch tensors in production, but `run_diarization` only reshapes a numpy array here; the real `build_pipeline` pipeline accepts the `{"waveform","sample_rate"}` dict with a numpy or tensor waveform (pyannote wraps it). The `main()` body is fleshed out in Task 8 against real data.

- [ ] **Step 4: Run test to verify it passes.**

Run: `uv run --group eval pytest eval/tests/test_run.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit.**

```bash
git add eval/run.py eval/tests/test_run.py
git commit -m "feat(eval): run a config over files → hypothesis RTTM"
```

---

## Task 7: `sweep.py` — config grid → sorted comparison table

**Files:**
- Create: `eval/sweep.py`
- Test: `eval/tests/test_sweep.py`

**Interfaces:**
- Consumes: `DiarizeConfig` (Task 2), `RunReport` (Task 3).
- Produces: `summarize_sweep(results: list[tuple[DiarizeConfig, RunReport]]) -> str` — a Markdown table, one row per config, sorted ascending by `overall_der`, logging every config so none is silently dropped.

- [ ] **Step 1: Write the failing test.** `eval/tests/test_sweep.py`:

```python
"""Tests for the sweep comparison-table rendering."""

from eval.configs import DiarizeConfig
from eval.score import RunReport
from eval.sweep import summarize_sweep


def _report(der: float) -> RunReport:
    """A minimal RunReport carrying just an overall DER."""
    return RunReport(files=[], overall_der=der, overall_confusion=0.0, overall_false_alarm=0.0,
                     overall_missed_detection=0.0, speaker_count_mae=0.0, speaker_count_bias=0.0)


def test_table_sorted_ascending_by_der() -> None:
    """The best (lowest-DER) config appears first."""
    results = [
        (DiarizeConfig(label="baseline"), _report(0.40)),
        (DiarizeConfig(label="community1"), _report(0.22)),
    ]
    table = summarize_sweep(results)
    assert table.index("community1") < table.index("baseline")
    assert "0.220" in table and "0.400" in table
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `uv run --group eval pytest eval/tests/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.sweep'`.

- [ ] **Step 3: Write minimal implementation.** `eval/sweep.py`:

```python
"""Compare diarization configurations by DER.

``summarize_sweep`` renders a ranked table from ``(config, RunReport)`` pairs.
The actual run/score orchestration is driven from ``main`` (Task 8 runbook) so
that the pure table rendering stays unit-testable without torch or data.
"""

from __future__ import annotations

from eval.configs import DiarizeConfig
from eval.score import RunReport


def summarize_sweep(results: list[tuple[DiarizeConfig, RunReport]]) -> str:
    """Render a DER-ranked comparison table over the swept configurations.

    Args:
        results: One ``(config, report)`` pair per configuration evaluated.

    Returns:
        A Markdown table sorted ascending by overall DER (best first), one row
        per configuration.
    """
    ranked = sorted(results, key=lambda pair: pair[1].overall_der)
    header = (
        "| config | DER | conf | FA | miss | count MAE |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {cfg.label} | {rep.overall_der:.3f} | {rep.overall_confusion:.1f} | "
        f"{rep.overall_false_alarm:.1f} | {rep.overall_missed_detection:.1f} | {rep.speaker_count_mae:.2f} |\n"
        for cfg, rep in ranked
    )
    return header + body
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `uv run --group eval pytest eval/tests/test_sweep.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Run the whole harness suite + lint gate.**

Run: `uv run --group eval pytest eval/tests -v`
Expected: PASS (all tests from Tasks 2–7).
Run: `uv run --group dev pre-commit run ruff pyrefly --files eval/configs.py eval/score.py eval/prepare_data.py eval/run.py eval/sweep.py`
Expected: ruff + pyrefly PASS.

- [ ] **Step 6: Commit.**

```bash
git add eval/sweep.py eval/tests/test_sweep.py
git commit -m "feat(eval): DER-ranked config comparison table"
```

---

## Task 8: README + end-to-end baseline runbook (integration)

**Files:**
- Create: `eval/README.md`
- Modify: `eval/prepare_data.py::main`, `eval/run.py::main`, `eval/sweep.py` (fill CLIs against real data)

**Interfaces:**
- Consumes: everything above + the `eval-run` group (pyannote.audio) + downloaded corpora + a GPU.
- Produces: a documented, repeatable flow that yields the first baseline DER (`report.md`). This task is an **integration/runbook** step — it requires network, the heavy install, and model access — so it is verified by producing a real report over a small subset, not by a unit test.

- [ ] **Step 1: Write `eval/README.md`** with: purpose (dev-only, not shipped); install (`uv sync --group eval --group eval-run`); the one-time data prep (VoxConverse dev audio + RTTM sources; AMI via `pyannote/AMI-diarization-setup`; where files land under `data/`; how `write_database_yml` is called); the HF gated-model note for `community-1` (`HF_TOKEN` + accept the license); and the run/sweep commands. Include an explicit "small subset first" instruction (2–3 VoxConverse files) to smoke the pipeline before the full set.

- [ ] **Step 2: Flesh out the three `main()`/CLI bodies** to parse args and wire `pyannote.database` (`get_protocol`, iterate `protocol.test()`, each file's `annotation`/`annotated` for scoring via `score_run`, `audio_path` for `run_diarization`). Keep the pure functions from Tasks 2–7 untouched.

- [ ] **Step 3: Install the heavy group and smoke a tiny subset.**

Run: `uv sync --group eval --group eval-run`
Then prepare 2–3 VoxConverse dev files into `data/` per the README and run the baseline config over just those.
Expected: `run.py` writes hypothesis RTTMs; `score.py` prints a `report.md` with a plausible per-file + OVERALL DER. This confirms `build_pipeline().parameters(instantiated=True)` / `instantiate` wiring works on the installed pyannote version (adjust `build_pipeline` if the accessor differs), and that decode → pipeline → RTTM → score runs end to end.

- [ ] **Step 4: Record the baseline.** Run the baseline config (`DiarizeConfig(label="baseline-3.1")`, no overrides) over the full VoxConverse dev set (and AMI test if prepared); commit the resulting `report.md` under `eval/reports/2026-07-11-baseline.md` and sanity-check the DER against pyannote's published VoxConverse/AMI numbers (same ballpark ⇒ harness is wired correctly).

- [ ] **Step 5: Commit.**

```bash
git add eval/README.md eval/prepare_data.py eval/run.py eval/sweep.py eval/reports/2026-07-11-baseline.md
git commit -m "feat(eval): end-to-end runbook + recorded 3.1 baseline DER"
```

---

## Self-Review

**1. Spec coverage.**
- *Measure backend in-process* → Tasks 5–6 (factory + run over decoded audio). ✓
- *Shared behavior-preserving factory* → Task 5 (`_resolve_param_overrides` returns None when no override; server refactor). ✓
- *VoxConverse + AMI via pyannote.database* → Tasks 4, 8. ✓
- *DER + components (miss/FA/confusion) + speaker-count (MAE + bias)* → Task 3. ✓
- *0.25 s collar, overlap included, within UEM; correct accumulation* → Task 3 (`score_run` single-metric accumulation, `uem=` passthrough). ✓
- *Sweep grid → ranked table* → Task 7 + Task 8 CLI. ✓
- *configs.py / run.py / score.py / sweep.py / prepare_data.py / README* → Tasks 2–8. ✓
- *Dev-only, excluded from images/bundle; airgap no serve-time fetch* → Task 1 (.dockerignore, .gitignore) + Task 4 (only `main()` fetches). ✓
- *Tests torch-free (mock pipeline / pure helpers)* → Tasks 2,3,4,5,6,7 all run under `--group eval` (no pyannote.audio). ✓
- *Model-aware knob guard; loud error on missing knob* → Task 5 (`_resolve_param_overrides` raises). ✓
- *Tuning loop (baseline → community-1 → sweep)* → enabled by the parameterized factory + sweep; baseline recorded in Task 8. ✓
- *Error handling: gated-model None-guard message preserved* → Task 5 (verbatim message). ✓

**2. Placeholder scan.** The `main()` bodies intentionally `raise NotImplementedError` in Tasks 4/6 and are completed in Task 8 (integration) — this is a deliberate two-phase split (pure logic unit-tested first, network/GPU wiring last), not a placeholder in shipped logic. All unit-tested functions are fully implemented. No `TODO`/`TBD`/"add error handling" left.

**3. Type consistency.** `DiarizeConfig` fields and `pipeline_kwargs`/`as_dict` match across Tasks 2/6/7. `RunReport` fields match between Task 3 (definition) and Task 7 (`_report` fixture + `summarize_sweep` access). `_resolve_param_overrides` signature matches between Task 5 definition and its test. `decode_audio`/`SAMPLE_RATE` names match between `diarize_audio.py` (Task 5) and `run.py` (Task 6). `score_run` tuple shape `(uri, ref, hyp, uem)` matches between Task 3 and the Task 8 CLI description.

*Fixes applied inline: none needed.*
