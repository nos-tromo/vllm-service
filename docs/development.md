# Development

## Linting the Python sources

The FastAPI servers in `src/`, the router's audit callback in
`docker/router/` and the tests in `eval/` are linted with the nos-tromo
org-wide strict regime — `ruff` + `pyrefly (strict)` via
[`.pre-commit-config.yaml`](../.pre-commit-config.yaml), mirroring the canonical
config in [`nos-tromo/.github`](https://github.com/nos-tromo/.github)
`configs/python-strict/`. The `python-lint` CI job (in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the same hooks and
additionally runs `validate_strict_config.py`, which fails the build if this
repo's `pyproject.toml` / `.pre-commit-config.yaml` drift from the canonical
config. The vendored audit library (`docker/router/audit_event.py`,
`docker/router/audit_runtime.py`) must stay byte-identical to its canonical
copy in `nos-tromo/.github` `configs/audit/`, which that repo's
`scripts/validate_audit_lib.py --consumer-root .` checks; never edit those two
files, re-vendor them.

Run it locally with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                              # create the lint venv (pyrefly, pre-commit, typed deps)
uv run pre-commit run --all-files    # ruff check + ruff format + pyrefly over src/, eval/, docker/router/
```

The heavy ML backends (torch, transformers, openai-whisper, pyannote, silero)
and LiteLLM are **not** installed for linting — `pyrefly` treats them as `Any`
(`ignore-missing-imports`). Only the light, typed shared deps (`fastapi`,
`pydantic`, `numpy`) are installed, so strict mode type-checks the first-party
code.

## Unit tests

The unit tests live in `eval/tests/` next to the diarization eval harness
(see [`eval/README.md`](../eval/README.md)) and stub the heavy dependencies,
so they need no GPU, no model and no network. CI does not run them.

```bash
uv run --group eval pytest eval/tests/                           # every unit test
uv run --group eval pytest eval/tests/test_router_audit.py       # the router's audit callback
uv run --group eval pytest eval/tests/test_revision_pinning.py   # every model launch's revision
```

The audit callback's tests stub LiteLLM and call its hooks in the order the
pinned LiteLLM release calls them for each route; when you bump the router
image, check that order against the new release.

The revision tests run each vLLM command from `docker/compose.yaml` under `sh`
(`dash` where installed) with a stub `vllm` that prints its arguments, and
import the first-party servers over stub loaders that record what they were
asked to load. A new `vllm serve` service fails them until it has a revision
variable.
