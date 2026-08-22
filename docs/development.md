# Development

## Linting the Python servers

The FastAPI servers in `src/` are linted with the nos-tromo org-wide strict
regime — `ruff` + `pyrefly (strict)` via
[`.pre-commit-config.yaml`](../.pre-commit-config.yaml), mirroring the canonical
config in [`nos-tromo/.github`](https://github.com/nos-tromo/.github)
`configs/python-strict/`. The `python-lint` CI job (in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the same hooks and
additionally runs `validate_strict_config.py`, which fails the build if this
repo's `pyproject.toml` / `.pre-commit-config.yaml` drift from the canonical
config.

Run it locally with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                              # create the lint venv (pyrefly, pre-commit, typed deps)
uv run pre-commit run --all-files    # ruff check + ruff format + pyrefly over src/
```

The heavy ML backends (torch, transformers, openai-whisper, pyannote, silero)
are **not** installed for linting — `pyrefly` treats them as `Any`
(`ignore-missing-imports`). Only the light, typed shared deps (`fastapi`,
`pydantic`, `numpy`) are installed, so strict mode type-checks the first-party
code.
