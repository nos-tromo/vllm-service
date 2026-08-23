# vLLM Service Documentation

This directory contains the in-repo reference manual for the **standalone
vLLM service** — the shared inference tier of the nos-tromo federation. It
complements the top-level [`README.md`](../README.md) (which focuses on the
routed-endpoint model and the quick start) with topic-by-topic deep dives.

## Table of contents

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | The containers behind the LiteLLM router, the gliner Ray Serve watchdog, and the `vllm-net` / `inference-net` seams |
| [deployment-shapes.md](deployment-shapes.md) | The seven CPU-only standalone shapes: shared bring-up, cache population, per-shape defaults and gotchas |
| [configuration.md](configuration.md) | The env knobs that need explaining — Gemma tool calling, the embed batch budget, diarization speaker granularity — and how to switch models |
| [api-reference.md](api-reference.md) | Endpoint surface, the master-key auth model, and every request/response body for both the routed and standalone addresses |
| [airgap-bundles.md](airgap-bundles.md) | Producing versioned image tarballs on a build host and loading them on an airgapped target |
| [development.md](development.md) | The `ruff` + `pyrefly (strict)` regime for the FastAPI servers in `src/` |

## Who this is for

- **Operators** bringing the stack up on a CUDA host — start with the
  top-level [`README.md`](../README.md) quick start, then
  [configuration.md](configuration.md) for the knobs and
  [architecture.md](architecture.md) for what is actually running.
- **Operators on a non-CUDA host** (Mac dev box, ROCm, CPU-only Linux) — go
  straight to [deployment-shapes.md](deployment-shapes.md); it is the whole
  runbook for the `-only` containers.
- **Airgap / release engineers** shipping an artifact to a disconnected
  host — [airgap-bundles.md](airgap-bundles.md).
- **API consumers** wiring an app against the router or against a standalone
  shape — [api-reference.md](api-reference.md).

## Conventions used in these docs

- **Paths are relative to the repo root** (for example `docker/compose.yaml`,
  `src/embed_server.py`). No absolute or machine-local paths appear anywhere.
- **Every example is synthetic.** Sample texts, filenames and entity names are
  invented; no production or testing data appears in this repository.
- **Env knobs are quoted with their default** as shipped in `.env.example`,
  which remains the annotated source of truth for the full set.
- Documentation is plain Markdown (GitHub Flavored). No build step is
  required.
- Dated `YYYY-MM-DD-*.md` files alongside these pages are design and plan
  history, not reference material; they are not listed above.
