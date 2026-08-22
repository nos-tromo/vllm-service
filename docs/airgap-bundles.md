# Offline image bundles

For airgapped hosts, customer deployments, or any environment without
Docker Hub access, `make bundle` produces a versioned `.tar.gz` pair you
can ship alongside the `docker/` directory (which holds `compose.yaml` and
`litellm.config.yaml`) and `.env`.

## Producing the bundle

On a build host with internet:

```bash
make bundle              # full stack (chat, embed, rerank, gliner, clip, asr, diarize, vad, router)
make bundle-gliner-only     # NER-only shape (just vllm-service-gliner-cpu)
make bundle-rerank-only  # Rerank-only shape (just vllm-service-rerank-only)
make bundle-clip-only    # CLIP-only shape (just vllm-service-clip-cpu)
make bundle-embed-only   # Embed-only shape (just vllm-service-embed-only)
make bundle-diarize-only # Diarize-only shape (just vllm-service-diarize-cpu)
make bundle-asr-only     # ASR-only shape (just vllm-service-asr-cpu)
make bundle-vad-only     # VAD-only shape (just vllm-service-vad-cpu)
```

This computes `VLLM_SERVICE_VERSION` as `YYYY-MM-DD-<short-sha>` (override by
exporting it before invocation), builds the locally-buildable services with
that version tag, pulls the externally-hosted images (LiteLLM Proxy), then
writes two gzipped tarballs in the cwd:

| File | Contents |
|---|---|
| `vllm-service-built-<profile>-<version>.tar.gz` | Locally-built `vllm-service-{chat,embed,rerank,gliner,...}` images. |
| `vllm-service-pulled-<profile>-<version>.tar.gz` | Externally-hosted images (LiteLLM router); re-tagged so the `name:tag@digest` references in `docker/compose.yaml` resolve after `docker load`. |

The compose file references the version through
`image: vllm-service-<svc>:${VLLM_SERVICE_VERSION:-latest}`, so it falls
back to `:latest` for normal dev workflows and uses the pinned tag whenever
the variable is set.

## Loading and running the bundle

Ship the two tarballs along with the matching `docker/` directory (which
holds `compose.yaml` and `litellm.config.yaml`) and a `.env`. Then on the
target host:

```bash
docker load -i vllm-service-built-core-<version>.tar.gz
docker load -i vllm-service-pulled-core-<version>.tar.gz
export VLLM_SERVICE_VERSION=<version>
docker compose --env-file .env -f docker/compose.yaml up --no-build -d
```

The target host runs the production shape — `docker/compose.yaml` without
the dev override — so no host ports are published.

The version is embedded in the tarball filenames, so the operator just
reads it off the file. Verify with `docker images | grep vllm-service`
between `load` and `up`.

> `--no-build` does **not** suppress pulls from a registry. If the tagged
> image isn't loaded locally, Compose still tries to resolve it against
> Docker Hub and errors with a DNS / "no such host" failure on offline
> machines. Always `docker load` first.
