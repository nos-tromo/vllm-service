# Offline image bundles

For airgapped hosts, customer deployments, or any environment without
Docker Hub access, `make bundle` produces a versioned `.tar.gz` pair you
can ship alongside the `docker/` directory (which holds `compose.yaml` and
`litellm.config.yaml`) and `.env`.

## Producing the bundle

On a build host with internet:

```bash
make bundle              # full stack (chat, embed, rerank, gliner, clip, asr, diarize, vad, faces, topics, router)
make bundle-dev          # full stack, from the current working tree (dev iteration / soak)
make bundle-gliner-only     # NER-only shape (just vllm-service-gliner-cpu)
make bundle-rerank-only  # Rerank-only shape (just vllm-service-rerank-only)
make bundle-clip-only    # CLIP-only shape (just vllm-service-clip-cpu)
make bundle-embed-only   # Embed-only shape (just vllm-service-embed-only)
make bundle-diarize-only # Diarize-only shape (just vllm-service-diarize-cpu)
make bundle-asr-only     # ASR-only shape (just vllm-service-asr-cpu)
make bundle-vad-only     # VAD-only shape (just vllm-service-vad-cpu)
make bundle-topics-only  # Topics-only shape (just vllm-service-topics-cpu)
make bundle-faces-only   # Faces-only shape (just vllm-service-faces-cpu)
```

Each target builds the locally-buildable services with a version tag, pulls the
externally-hosted images (LiteLLM Proxy), then writes two gzipped tarballs in
the cwd. Where the version comes from depends on which target you ran.

**`make bundle` is the production path.** It resolves the latest **annotated**
tag reachable from `HEAD`, checks it out detached, and builds the tagged tree.
On success HEAD **stays on the tag**, so the repo files match the bundled
images; `git switch main` returns. A failed build switches back (an `EXIT`
trap), and when `HEAD` already is the tag's commit nothing is switched.
`VLLM_SERVICE_VERSION` is then that tag verbatim — `vX.Y.Z`. It refuses
rather than ever producing an unversioned artifact when:

- tracked files are dirty (a checkout would clobber them) — commit or stash
  first, or use `make bundle-dev`; or
- no annotated tag is reachable from `HEAD`.

Untracked and gitignored files such as `.env` are ignored by the dirty check
and survive the checkout.

**`make bundle-dev` is the dev path.** It skips the checkout and bundles the
current working tree as-is, versioned `<commit-date>-<short-sha>` (today's date
outside a git repo). Use it for dev iteration and staging soak; a release
candidate promoted to production must come from `make bundle`.

Exporting `VLLM_SERVICE_VERSION_OVERRIDE` short-circuits both paths: it bundles
the working tree as-is under the version you name. The nine
`make bundle-<shape>` targets take the production path as well — there is no
`bundle-dev-<shape>` target, so use `VLLM_SERVICE_VERSION_OVERRIDE` for a
working-tree bundle of a single shape.

Whichever path runs, the resolved version is exported as
`VLLM_SERVICE_VERSION`, echoed on stdout, and persisted to
`.vllm-service-version`. The two tarballs are:

| File | Contents |
|---|---|
| `vllm-service-built-<profile>-<version>.tar.gz` | Locally-built `vllm-service-{chat,embed,rerank,gliner,...}` images. |
| `vllm-service-pulled-<profile>-<version>.tar.gz` | Externally-hosted images (LiteLLM router); re-tagged so the `name:tag@digest` references in `docker/compose.yaml` resolve after `docker load`. |

The compose file references the version through
`image: vllm-service-<svc>:${VLLM_SERVICE_VERSION:-latest}`, so it falls
back to `:latest` for normal dev workflows and uses the pinned tag whenever
the variable is set.

The topics image installs from its own hash-checked lock, `docker/topics/uv.lock`.
The one build step outside it is on aarch64, where `hdbscan` compiles from its
sdist and fetches that build's requirements (`setuptools`, `wheel`, `cython`,
`numpy`) unpinned, so build it on the networked build host like every other
image; the tarball then carries everything the target needs. See
[Topics-only](deployment-shapes.md#topics-only).

The faces image also installs from its own hash-checked lock,
`docker/faces/uv.lock`; every package in it ships a wheel, so nothing is built
or fetched outside the lock. See [Faces-only](deployment-shapes.md#faces-only).

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

The bundles carry images, not model weights; those move into the
`huggingface-cache` volume with the deploy repo's model-transfer scripts. Set
each model's `*_MODEL_REVISION` in `.env` to the commit those scripts report:
the services then load exactly that snapshot, and one whose commit is missing
from the volume does not start. See
[configuration.md](configuration.md#model-revisions).

The `faces` service needs one file from each of two repositories,
`face_detection_yunet_2023mar.onnx` from `opencv/face_detection_yunet` and
`face_recognition_sface_2021dec.onnx` from `opencv/face_recognition_sface`
(YuNet is MIT-licensed, SFace Apache-2.0). Transfer only those files and each
repository's `LICENSE`: both repositories also hold `example_outputs/` with
photographs of people, which have no place on the target host. The service
never downloads; with either file missing it does not start. The router waits
only for it to start, so a host upgraded to this release without the two files
loses only `/faces/*`, never the rest of inference.

The version is embedded in the tarball filenames, so the operator just
reads it off the file. Verify with `docker images | grep vllm-service`
between `load` and `up`.

> `--no-build` does **not** suppress pulls from a registry. If the tagged
> image isn't loaded locally, Compose still tries to resolve it against
> Docker Hub and errors with a DNS / "no such host" failure on offline
> machines. Always `docker load` first.
