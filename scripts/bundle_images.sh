#!/usr/bin/env bash
set -euo pipefail

# Always compute a fresh version from git so repeated bundle runs produce
# distinct tags. Uses the commit date (not the build date) for reproducibility.
# Falls back to today's date when not in a git repo.
# .vllm-service-version (if present) is never used as input here — it is only
# written as output for production hosts.
# To pin a specific tag, set VLLM_SERVICE_VERSION_OVERRIDE in your shell before
# invoking make.
PROFILE_ARG="${1:-}"
COMPOSE_FILE="docker/compose.yaml"
case "$PROFILE_ARG" in
  "")
    PROFILE_LABEL="core"
    ;;
  "gliner-only")
    # Separate compose project — different topology (single CPU container
    # on inference-net, no router), so it has its own compose file and
    # carries no compose profile.
    PROFILE_LABEL="gliner-only"
    COMPOSE_FILE="docker/compose.gliner-only.yaml"
    ;;
  "rerank-only")
    # Separate compose project — single FastAPI/transformers container
    # on inference-net, no router, no GPU. Same topology shape as gliner-only.
    PROFILE_LABEL="rerank-only"
    COMPOSE_FILE="docker/compose.rerank-only.yaml"
    ;;
  "clip-only")
    # Separate compose project — single FastAPI/transformers CLIP
    # image+text container on inference-net, no router, no GPU. Same
    # topology shape as gliner-only/rerank-only.
    PROFILE_LABEL="clip-only"
    COMPOSE_FILE="docker/compose.clip-only.yaml"
    ;;
  "diarize-only")
    # Separate compose project — single FastAPI/pyannote speaker-diarization
    # container on inference-net, no router, no GPU. Same topology shape as
    # gliner-only/rerank-only/clip-only.
    PROFILE_LABEL="diarize-only"
    COMPOSE_FILE="docker/compose.diarize-only.yaml"
    ;;
  "asr-only")
    # Separate compose project — single FastAPI/openai-whisper ASR container
    # on inference-net, no router, no GPU. Same topology shape as the other
    # -only shapes.
    PROFILE_LABEL="asr-only"
    COMPOSE_FILE="docker/compose.asr-only.yaml"
    ;;
  "vad-only")
    # Separate compose project — single FastAPI/Silero voice-activity-detection
    # container on inference-net, no router, no GPU. Same topology shape as the
    # other -only shapes.
    PROFILE_LABEL="vad-only"
    COMPOSE_FILE="docker/compose.vad-only.yaml"
    ;;
  *)
    echo "Usage: $0 [gliner-only|rerank-only|clip-only|diarize-only|asr-only|vad-only]" >&2
    exit 2
    ;;
esac

# YYYY-MM-DD plus short git sha; override by exporting VLLM_SERVICE_VERSION beforehand.
if [[ -n "${VLLM_SERVICE_VERSION_OVERRIDE:-}" ]]; then
  export VLLM_SERVICE_VERSION="$VLLM_SERVICE_VERSION_OVERRIDE"
else
  _git_sha=$(git rev-parse --short HEAD 2>/dev/null || true)
  _git_date=$(git log -1 --format=%cs 2>/dev/null || true)
  _date="${_git_date:-$(date +%Y-%m-%d)}"
  export VLLM_SERVICE_VERSION="${_date}${_git_sha:+-${_git_sha}}"
fi
echo "VLLM_SERVICE_VERSION=$VLLM_SERVICE_VERSION"

# Persist the version so production hosts can run 'make no-build-*' without
# git or the original build date. Copy this file alongside docker-compose.yml.
echo "$VLLM_SERVICE_VERSION" > .vllm-service-version

docker compose --env-file .env -f "$COMPOSE_FILE" build
docker compose --env-file .env -f "$COMPOSE_FILE" pull --ignore-buildable

# Partition compose's image list into built (no slash) and pulled (registry refs).
# Docker Desktop sometimes drops the name:tag binding when you pull
# `name:tag@digest`, leaving only the digest. Re-tag explicitly so `docker save`
# produces a tarball that loads back with both tag and digest bindings —
# compose needs that for `image: name:tag@digest` references.
built=()
pulled=()
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  if [[ "$img" == */* ]]; then
    if [[ "$img" =~ ^(.+):([^@]+)@(sha256:[a-f0-9]+)$ ]]; then
      name="${BASH_REMATCH[1]}"
      tag="${BASH_REMATCH[2]}"
      digest="${BASH_REMATCH[3]}"
      docker tag "${name}@${digest}" "${name}:${tag}"
      pulled+=("${name}:${tag}")
    else
      pulled+=("$img")
    fi
  else
    built+=("$img")
  fi
done < <(docker compose --env-file .env -f "$COMPOSE_FILE" config --images)

echo "Built images:  ${built[*]:-<none>}"
echo "Pulled images: ${pulled[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "vllm-service-built-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz"
fi
if (( ${#pulled[@]} > 0 )); then
  docker save "${pulled[@]}" | gzip > "vllm-service-pulled-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz"
fi

echo "Wrote: vllm-service-built-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz, vllm-service-pulled-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz"
