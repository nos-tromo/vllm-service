#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091,SC2154  # sources vendored scripts/bundle-lib.sh (sets BUNDLE_*)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
. scripts/bundle-lib.sh

# The shape arg selects a standalone compose project (different topology, its
# own compose file). No arg = the full "core" stack.
PROFILE_ARG="${1:-}"
COMPOSE_FILE="docker/compose.yaml"
case "$PROFILE_ARG" in
  "")           PROFILE_LABEL="core" ;;
  gliner-only)  PROFILE_LABEL="gliner-only";  COMPOSE_FILE="docker/compose.gliner-only.yaml" ;;
  rerank-only)  PROFILE_LABEL="rerank-only";  COMPOSE_FILE="docker/compose.rerank-only.yaml" ;;
  clip-only)    PROFILE_LABEL="clip-only";    COMPOSE_FILE="docker/compose.clip-only.yaml" ;;
  diarize-only) PROFILE_LABEL="diarize-only"; COMPOSE_FILE="docker/compose.diarize-only.yaml" ;;
  asr-only)     PROFILE_LABEL="asr-only";     COMPOSE_FILE="docker/compose.asr-only.yaml" ;;
  vad-only)     PROFILE_LABEL="vad-only";     COMPOSE_FILE="docker/compose.vad-only.yaml" ;;
  embed-only)   PROFILE_LABEL="embed-only";   COMPOSE_FILE="docker/compose.embed-only.yaml" ;;
  *) echo "Usage: $0 [gliner-only|rerank-only|clip-only|diarize-only|asr-only|vad-only|embed-only]" >&2; exit 2 ;;
esac

[[ -n "${BUNDLE_DEV:-}" ]] || bundle_checkout_release vllm-service
bundle_version vllm-service; VER="$BUNDLE_VERSION"

COMPOSE=(docker compose --env-file .env -f "$COMPOSE_FILE")
"${COMPOSE[@]}" build
"${COMPOSE[@]}" pull --ignore-buildable
bundle_partition_images < <("${COMPOSE[@]}" config --images)

echo "Built images:  ${BUNDLE_BUILT[*]:-<none>}"
echo "Pulled images: ${BUNDLE_PULLED[*]:-<none>}"
if (( ${#BUNDLE_BUILT[@]} > 0 )); then
  docker save "${BUNDLE_BUILT[@]}" | gzip > "vllm-service-built-${PROFILE_LABEL}-${VER}.tar.gz"
fi
if (( ${#BUNDLE_PULLED[@]} > 0 )); then
  docker save "${BUNDLE_PULLED[@]}" | gzip > "vllm-service-pulled-${PROFILE_LABEL}-${VER}.tar.gz"
fi
echo "Wrote: vllm-service-built-${PROFILE_LABEL}-${VER}.tar.gz, vllm-service-pulled-${PROFILE_LABEL}-${VER}.tar.gz"
