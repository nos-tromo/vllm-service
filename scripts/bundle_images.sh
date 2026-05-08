#!/usr/bin/env bash
set -euo pipefail

# Profile arg: empty (default) = core stack; "media" = adds audio + translate.
PROFILE_ARG="${1:-}"
if [[ -n "$PROFILE_ARG" && "$PROFILE_ARG" != "media" ]]; then
  echo "Usage: $0 [media]" >&2
  exit 2
fi
PROFILE_LABEL="${PROFILE_ARG:-core}"
COMPOSE_PROFILE_FLAGS=()
[[ -n "$PROFILE_ARG" ]] && COMPOSE_PROFILE_FLAGS=(--profile "$PROFILE_ARG")

# YYYY-MM-DD plus short git sha; override by exporting VLLM_SERVICE_VERSION beforehand.
export VLLM_SERVICE_VERSION="${VLLM_SERVICE_VERSION:-$(date +%Y-%m-%d)-$(git rev-parse --short HEAD)}"
echo "VLLM_SERVICE_VERSION=$VLLM_SERVICE_VERSION"

docker compose ${COMPOSE_PROFILE_FLAGS[@]+"${COMPOSE_PROFILE_FLAGS[@]}"} build
docker compose ${COMPOSE_PROFILE_FLAGS[@]+"${COMPOSE_PROFILE_FLAGS[@]}"} pull --ignore-buildable

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
done < <(docker compose ${COMPOSE_PROFILE_FLAGS[@]+"${COMPOSE_PROFILE_FLAGS[@]}"} config --images)

echo "Built images:  ${built[*]:-<none>}"
echo "Pulled images: ${pulled[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "vllm-service-built-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz"
fi
if (( ${#pulled[@]} > 0 )); then
  docker save "${pulled[@]}" | gzip > "vllm-service-pulled-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz"
fi

echo "Wrote: vllm-service-built-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz, vllm-service-pulled-${PROFILE_LABEL}-${VLLM_SERVICE_VERSION}.tar.gz"
