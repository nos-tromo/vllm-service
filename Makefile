# Build-host helpers for the vLLM service stack.

.PHONY: network volume bundle bundle-media build build-media up up-media stop stop-media

# Versioned image tag.
# On production: read from .vllm-service-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting VLLM_SERVICE_VERSION before invoking make.
VLLM_SERVICE_VERSION ?= $(shell \
    cat .vllm-service-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export VLLM_SERVICE_VERSION

# Create the external Docker network (one-time per host; idempotent)
network:
	DOCKER_BUILDKIT=1 docker network create inference-net

# Create the external Docker volume for Hugging Face cache (one-time per host; idempotent
volume:
	DOCKER_BUILDKIT=1 docker volume create huggingface-cache

# Build core stack and ship as versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh

# Same as `bundle` but adds the media profile (audio + translate).
bundle-media:
	./scripts/bundle_images.sh media

# Core stack only (chat, embed, rerank).
build:
	DOCKER_BUILDKIT=1 docker compose build

# Core + media services (translate, audio).
build-media:
	DOCKER_BUILDKIT=1 docker compose --profile media build

# Core stack only (chat, embed, rerank).
up:
	DOCKER_BUILDKIT=1 docker compose up --no-build

# Core + media services (translate, audio).
up-media:
	DOCKER_BUILDKIT=1 docker compose --profile media up --no-build

# Stop all services.
stop:
	DOCKER_BUILDKIT=1 docker compose stop

# Stop the core + media services (audio, translate).
stop-media:
	DOCKER_BUILDKIT=1 docker compose --profile media stop