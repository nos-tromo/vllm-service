# Build-host helpers for the vLLM service stack.

.PHONY: bundle bundle-media build build-media no-build no-build-media up up-media stop stop-media

# Versioned image tag.
# On production: read from .vllm-service-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting VLLM_SERVICE_VERSION before invoking make.
VLLM_SERVICE_VERSION ?= $(shell \
    cat .vllm-service-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export VLLM_SERVICE_VERSION

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
no-build:
	docker compose up --no-build

# Core + media services (translate, audio).
no-build-media:
	docker compose --profile media up --no-build

# Start core stack only (chat, embed, rerank).
up:
	docker compose up

# Start media services only (audio, translate).
up-media:
	docker compose --profile media up

# Stop all services.
stop:
	docker compose stop

# Stop the core + media services (audio, translate).
stop-media:
	docker compose --profile media stop