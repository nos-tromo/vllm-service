# Build-host helpers for the vLLM service stack.

.PHONY: build build-media bundle bundle-media up up-media

# Versioning: use date + git short hash for image tags, but allow override via env var.
VLLM_SERVICE_VERSION ?= $(shell date +%Y-%m-%d)-$(shell git rev-parse --short HEAD)
export VLLM_SERVICE_VERSION

# Core stack only (chat, embed, rerank).
build:
	DOCKER_BUILDKIT=1 docker compose build

# Core + media services (translate, audio).
build-media:
	DOCKER_BUILDKIT=1 docker compose --profile media build

# Build core stack and ship as versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh

# Same as `bundle` but adds the media profile (audio + translate).
bundle-media:
	./scripts/bundle_images.sh media

# Start core stack only (chat, embed, rerank).
up:
	docker compose up -d

# Start media services only (audio, translate).
up-media:
	docker compose --profile media up -d