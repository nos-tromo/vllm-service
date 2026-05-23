# Build-host helpers for the vLLM service stack.
#
# The service set is read from PROFILE in .env: leave it empty for the core
# stack (chat, embed, rerank, ner); set PROFILE=media to also start audio +
# translate. Override per-invocation with `make up PROFILE=media`.

.DEFAULT_GOAL := help

.PHONY: help network volumes build bundle up stop

# Service-set profile. Read from .env; empty = core stack.
PROFILE ?= $(strip $(shell test -f .env && grep -E '^PROFILE=' .env | cut -d= -f2))

# Versioned image tag.
# On production: read from .vllm-service-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting VLLM_SERVICE_VERSION before invoking make.
VLLM_SERVICE_VERSION ?= $(shell \
    cat .vllm-service-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export VLLM_SERVICE_VERSION

COMPOSE      := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml
# Empty PROFILE -> no flag (core stack); PROFILE=media -> --profile media.
PROFILE_FLAG := $(if $(PROFILE),--profile $(PROFILE),)

help:
	@echo "vllm-service — build-host helpers. Active service set: $(if $(PROFILE),$(PROFILE),core)"
	@echo
	@echo "  make network   create the external inference-net"
	@echo "  make volumes   create the huggingface-cache Docker volume"
	@echo "  make build     build images for the active service set"
	@echo "  make bundle    ship images as a versioned .tar.gz pair"
	@echo "  make up        run the active service set (no rebuild)"
	@echo "  make stop      stop the active service set"
	@echo
	@echo "Leave PROFILE empty in .env for the core stack; set PROFILE=media"
	@echo "to add audio + translate. Override: make up PROFILE=media"

# Create the external Docker network (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true

# Create the external Hugging Face cache volume (one-time per host; idempotent).
volumes:
	docker volume create huggingface-cache >/dev/null 2>&1 || true

# Build images for the active service set.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) $(PROFILE_FLAG) build

# Build images and ship as a versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh $(PROFILE)

# Run the active service set without rebuilding images.
up:
	$(COMPOSE) $(PROFILE_FLAG) up --no-build

# Stop the active service set.
stop:
	$(COMPOSE) $(PROFILE_FLAG) stop
