# Build-host helpers for the vLLM service stack.
#
# Two deployment shapes:
#
# 1. Full stack (CUDA host) — chat, embed, rerank, ner, router; optional
#    audio + translate via PROFILE=media. Lives in docker/compose.yaml.
#    Targets: build, up, stop, bundle.
#
# 2. NER-only (Mac, CPU-only, ROCm host running Ollama for chat/embed) —
#    just the GLiNER service on inference-net, no router, no GPU
#    requirement. Lives in docker/compose.gliner-only.yaml.
#    Targets: build-gliner-only, up-gliner-only, stop-gliner-only, bundle-gliner-only.
#
# 3. Rerank-only (same hosts as NER-only) — a single FastAPI/transformers
#    rerank container on inference-net, no router, no GPU requirement.
#    Lives in docker/compose.rerank-only.yaml. Pairs with `gliner-only` so a
#    CPU host can offer both /gliner and /rerank to docint/chorus.
#    Targets: build-rerank-only, up-rerank-only, stop-rerank-only,
#             bundle-rerank-only.
#
# 4. CLIP-only (same hosts as NER-only / Rerank-only) — a single
#    FastAPI/transformers CLIP image+text container on inference-net,
#    no router, no GPU requirement. Lives in docker/compose.clip-only.yaml.
#    Pairs with `gliner-only` and `rerank-only` so a CPU host can offer
#    /gliner, /rerank, and /clip/embed_{image,text} together.
#    Targets: build-clip-only, up-clip-only, stop-clip-only,
#             bundle-clip-only.

.DEFAULT_GOAL := help

.PHONY: help network volumes \
        build bundle up up-dev stop \
        build-gliner-only bundle-gliner-only up-gliner-only up-dev-gliner-only stop-gliner-only \
        build-rerank-only bundle-rerank-only up-rerank-only up-dev-rerank-only stop-rerank-only \
        build-clip-only bundle-clip-only up-clip-only up-dev-clip-only stop-clip-only

# Service-set profile for the full stack. Read from .env; empty = core stack.
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

COMPOSE                 := docker compose --env-file .env -f docker/compose.yaml
COMPOSE_DEV             := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml
COMPOSE_NER_ONLY        := docker compose --env-file .env -f docker/compose.gliner-only.yaml
COMPOSE_NER_ONLY_DEV    := docker compose --env-file .env -f docker/compose.gliner-only.yaml -f docker/compose.gliner-only.override.yaml
COMPOSE_RERANK_ONLY     := docker compose --env-file .env -f docker/compose.rerank-only.yaml
COMPOSE_RERANK_ONLY_DEV := docker compose --env-file .env -f docker/compose.rerank-only.yaml -f docker/compose.rerank-only.override.yaml
COMPOSE_CLIP_ONLY       := docker compose --env-file .env -f docker/compose.clip-only.yaml
COMPOSE_CLIP_ONLY_DEV   := docker compose --env-file .env -f docker/compose.clip-only.yaml -f docker/compose.clip-only.override.yaml
# Empty PROFILE -> no flag (core stack); PROFILE=media -> --profile media.
PROFILE_FLAG := $(if $(PROFILE),--profile $(PROFILE),)

help:
	@echo "vllm-service — build-host helpers."
	@echo
	@echo "Full stack (CUDA, active service set: $(if $(PROFILE),$(PROFILE),core)):"
	@echo "  make network          create the external inference-net"
	@echo "  make volumes          create the huggingface-cache Docker volume"
	@echo "  make build            build images for the active service set"
	@echo "  make bundle           ship images as a versioned .tar.gz pair"
	@echo "  make up               run the active service set (production shape, no host ports)"
	@echo "  make up-dev           like 'up', but publishes the router port on the host"
	@echo "  make stop             stop the active service set"
	@echo
	@echo "  Leave PROFILE empty in .env for the core stack; set PROFILE=media"
	@echo "  to add audio + translate. Override: make up PROFILE=media"
	@echo
	@echo "NER-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-gliner-only    build the gliner-cpu image"
	@echo "  make bundle-gliner-only   ship the gliner-cpu image as a versioned .tar.gz"
	@echo "  make up-gliner-only       run the GLiNER service on inference-net (no host port)"
	@echo "  make up-dev-gliner-only   like 'up-gliner-only', but publishes the port on the host"
	@echo "  make stop-gliner-only     stop the GLiNER service"
	@echo
	@echo "Rerank-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-rerank-only    build the rerank-cpu image"
	@echo "  make bundle-rerank-only   ship the rerank-cpu image as a versioned .tar.gz"
	@echo "  make up-rerank-only       run the rerank service on inference-net (no host port)"
	@echo "  make up-dev-rerank-only   like 'up-rerank-only', but publishes the port on the host"
	@echo "  make stop-rerank-only     stop the rerank service"
	@echo
	@echo "CLIP-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-clip-only      build the clip-cpu image"
	@echo "  make bundle-clip-only     ship the clip-cpu image as a versioned .tar.gz"
	@echo "  make up-clip-only         run the CLIP service on inference-net (no host port)"
	@echo "  make up-dev-clip-only     like 'up-clip-only', but publishes the port on the host"
	@echo "  make stop-clip-only       stop the CLIP service"

# --- Full stack ---------------------------------------------------------

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

# Run the active service set without rebuilding images (production shape, no host ports).
up:
	$(COMPOSE) $(PROFILE_FLAG) up --no-build

# Like 'up' but layers compose.override.yaml on top to publish the
# LiteLLM router port on the host.
up-dev:
	$(COMPOSE_DEV) $(PROFILE_FLAG) up --no-build

# Stop the active service set.
stop:
	$(COMPOSE) $(PROFILE_FLAG) stop

# --- NER-only stack -----------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites.

build-gliner-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_NER_ONLY) build

bundle-gliner-only:
	./scripts/bundle_images.sh gliner-only

up-gliner-only:
	$(COMPOSE_NER_ONLY) up --no-build

# Like 'up-gliner-only' but publishes the GLiNER port on the host.
up-dev-gliner-only:
	$(COMPOSE_NER_ONLY_DEV) up --no-build

stop-gliner-only:
	$(COMPOSE_NER_ONLY) stop

# --- Rerank-only stack --------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites.

build-rerank-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_RERANK_ONLY) build

bundle-rerank-only:
	./scripts/bundle_images.sh rerank-only

up-rerank-only:
	$(COMPOSE_RERANK_ONLY) up --no-build

# Like 'up-rerank-only' but publishes the rerank port on the host.
up-dev-rerank-only:
	$(COMPOSE_RERANK_ONLY_DEV) up --no-build

stop-rerank-only:
	$(COMPOSE_RERANK_ONLY) stop

# --- CLIP-only stack ----------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites.

build-clip-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_CLIP_ONLY) build

bundle-clip-only:
	./scripts/bundle_images.sh clip-only

up-clip-only:
	$(COMPOSE_CLIP_ONLY) up --no-build

# Like 'up-clip-only' but publishes the CLIP port on the host.
up-dev-clip-only:
	$(COMPOSE_CLIP_ONLY_DEV) up --no-build

stop-clip-only:
	$(COMPOSE_CLIP_ONLY) stop
