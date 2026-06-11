# Build-host helpers for the vLLM service stack.
#
# Two deployment shapes:
#
# 1. Full stack (CUDA host) — chat, embed, rerank, clip, audio, diarize,
#    gliner, router. Lives in docker/compose.yaml.
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
#
# 5. Diarize-only (same hosts as NER-only / Rerank-only / CLIP-only) — a
#    single FastAPI/pyannote speaker-diarization container on inference-net,
#    no router, no GPU requirement. Lives in docker/compose.diarize-only.yaml.
#    Pairs with the other three so a CPU host can offer /gliner, /rerank,
#    /clip/embed_{image,text}, and /diarize together.
#    Targets: build-diarize-only, up-diarize-only, stop-diarize-only,
#             bundle-diarize-only.

.DEFAULT_GOAL := help

.PHONY: help network volumes \
        build bundle up up-dev stop down \
        build-gliner-only bundle-gliner-only up-gliner-only up-dev-gliner-only stop-gliner-only down-gliner-only \
        build-rerank-only bundle-rerank-only up-rerank-only up-dev-rerank-only stop-rerank-only down-rerank-only \
        build-clip-only bundle-clip-only up-clip-only up-dev-clip-only stop-clip-only down-clip-only \
        build-diarize-only bundle-diarize-only up-diarize-only up-dev-diarize-only stop-diarize-only down-diarize-only

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
COMPOSE_DIARIZE_ONLY    := docker compose --env-file .env -f docker/compose.diarize-only.yaml
COMPOSE_DIARIZE_ONLY_DEV := docker compose --env-file .env -f docker/compose.diarize-only.yaml -f docker/compose.diarize-only.override.yaml

help:
	@echo "vllm-service — build-host helpers."
	@echo
	@echo "Full stack (CUDA):"
	@echo "  make network          create the external inference-net"
	@echo "  make volumes          create the huggingface-cache Docker volume"
	@echo "  make build            build images for the active service set"
	@echo "  make bundle           ship images as a versioned .tar.gz pair"
	@echo "  make up               run the active service set (production shape, no host ports)"
	@echo "  make up-dev           like 'up', but publishes the router port on the host"
	@echo "  make stop             stop the active service set"
	@echo "  make down             stop + remove the active service set"
	@echo
	@echo "NER-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-gliner-only    build the gliner-cpu image"
	@echo "  make bundle-gliner-only   ship the gliner-cpu image as a versioned .tar.gz"
	@echo "  make up-gliner-only       run the GLiNER service on inference-net (no host port)"
	@echo "  make up-dev-gliner-only   like 'up-gliner-only', but publishes the port on the host"
	@echo "  make stop-gliner-only     stop the GLiNER service"
	@echo "  make down-gliner-only     stop + remove the GLiNER service"
	@echo
	@echo "Rerank-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-rerank-only    build the rerank-only image"
	@echo "  make bundle-rerank-only   ship the rerank-only image as a versioned .tar.gz"
	@echo "  make up-rerank-only       run the rerank service on inference-net (no host port)"
	@echo "  make up-dev-rerank-only   like 'up-rerank-only', but publishes the port on the host"
	@echo "  make stop-rerank-only     stop the rerank service"
	@echo "  make down-rerank-only     stop + remove the rerank service"
	@echo
	@echo "CLIP-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-clip-only      build the clip-cpu image"
	@echo "  make bundle-clip-only     ship the clip-cpu image as a versioned .tar.gz"
	@echo "  make up-clip-only         run the CLIP service on inference-net (no host port)"
	@echo "  make up-dev-clip-only     like 'up-clip-only', but publishes the port on the host"
	@echo "  make stop-clip-only       stop the CLIP service"
	@echo "  make down-clip-only       stop + remove the CLIP service"
	@echo
	@echo "Diarize-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-diarize-only   build the diarize-cpu image"
	@echo "  make bundle-diarize-only  ship the diarize-cpu image as a versioned .tar.gz"
	@echo "  make up-diarize-only      run the diarization service on inference-net (no host port)"
	@echo "  make up-dev-diarize-only  like 'up-diarize-only', but publishes the port on the host"
	@echo "  make stop-diarize-only    stop the diarization service"
	@echo "  make down-diarize-only    stop + remove the diarization service"

# --- Full stack ---------------------------------------------------------

# Create the external Docker network (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true

# Create the external Hugging Face cache volume (one-time per host; idempotent).
volumes:
	docker volume create huggingface-cache >/dev/null 2>&1 || true

# Build images for the active service set.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) build

# Build images and ship as a versioned .tar.gz pair (built + pulled).
bundle:
	./scripts/bundle_images.sh

# Run the active service set without rebuilding images (production shape, no host ports).
up:
	$(COMPOSE) up --no-build

# Like 'up' but layers compose.override.yaml on top to publish the
# LiteLLM router port on the host.
up-dev:
	$(COMPOSE_DEV) up --no-build

# Stop the active service set.
stop:
	$(COMPOSE) stop

# Stop + remove the active service set. The huggingface-cache volume is
# external, so model weights survive — the next 'up' won't re-download.
down:
	$(COMPOSE) down

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

# Stop + remove the GLiNER service. External huggingface-cache survives.
down-gliner-only:
	$(COMPOSE_NER_ONLY) down

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

# Stop + remove the rerank service. External huggingface-cache survives.
down-rerank-only:
	$(COMPOSE_RERANK_ONLY) down

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

# Stop + remove the CLIP service. External huggingface-cache survives.
down-clip-only:
	$(COMPOSE_CLIP_ONLY) down

# --- Diarize-only stack -------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites. The pyannote weights are gated on Hugging Face — see the
# README "Diarize-only deployment" for the one-time pre-download.

build-diarize-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_DIARIZE_ONLY) build

bundle-diarize-only:
	./scripts/bundle_images.sh diarize-only

up-diarize-only:
	$(COMPOSE_DIARIZE_ONLY) up --no-build

# Like 'up-diarize-only' but publishes the diarization port on the host.
up-dev-diarize-only:
	$(COMPOSE_DIARIZE_ONLY_DEV) up --no-build

stop-diarize-only:
	$(COMPOSE_DIARIZE_ONLY) stop

# Stop + remove the diarization service. External huggingface-cache survives.
down-diarize-only:
	$(COMPOSE_DIARIZE_ONLY) down
