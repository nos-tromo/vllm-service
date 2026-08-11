# Build-host helpers for the vLLM service stack.
#
# Two deployment shapes:
#
# 1. Full stack (CUDA host) — chat, embed, rerank, clip, asr, diarize, vad,
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
#
# 6. ASR-only (same hosts as NER-only / Rerank-only / CLIP-only /
#    Diarize-only) — a single FastAPI/openai-whisper container on
#    inference-net, no router, no GPU requirement. Lives in
#    docker/compose.asr-only.yaml. The CPU counterpart to the full-stack vLLM
#    `asr` service; speaks the same OpenAI /v1/audio/transcriptions contract.
#    Targets: build-asr-only, up-asr-only, stop-asr-only, bundle-asr-only.
#
# 7. VAD-only (same hosts as the other -only shapes) — a single
#    FastAPI/Silero voice-activity-detection container on inference-net, no
#    router, no GPU requirement. Lives in docker/compose.vad-only.yaml.
#    Same multipart /vad contract as the full-stack `vad` service.
#    Targets: build-vad-only, up-vad-only, stop-vad-only, bundle-vad-only.
#
# 8. Embed-only (same hosts as the other -only shapes) — a single
#    FastAPI/transformers bge-m3 container on inference-net, no router, no
#    GPU requirement. Lives in docker/compose.embed-only.yaml. Serves
#    dense embeddings (/v1/embeddings), sparse weights (/pooling), and
#    tokenization (/tokenize) — the same routes the full stack's router
#    passes through — from one loaded model, so docint only has to
#    repoint its embedding base and SPARSE_API_BASE.
#    Targets: build-embed-only, up-embed-only, stop-embed-only,
#             bundle-embed-only.

.DEFAULT_GOAL := help

.PHONY: help \
        build-gliner-only bundle-gliner-only up-gliner-only up-dev-gliner-only stop-gliner-only down-gliner-only \
        build-rerank-only bundle-rerank-only up-rerank-only up-dev-rerank-only stop-rerank-only down-rerank-only \
        build-clip-only bundle-clip-only up-clip-only up-dev-clip-only stop-clip-only down-clip-only \
        build-diarize-only bundle-diarize-only up-diarize-only up-dev-diarize-only stop-diarize-only down-diarize-only \
        build-asr-only bundle-asr-only up-asr-only up-dev-asr-only stop-asr-only down-asr-only \
        build-vad-only bundle-vad-only up-vad-only up-dev-vad-only stop-vad-only down-vad-only \
        build-embed-only bundle-embed-only up-embed-only up-dev-embed-only stop-embed-only down-embed-only

# The full-stack compose lifecycle (network/volumes/build/bundle/up/up-dev/
# stop/down/pre-commit) + the versioned image tag come from make/common.mk,
# vendored from nos-tromo/.github. The seven CPU `-only` shapes below are
# vllm-service-specific and stay here.
REPO     := vllm-service
NETWORKS := inference-net
VOLUMES  := huggingface-cache
TESTS    := no
include make/common.mk

COMPOSE_NER_ONLY        := docker compose --env-file .env -f docker/compose.gliner-only.yaml
COMPOSE_NER_ONLY_DEV    := docker compose --env-file .env -f docker/compose.gliner-only.yaml -f docker/compose.gliner-only.override.yaml
COMPOSE_RERANK_ONLY     := docker compose --env-file .env -f docker/compose.rerank-only.yaml
COMPOSE_RERANK_ONLY_DEV := docker compose --env-file .env -f docker/compose.rerank-only.yaml -f docker/compose.rerank-only.override.yaml
COMPOSE_CLIP_ONLY       := docker compose --env-file .env -f docker/compose.clip-only.yaml
COMPOSE_CLIP_ONLY_DEV   := docker compose --env-file .env -f docker/compose.clip-only.yaml -f docker/compose.clip-only.override.yaml
COMPOSE_DIARIZE_ONLY    := docker compose --env-file .env -f docker/compose.diarize-only.yaml
COMPOSE_DIARIZE_ONLY_DEV := docker compose --env-file .env -f docker/compose.diarize-only.yaml -f docker/compose.diarize-only.override.yaml
COMPOSE_ASR_ONLY        := docker compose --env-file .env -f docker/compose.asr-only.yaml
COMPOSE_ASR_ONLY_DEV    := docker compose --env-file .env -f docker/compose.asr-only.yaml -f docker/compose.asr-only.override.yaml
COMPOSE_VAD_ONLY        := docker compose --env-file .env -f docker/compose.vad-only.yaml
COMPOSE_VAD_ONLY_DEV    := docker compose --env-file .env -f docker/compose.vad-only.yaml -f docker/compose.vad-only.override.yaml
COMPOSE_EMBED_ONLY      := docker compose --env-file .env -f docker/compose.embed-only.yaml
COMPOSE_EMBED_ONLY_DEV  := docker compose --env-file .env -f docker/compose.embed-only.yaml -f docker/compose.embed-only.override.yaml

help:
	@echo "vllm-service — build-host helpers."
	@echo
	@echo "Development:"
	@echo "  make pre-commit       run ruff check + ruff format + pyrefly over src/ (no Docker)"
	@echo "  make verify           pre-push gate: pre-commit (ruff + pyrefly); mirrors CI's lint gate"
	@echo
	@echo "Full stack (CUDA):"
	@echo "  make network          create the external inference-net"
	@echo "  make volumes          create the huggingface-cache Docker volume"
	@echo "  make build            build images for the active service set"
	@echo "  make bundle           ship images as a versioned .tar.gz pair (latest annotated release tag)"
	@echo "  make bundle-dev       like 'bundle', but from the current working tree (dev/soak)"
	@echo "  make up               run the active service set (detached, no build; production shape, no host ports)"
	@echo "  make up-dev           like 'up' (detached, no build), but publishes the router port on the host"
	@echo "  make dev              build the active service set, then up-dev"
	@echo "  make stop             stop the active service set"
	@echo "  make down             stop + remove the active service set"
	@echo
	@echo "NER-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-gliner-only    build the gliner-cpu image"
	@echo "  make bundle-gliner-only   ship the gliner-cpu image as a versioned .tar.gz"
	@echo "  make up-gliner-only       run the GLiNER service on inference-net (detached, no host port)"
	@echo "  make up-dev-gliner-only   like 'up-gliner-only', but publishes the port on the host"
	@echo "  make stop-gliner-only     stop the GLiNER service"
	@echo "  make down-gliner-only     stop + remove the GLiNER service"
	@echo
	@echo "Rerank-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-rerank-only    build the rerank-only image"
	@echo "  make bundle-rerank-only   ship the rerank-only image as a versioned .tar.gz"
	@echo "  make up-rerank-only       run the rerank service on inference-net (detached, no host port)"
	@echo "  make up-dev-rerank-only   like 'up-rerank-only', but publishes the port on the host"
	@echo "  make stop-rerank-only     stop the rerank service"
	@echo "  make down-rerank-only     stop + remove the rerank service"
	@echo
	@echo "CLIP-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-clip-only      build the clip-cpu image"
	@echo "  make bundle-clip-only     ship the clip-cpu image as a versioned .tar.gz"
	@echo "  make up-clip-only         run the CLIP service on inference-net (detached, no host port)"
	@echo "  make up-dev-clip-only     like 'up-clip-only', but publishes the port on the host"
	@echo "  make stop-clip-only       stop the CLIP service"
	@echo "  make down-clip-only       stop + remove the CLIP service"
	@echo
	@echo "Diarize-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-diarize-only   build the diarize-cpu image"
	@echo "  make bundle-diarize-only  ship the diarize-cpu image as a versioned .tar.gz"
	@echo "  make up-diarize-only      run the diarization service on inference-net (detached, no host port)"
	@echo "  make up-dev-diarize-only  like 'up-diarize-only', but publishes the port on the host"
	@echo "  make stop-diarize-only    stop the diarization service"
	@echo "  make down-diarize-only    stop + remove the diarization service"
	@echo
	@echo "ASR-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-asr-only       build the asr-cpu image"
	@echo "  make bundle-asr-only      ship the asr-cpu image as a versioned .tar.gz"
	@echo "  make up-asr-only          run the ASR service on inference-net (detached, no host port)"
	@echo "  make up-dev-asr-only      like 'up-asr-only', but publishes the port on the host"
	@echo "  make stop-asr-only        stop the ASR service"
	@echo "  make down-asr-only        stop + remove the ASR service"
	@echo
	@echo "VAD-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-vad-only       build the vad-cpu image"
	@echo "  make bundle-vad-only      ship the vad-cpu image as a versioned .tar.gz"
	@echo "  make up-vad-only          run the VAD service on inference-net (detached, no host port)"
	@echo "  make up-dev-vad-only      like 'up-vad-only', but publishes the port on the host"
	@echo "  make stop-vad-only        stop the VAD service"
	@echo "  make down-vad-only        stop + remove the VAD service"
	@echo
	@echo "Embed-only stack (CPU; pairs with Ollama on non-CUDA hosts):"
	@echo "  make build-embed-only     build the embed-cpu image"
	@echo "  make bundle-embed-only    ship the embed-cpu image as a versioned .tar.gz"
	@echo "  make up-embed-only        run the dense+sparse embedding service on inference-net (detached, no host port)"
	@echo "  make up-dev-embed-only    like 'up-embed-only', but publishes the port on the host"
	@echo "  make stop-embed-only      stop the dense+sparse embedding service"
	@echo "  make down-embed-only      stop + remove the dense+sparse embedding service"


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
	$(COMPOSE_NER_ONLY) up -d --no-build

# Like 'up-gliner-only' but publishes the GLiNER port on the host.
up-dev-gliner-only:
	$(COMPOSE_NER_ONLY_DEV) up -d --no-build

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
	$(COMPOSE_RERANK_ONLY) up -d --no-build

# Like 'up-rerank-only' but publishes the rerank port on the host.
up-dev-rerank-only:
	$(COMPOSE_RERANK_ONLY_DEV) up -d --no-build

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
	$(COMPOSE_CLIP_ONLY) up -d --no-build

# Like 'up-clip-only' but publishes the CLIP port on the host.
up-dev-clip-only:
	$(COMPOSE_CLIP_ONLY_DEV) up -d --no-build

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
	$(COMPOSE_DIARIZE_ONLY) up -d --no-build

# Like 'up-diarize-only' but publishes the diarization port on the host.
up-dev-diarize-only:
	$(COMPOSE_DIARIZE_ONLY_DEV) up -d --no-build

stop-diarize-only:
	$(COMPOSE_DIARIZE_ONLY) stop

# Stop + remove the diarization service. External huggingface-cache survives.
down-diarize-only:
	$(COMPOSE_DIARIZE_ONLY) down

# --- ASR-only stack -----------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites. CPU openai-whisper — the Whisper weights are public, so the
# cache populates anonymously on first start (or pre-seed it offline).

build-asr-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_ASR_ONLY) build

bundle-asr-only:
	./scripts/bundle_images.sh asr-only

up-asr-only:
	$(COMPOSE_ASR_ONLY) up -d --no-build

# Like 'up-asr-only' but publishes the ASR port on the host.
up-dev-asr-only:
	$(COMPOSE_ASR_ONLY_DEV) up -d --no-build

stop-asr-only:
	$(COMPOSE_ASR_ONLY) stop

# Stop + remove the ASR service. External huggingface-cache survives.
down-asr-only:
	$(COMPOSE_ASR_ONLY) down

# --- VAD-only stack -----------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites. The silero-vad package bundles its weights, so this shape
# needs no model download at all.

build-vad-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_VAD_ONLY) build

bundle-vad-only:
	./scripts/bundle_images.sh vad-only

up-vad-only:
	$(COMPOSE_VAD_ONLY) up -d --no-build

# Like 'up-vad-only' but publishes the VAD port on the host.
up-dev-vad-only:
	$(COMPOSE_VAD_ONLY_DEV) up -d --no-build

stop-vad-only:
	$(COMPOSE_VAD_ONLY) stop

# Stop + remove the VAD service. External huggingface-cache survives.
down-vad-only:
	$(COMPOSE_VAD_ONLY) down

# --- Embed-only stack --------------------------------------------------
#
# Uses the same external inference-net + huggingface-cache as the full
# stack, so `make network` and `make volumes` remain the one-time
# prerequisites. Serves bge-m3 dense embeddings (/v1/embeddings), sparse
# weights (/pooling), and tokenization (/tokenize) — the same routes the
# full stack's router passes through — from one loaded model, so docint
# only has to repoint its embedding base and SPARSE_API_BASE.
build-embed-only:
	DOCKER_BUILDKIT=1 $(COMPOSE_EMBED_ONLY) build

bundle-embed-only:
	./scripts/bundle_images.sh embed-only

up-embed-only:
	$(COMPOSE_EMBED_ONLY) up -d --no-build

# Like 'up-embed-only' but publishes the embed port on the host.
up-dev-embed-only:
	$(COMPOSE_EMBED_ONLY_DEV) up -d --no-build

stop-embed-only:
	$(COMPOSE_EMBED_ONLY) stop

# Stop + remove the embed service. External huggingface-cache survives.
down-embed-only:
	$(COMPOSE_EMBED_ONLY) down

