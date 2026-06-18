# common.mk - shared compose lifecycle targets for nos-tromo projects.
#
# Canonical source: nos-tromo/.github/configs/make-common/common.mk
# Vendored verbatim into each repo at make/common.mk; CI fails on drift via
# scripts/validate_make_common.py (the same canonical-config + drift-check
# pattern used for python-strict). Do not edit the vendored copy - change the
# canonical file and re-vendor.
#
# The including Makefile sets, BEFORE `include make/common.mk`:
#   REPO      := <slug>                 # e.g. translator, chorus, vllm-service
#   NETWORKS  := inference-net [data-net]
# Optional knobs (sensible defaults):
#   VOLUMES          :=                       # external volumes to ensure (empty = none)
#   COMPOSE_FILE     ?= docker/compose.yaml
#   COMPOSE_OVERRIDE ?= docker/compose.override.yaml
#   BUILD_ENV        ?= DOCKER_BUILDKIT=1     # env prefix for `build`
#   UP_ENV           ?=                       # env prefix for `up`/`up-dev` (e.g. DOCKER_BUILDKIT=1)
#   UP_FLAGS         ?=                       # extra flags for `up`/`up-dev` (e.g. --no-build, -d)
#   TESTS            ?= yes                    # set to `no` in repos with no pytest suite
# Each repo keeps its own `help` and any unique targets (migrate, nuke, the -only shapes).

ifndef REPO
$(error common.mk: set REPO before `include make/common.mk`)
endif

COMPOSE_FILE     ?= docker/compose.yaml
COMPOSE_OVERRIDE ?= docker/compose.override.yaml
BUILD_ENV        ?= DOCKER_BUILDKIT=1
UP_ENV           ?=
UP_FLAGS         ?=
TESTS            ?= yes

# Versioned image tag. Production: read .<repo>-version (written by bundle.sh).
# Dev: compute YYYY-MM-DD[-<short-sha>]. Override by exporting <REPO_UC>_VERSION.
# The $(eval) reproduces the original lazy `?=` definition under a per-repo
# variable name; the $$$$ escaping is what survives eval's single expansion.
REPO_UC     := $(shell printf '%s' '$(REPO)' | tr 'a-z-' 'A-Z_')
VERSION_VAR := $(REPO_UC)_VERSION
$(eval $(VERSION_VAR) ?= $$(shell cat .$(REPO)-version 2>/dev/null || { _s=$$$$(git rev-parse --short HEAD 2>/dev/null); echo "$$$$(date +%Y-%m-%d)$$$${_s:+-$$$$_s}"; }))
export $(VERSION_VAR)

COMPOSE     := docker compose --env-file .env -f $(COMPOSE_FILE)
COMPOSE_DEV := docker compose --env-file .env -f $(COMPOSE_FILE) -f $(COMPOSE_OVERRIDE)

.PHONY: network volumes build bundle up up-dev stop down logs pre-commit

# Create the external networks this repo joins (one-time per host; idempotent).
network:
	@for n in $(NETWORKS); do docker network create $$n >/dev/null 2>&1 || true; done

# Create the external volumes this repo owns (idempotent; no-op when VOLUMES empty).
volumes:
	@for v in $(VOLUMES); do \
		docker volume create $$v >/dev/null 2>&1 || true; \
		printf 'Ensured Docker volume exists: %s\n' "$$v"; \
	done

build:
	$(BUILD_ENV) $(COMPOSE) build

bundle:
	./scripts/bundle_images.sh

up:
	$(UP_ENV) $(COMPOSE) up $(UP_FLAGS)

up-dev:
	$(UP_ENV) $(COMPOSE_DEV) up $(UP_FLAGS)

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

pre-commit:
	uv run pre-commit run --all-files

ifeq ($(TESTS),yes)
.PHONY: test
test:
	uv run pytest -q
endif
