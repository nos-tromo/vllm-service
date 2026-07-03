# common.mk - shared compose lifecycle targets for nos-tromo projects.
#
# Canonical source: nos-tromo/.github/configs/make-common/common.mk
# Vendored verbatim into each repo at make/common.mk; CI fails on drift via
# scripts/validate_make_common.py (the same canonical-config + drift-check
# pattern used for python-strict). Do not edit the vendored copy - change the
# canonical file and re-vendor.
#
# `up` / `up-dev` are detached and never build: they run `up -d --no-build`
# (production shape), matching the bespoke pulled-image members (data-plane,
# open-webui-service). Build first, then bring up: `make build && make up-dev`
# in dev (or just `make dev`); load/pull images before `make up` in prod.
# `bundle` builds the latest reachable annotated tag (production); `bundle-dev`
# bundles the current working tree (dev/soak). See scripts/bundle-lib.sh.
#
# The including Makefile sets, BEFORE `include make/common.mk`:
#   REPO      := <slug>                 # e.g. translator, chorus, vllm-service
#   NETWORKS  := inference-net [data-net]
# Optional knobs (sensible defaults):
#   VOLUMES          :=                       # external volumes to ensure (empty = none)
#   COMPOSE_FILE     ?= docker/compose.yaml
#   COMPOSE_OVERRIDE ?= docker/compose.override.yaml
#   BUILD_ENV        ?= DOCKER_BUILDKIT=1     # env prefix for `build`
#   TESTS            ?= yes                    # set to `no` in repos with no pytest suite
# Each repo keeps its own `help` and any unique targets (migrate, nuke, the -only shapes).

ifndef REPO
$(error common.mk: set REPO before `include make/common.mk`)
endif

COMPOSE_FILE     ?= docker/compose.yaml
COMPOSE_OVERRIDE ?= docker/compose.override.yaml
BUILD_ENV        ?= DOCKER_BUILDKIT=1
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

.PHONY: network volumes build bundle bundle-dev up up-dev dev stop down logs pre-commit

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

# Airgap release artifact. `bundle` is PRODUCTION: it builds the latest annotated
# tag reachable from HEAD (checks it out, builds, restores your branch) and
# refuses on a dirty tree or when no tag is reachable. `bundle-dev` bundles the
# current working tree as-is (date+sha / override) for dev iteration and soak.
bundle:
	./scripts/bundle_images.sh

bundle-dev:
	BUNDLE_DEV=1 ./scripts/bundle_images.sh

# Detached, no build, production shape. `--no-build` errors if the image is
# absent, so build first (`make build` in dev) or load/pull it (in prod).
up:
	$(COMPOSE) up -d --no-build

# Like `up` but layers the dev override (publishes host ports). Still detached
# and no-build - run `make build` first, or use `make dev`.
up-dev:
	$(COMPOSE_DEV) up -d --no-build

# Dev convenience: build the image(s), then bring up with host ports.
dev: build up-dev

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
