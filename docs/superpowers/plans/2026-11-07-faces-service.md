# Face detection and grouping service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve YuNet face detection and SFace-based per-video face grouping as
`POST /faces/detect` and `POST /faces/group` behind the router, CPU-only and
airgap-clean.

**Architecture:** A first-party FastAPI server (`src/faces_server.py`) loads both
ONNX models from the `huggingface-cache` volume at pinned revisions and runs
them through OpenCV; the clustering is pure NumPy in `src/faces_cluster.py`. The
router forwards both routes as JSON pass-throughs and records a per-route model
revision in its audit events.

**Tech Stack:** Python 3.11, FastAPI, opencv-python-headless, NumPy,
huggingface_hub, LiteLLM pass-through, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-07-faces-service-design.md`

## Global Constraints

- Python `>=3.11,<3.12`; ruff (line length 120, Google docstrings) and pyrefly strict, via `uv run pre-commit run --all-files`.
- Top-level JSON keys are exactly `images`, `min_score` (detect) and `tracks` (group); request models use `extra="ignore"`.
- Limits: ≤ 8 images per detect request; long side downscaled to ≤ 1280 px; ≤ 500 tracks and ≤ 3 faces per track per group request; body ≤ 32 MiB (`413`); model lock wait ≤ 30 s, else `503` with `Retry-After: 5`.
- Defaults: `min_score` 0.7; NMS IoU 0.4; outlier cut 0.3 cosine to the medoid; `FACE_GROUP_THRESHOLD` 0.42.
- Model variables: `FACE_DETECT_MODEL=opencv/face_detection_yunet`, `FACE_DETECT_MODEL_FILE=face_detection_yunet_2023mar.onnx`, `FACE_EMBED_MODEL=opencv/face_recognition_sface`, `FACE_EMBED_MODEL_FILE=face_recognition_sface_2021dec.onnx`, `FACE_DETECT_MODEL_REVISION`, `FACE_EMBED_MODEL_REVISION` (unset → no `revision` argument).
- No response, log line or exception message may carry an embedding, an image or a base64 string; logs carry counts only.
- Images run as uid 10001 under `x-hardened`; no runtime download (`HF_HUB_OFFLINE=1`).
- No real faces, photos or videos in the repo; tests stub `cv2` and use synthetic arrays.
- Version 0.8.1 → 0.9.0 in `pyproject.toml` and `uv.lock`.
- Pre-existing on `origin/main`: 6 failures in `eval/tests/test_diarize_server_gate.py` (torch stub without `cuda`); not this plan's to fix, and they must stay the only failures.

## Review Focus

- A detect batch where one image is corrupt: the others still return faces and the corrupt one returns `{"error": "undecodable"}` (Task 4, `test_detect_marks_undecodable_image_and_keeps_others`).
- Two people on screen together whose features are close: they must never get the same label (Task 2, `test_cannot_link_keeps_cooccurring_tracks_apart`).
- A track whose faces are mostly someone else after a missed cut: the outliers are dropped before averaging (Task 2, `test_outlier_faces_do_not_drag_the_track`).
- An oversized or flooding client: a 40 MiB body is a 413 before parsing and a held lock is a 503, not a queue (Task 4, `test_body_over_limit_is_413`, `test_busy_lock_is_503_with_retry_after`).
- An image larger than 1280 px or with a face over 300 px: it is downscaled and the half-scale pass still finds the face; coordinates stay normalized to the submitted image (Task 4, `test_two_scales_map_back_to_normalized_coordinates`).

---

### Task 0: Confirm the pinned router forwards the JSON shape

**Files:** none changed; findings go into the Task 5 commit message if anything differs.

- [ ] **Step 1: Read LiteLLM 1.97.0's pass-through kwarg filter from the pinned image**

Run:
```bash
docker run --rm --entrypoint python ghcr.io/berriai/litellm-non_root:v1.97.0@sha256:e2de74cae0d1f708bebe33bf3e03c3335029e27ff7f0dbddc7ff94d0edd63614 -c "import inspect, litellm.proxy.pass_through_endpoints.pass_through_endpoints as p; src = inspect.getsource(p); i = src.find('def _init_kwargs_for_pass_through_endpoint'); print(src[i:i+2500])"
```
Expected: the filter removes only LiteLLM parameter names; `images`, `min_score`, `tracks` are not among them, and the body is forwarded as JSON. If any of the three keys would be stripped, stop and report.

- [ ] **Step 2: Note the result** (no commit).

---

### Task 1: Per-route pass-through revisions in the router audit

**Files:**
- Modify: `docker/router/router_audit.py` (`PASS_THROUGH_REVISIONS` block near L65-75; `_passthrough_revision` near L866-877; its caller in `_facts` near L936-938)
- Modify: `eval/tests/test_revision_pinning.py` (`_router_pass_through_vars` L76-92; `test_router_records_the_variable_each_pass_through_backend_loads_by` L163-175)
- Modify: `eval/tests/test_router_audit.py` (`PINNED`/`LOADED_BY` L1174-1186; `test_pass_through_records_its_backends_revision` L1252-1266)
- Modify: `docs/audit.md` (the `ai.revision` row, L37)

**Interfaces:**
- Produces: `PASS_THROUGH_ROUTE_REVISIONS: Final[Mapping[str, str]] = {"/faces/detect": "FACE_DETECT_MODEL_REVISION", "/faces/group": "FACE_EMBED_MODEL_REVISION"}`; `_passthrough_revision(self, backend: str | None, route: str | None) -> str | None` — route map first, then host map.

- [ ] **Step 1: Write the failing tests**

In `test_router_audit.py`: extend `PINNED` with `"FACE_DETECT_MODEL_REVISION": "4" * 40, "FACE_EMBED_MODEL_REVISION": "5" * 40`; add `LOADED_BY_ROUTE = {"/faces/detect": "FACE_DETECT_MODEL_REVISION", "/faces/group": "FACE_EMBED_MODEL_REVISION"}`; in the parametrized test use `variable = LOADED_BY_ROUTE.get(route) or LOADED_BY.get(backend)`. Add:

```python
def test_route_revision_wins_over_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A route in the route map records its own variable, not the host's."""
    targets = {"/faces/detect": "faces", "/faces/group": "faces"}
    audit, stream = _audit(env=PINNED, targets=lambda: targets)
    _run(lambda: _pass_through_call(audit, "/faces/group"))
    (event,) = _ai_calls(stream)
    assert event["ai"]["revision"] == "5" * 40
```

In `test_revision_pinning.py`: generalize the AST reader to `_router_map(name: str) -> dict[str, str]` and assert `set(recorded) | {host for route, host in routes if route in route_map} == _pass_through_hosts() - UNPINNED_PASS_THROUGH`, plus:

```python
def test_route_revision_map_names_configured_routes() -> None:
    """Every key of PASS_THROUGH_ROUTE_REVISIONS is a configured pass-through path."""
    assert set(_router_map("PASS_THROUGH_ROUTE_REVISIONS")) <= _pass_through_paths()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --group eval pytest eval/tests/test_router_audit.py -k "revision" eval/tests/test_revision_pinning.py -q`
Expected: FAIL (`PASS_THROUGH_ROUTE_REVISIONS` not found / revision missing).

- [ ] **Step 3: Implement** the map and the two-argument `_passthrough_revision`; pass `call.route` at the `_facts` call site. Update the `PASS_THROUGH_REVISIONS` comment and `docs/audit.md` L37 (`/faces/detect` → `FACE_DETECT_MODEL_REVISION`, `/faces/group` → `FACE_EMBED_MODEL_REVISION`).

- [ ] **Step 4: Run the full suite**

Run: `uv run --group eval pytest eval/tests/ -q`
Expected: only the 6 pre-existing `test_diarize_server_gate.py` failures. (The `/faces/*` routes are not configured yet, so the subset assertion passes vacuously; Task 5 makes it bite.)

- [ ] **Step 5: Commit**

```bash
git add docker/router/router_audit.py eval/tests/test_router_audit.py eval/tests/test_revision_pinning.py docs/audit.md
git commit -m "feat(router): record a per-route model revision for pass-through routes"
```

---

### Task 2: Track clustering in pure NumPy

**Files:**
- Create: `src/faces_cluster.py`
- Test: `eval/tests/test_faces_cluster.py`

**Interfaces:**
- Produces: `track_vector(embeddings: np.ndarray, *, outlier_cos: float = 0.3) -> np.ndarray | None` (rows = faces of one track, L2-normalized; returns the normalized mean of the kept rows, `None` for zero rows); `cluster_tracks(vectors: list[np.ndarray | None], samples: list[frozenset[int]], *, threshold: float) -> list[int]`.

- [ ] **Step 1: Write the failing tests** (synthetic unit vectors, `numpy.random.default_rng(0)`)

```python
def test_similar_tracks_merge_and_labels_follow_first_track() -> None:
    a, b = _unit(0), _unit(1)  # two well-separated directions
    labels = cluster_tracks([a, _near(a), b, _near(b)], [frozenset({0}), frozenset({5}), frozenset({1}), frozenset({6})], threshold=0.42)
    assert labels == [0, 0, 1, 1]

def test_cannot_link_keeps_cooccurring_tracks_apart() -> None:
    a = _unit(0)
    labels = cluster_tracks([a, _near(a)], [frozenset({3, 4}), frozenset({4, 5})], threshold=0.42)
    assert labels == [0, 1]

def test_constraint_survives_merges() -> None:
    # t0~t2 merge first; t1 co-occurs with t2, so t1 may not join the merged cluster.
    a = _unit(0)
    labels = cluster_tracks([a, _near(a), _near(a)], [frozenset({0}), frozenset({9}), frozenset({9, 10})], threshold=0.42)
    assert labels[1] != labels[2]

def test_none_vector_is_unassigned() -> None:
    assert cluster_tracks([_unit(0), None], [frozenset({0}), frozenset({1})], threshold=0.42) == [0, -1]

def test_outlier_faces_do_not_drag_the_track() -> None:
    a, b = _unit(0), _unit(1)
    vec = track_vector(np.stack([a, _near(a), b]))
    assert vec is not None and float(vec @ a) > 0.95

def test_below_threshold_stays_split() -> None:
    assert cluster_tracks([_unit(0), _unit(1)], [frozenset({0}), frozenset({1})], threshold=0.42) == [0, 1]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --group eval pytest eval/tests/test_faces_cluster.py -q`
Expected: FAIL (`ModuleNotFoundError: faces_cluster`).

- [ ] **Step 3: Implement `src/faces_cluster.py`**

`track_vector`: medoid = row with the highest mean cosine to the others; keep rows with cosine ≥ `outlier_cos` to it (always the medoid); mean, renormalize. `cluster_tracks`, exact average linkage with constraints:

```python
# n = assigned tracks (vector not None); S = V @ V.T; C[i, j] = bool(samples[i] & samples[j])
# clusters start as singletons; sizes = 1
# loop: mask C and the diagonal to -inf; (i, j) = argmax(S); stop if S[i, j] < threshold
#   S[i, :] = (size_i * S[i, :] + size_j * S[j, :]) / (size_i + size_j); S[:, i] = S[i, :]
#   C[i, :] |= C[j, :]; C[:, i] = C[i, :]; drop row/col j (set to -inf / mark dead); size_i += size_j
# labels: number clusters by the smallest original track index they contain; unassigned → -1
```

- [ ] **Step 4: Run the tests**

Run: `uv run --group eval pytest eval/tests/test_faces_cluster.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/faces_cluster.py eval/tests/test_faces_cluster.py
git commit -m "feat(faces): average-linkage track clustering with cannot-link constraints"
```

---

### Task 3: The image's dependency closure

**Files:**
- Create: `docker/faces/pyproject.toml`, `docker/faces/uv.lock`
- Modify: `.github/dependabot.yml` (an `uv` entry for `/docker/faces`, like `/docker/topics`)

- [ ] **Step 1: Write `docker/faces/pyproject.toml`** — `name = "vllm-service-faces"`, `version = "0.0.0"`, `requires-python = ">=3.11,<3.12"`, `[tool.uv] package = false`, `environments = ["sys_platform == 'linux'", "sys_platform == 'darwin'"]`; dependencies: `opencv-python-headless>=4.10,<5`, `numpy>=2.0,<2.6`, `fastapi>=0.162.0`, `uvicorn[standard]>=0.54.0`, `pydantic>=2.0`, `huggingface-hub>=0.34`, `anyio>=4`.

- [ ] **Step 2: Lock it**

Run: `cd docker/faces && uv lock && cd -`
Expected: `docker/faces/uv.lock` written, no torch in it (`grep -c '^name = "torch"' docker/faces/uv.lock` → 0).

- [ ] **Step 3: Commit**

```bash
git add docker/faces/pyproject.toml docker/faces/uv.lock .github/dependabot.yml
git commit -m "build(faces): hash-locked dependency closure for the faces image"
```

---

### Task 4: The faces server

**Files:**
- Create: `src/faces_server.py`, `docker/Dockerfile.faces.cpu`
- Test: `eval/tests/test_faces_server.py`; modify `eval/tests/test_revision_pinning.py` (a faces loader test outside `SERVERS`)

**Interfaces:**
- Consumes: `faces_cluster.track_vector`, `faces_cluster.cluster_tracks` (Task 2).
- Produces: `app` with `POST /faces/detect`, `POST /faces/group`, `GET /health`, the wire contract in the spec; module-level `detector`, `recognizer` (cv2 objects) and `_model_lock: threading.Lock`.

- [ ] **Step 1: Write the failing tests**

Stub `cv2` and `huggingface_hub` in `sys.modules` before importing (pattern: `eval/tests/test_clip_server.py` `_install_ml_stubs`): `hf_hub_download(repo_id, filename, **kw)` records its call and returns a fake path; `cv2.FaceDetectorYN.create(...)` returns a fake whose `setInputSize` records sizes and whose `detect(img)` returns rows configured per test; `cv2.FaceRecognizerSF.create(...)` returns a fake with `alignCrop(img, row)` → `img` and `feature(img)` → a configured vector; `cv2.imdecode` decodes a tiny synthetic PNG via a registry keyed by the base64 string, `None` for `"broken"`; `cv2.resize` returns an array of the requested shape. Use `fastapi.testclient.TestClient`. Tests:

```python
def test_detect_returns_normalized_faces_in_order(client) -> None:
    body = {"images": [_img(640, 360), _img(640, 360)]}
    out = client.post("/faces/detect", json=body).json()
    assert len(out["results"]) == 2
    face = out["results"][0]["faces"][0]
    assert all(0.0 <= v <= 1.0 for v in face["box"]) and len(face["landmarks"]) == 5

def test_detect_marks_undecodable_image_and_keeps_others(client) -> None:
    out = client.post("/faces/detect", json={"images": ["broken", _img(320, 240)]}).json()
    assert out["results"][0] == {"error": "undecodable"} and "faces" in out["results"][1]

def test_detect_all_undecodable_is_422(client) -> None:
    assert client.post("/faces/detect", json={"images": ["broken"]}).status_code == 422

def test_two_scales_map_back_to_normalized_coordinates(client) -> None:
    # The fake detector reports the same face at full and half scale; NMS keeps one, coords agree.
    out = client.post("/faces/detect", json={"images": [_img(2560, 1440)]}).json()
    assert len(out["results"][0]["faces"]) == 1
    assert max(SET_INPUT_SIZES) <= (1280, 720)

def test_more_than_eight_images_is_400(client) -> None: ...
def test_body_over_limit_is_413(client) -> None:  # Content-Length 40 MiB → 413 before parsing
def test_busy_lock_is_503_with_retry_after(client, monkeypatch) -> None:  # lock held, wait patched to 0.01 s
def test_group_returns_labels_only(client) -> None:
    out = client.post("/faces/group", json={"tracks": [_track([0]), _track([1])]}).json()
    assert set(out) == {"labels"} and all(isinstance(x, int) for x in out["labels"])
def test_no_response_carries_a_vector_or_image(client) -> None:  # serialized bodies contain no float list longer than 10 and no base64 run
def test_unknown_top_level_keys_are_ignored(client) -> None:
def test_models_load_at_their_pinned_revisions(monkeypatch) -> None:  # FACE_*_REVISION set → each hf_hub_download got its own revision and file name
def test_models_load_without_revision_when_unset(monkeypatch) -> None:  # unset/empty → no "revision" kwarg
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --group eval pytest eval/tests/test_faces_server.py -q`
Expected: FAIL (`ModuleNotFoundError: faces_server`).

- [ ] **Step 3: Implement `src/faces_server.py`**

Module docstring carries the wire contract (as `vad_server.py` does). Load models at import through `hf_hub_download(repo_id, filename, **({"revision": rev} if rev else {}))`. Handlers are `async def` taking `Request`: check `Content-Length` (> 32 MiB → 413), read `request.stream()` with a running cap (→ 413), parse with `DetectRequest.model_validate_json` / `GroupRequest.model_validate_json` (`extra="ignore"`; count limits → 400), then run the CPU work with `anyio.to_thread.run_sync` under `_model_lock.acquire(timeout=30)` (→ 503, `Retry-After: 5`). Detection per image: decode (`np.frombuffer` + `cv2.imdecode`), downscale to ≤ 1280 px, `setInputSize((w, h))` + `detect` at 1× and at 0.5×, rescale the half-scale rows, merge by NMS at IoU 0.4 keeping the higher score, drop rows below `min_score`, normalize by the submitted image's size. Grouping: per face decode, build a YuNet-shaped row from the landmarks, `alignCrop` + `feature`, L2-normalize; `track_vector` per track; `cluster_tracks(..., threshold=FACE_GROUP_THRESHOLD)`. Log one INFO line per request with counts only.

- [ ] **Step 4: Write `docker/Dockerfile.faces.cpu`** — two stages like `docker/Dockerfile.topics.cpu`: builder installs `docker/faces/uv.lock` with `--locked`/`--require-hashes` into `/opt/venv`; runtime adds `curl`, copies the venv and `src/faces_cluster.py src/faces_server.py` to `/app`, creates uid 10001 with `/home/app/.cache/huggingface/hub` owned by it, sets `HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1`, `OPENCV_LOG_LEVEL=ERROR`, a build smoke `python -c "import cv2, faces_cluster; print(cv2.__version__)"`, and `CMD ["uvicorn", "--app-dir", "/app", "faces_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]`.

- [ ] **Step 5: Run the tests and the build**

Run: `uv run --group eval pytest eval/tests/test_faces_server.py eval/tests/test_revision_pinning.py -q` → PASS.
Run: `docker build -f docker/Dockerfile.faces.cpu -t vllm-service-faces-cpu:dev .` → succeeds, smoke prints the OpenCV version.

- [ ] **Step 6: Commit**

```bash
git add src/faces_server.py docker/Dockerfile.faces.cpu eval/tests/test_faces_server.py eval/tests/test_revision_pinning.py
git commit -m "feat(faces): YuNet detection and SFace grouping server"
```

---

### Task 5: Full-stack service and router routes

**Files:**
- Modify: `docker/compose.yaml` (`x-no-proxy-env` L3-5 add `faces`; new `faces` service after `vad`; router `depends_on` adds `faces: {condition: service_healthy}`)
- Modify: `docker/litellm.config.yaml` (two entries after `/vad`)

- [ ] **Step 1: Add the `faces` service** — `<<: *hardened`, `read_only: true`, image `vllm-service-faces:${VLLM_SERVICE_VERSION:-latest}`, build `docker/Dockerfile.faces.cpu`, `depends_on: <<: *volume-permissions-dep`, environment `<<: *no-proxy-env`, `HF_HUB_OFFLINE`, the six `FACE_*_MODEL*` variables with the spec's defaults and `FACE_GROUP_THRESHOLD: ${FACE_GROUP_THRESHOLD:-0.42}`, volume `huggingface-cache:/home/app/.cache/huggingface/hub`, `tmpfs: - /tmp:size=64m,uid=10001,gid=10001`, the `curl -sf http://localhost:8000/health` healthcheck (10s/5s/60/120s), networks `vllm-net` + `inference-net`, `restart: unless-stopped`. A comment block like `vad`'s: not vLLM, CPU only, the endpoint shapes.

- [ ] **Step 2: Add the routes**

```yaml
    # Face detection (YuNet) and per-video grouping (SFace). JSON bodies with
    # base64 images; headers pin JSON. Embeddings never leave the backend.
    - path: "/faces/detect"
      target: "http://faces:8000/faces/detect"
      auth: true
      headers:
        content-type: "application/json"
    - path: "/faces/group"
      target: "http://faces:8000/faces/group"
      auth: true
      headers:
        content-type: "application/json"
```

- [ ] **Step 3: Validate**

Run: `docker compose --env-file /dev/null -f docker/compose.yaml config -q && uv run --group eval pytest eval/tests/test_revision_pinning.py eval/tests/test_router_audit.py eval/tests/test_configs.py -q`
Expected: config valid; tests PASS (the route-map subset test now covers both configured routes).

- [ ] **Step 4: Commit**

```bash
git add docker/compose.yaml docker/litellm.config.yaml
git commit -m "feat(stack): run the faces service and route /faces/* through the router"
```

---

### Task 6: The `faces-only` CPU shape

**Files:**
- Create: `docker/compose.faces-only.yaml`, `docker/compose.faces-only.override.yaml`
- Modify: `Makefile` (header comment's shape list, `.PHONY`, `COMPOSE_FACES_ONLY{,_DEV}`, targets `build-/bundle-/up-/up-dev-/stop-/down-faces-only`, help text), `scripts/bundle_images.sh` (`faces-only` arm and usage line)

- [ ] **Step 1: Write the shape** by copying `docker/compose.vad-only.yaml`'s structure: project `vllm-service-faces-only`, `volume-permissions` on image `vllm-service-faces-cpu`, service `faces-only` (alias `faces-only`, the `faces` service's environment, tmpfs `/tmp`), external `huggingface-cache`. Override publishes `${FACES_HOST_PORT:-8009}:8000`.

- [ ] **Step 2: Wire Makefile and bundle script** following the `vad-only` targets line by line.

- [ ] **Step 3: Validate**

Run: `docker compose --env-file /dev/null -f docker/compose.faces-only.yaml -f docker/compose.faces-only.override.yaml config -q && make -n bundle-faces-only | head -3`
Expected: valid; the bundle command names `faces-only`.

- [ ] **Step 4: Smoke the shape** — `make build-faces-only up-dev-faces-only` with the two models fetched into the cache, then `curl -s localhost:8009/health` → `{"status":"ok",...}` and a detect call on a synthetic image (a drawn shape, no face) → `{"results":[{"faces":[]}]}`. `make down-faces-only`.

- [ ] **Step 5: Commit**

```bash
git add docker/compose.faces-only.yaml docker/compose.faces-only.override.yaml Makefile scripts/bundle_images.sh
git commit -m "feat(shape): faces-only CPU shape"
```

---

### Task 7: Documentation

**Files:** `docs/api-reference.md` (new `/faces/detect` and `/faces/group` sections after `/vad`), `docs/deployment-shapes.md` ("all eight" → nine; the shape table), `docs/configuration.md` (the variables; the model-revisions table L246-266), `docs/airgap-bundles.md`, `docs/architecture.md`, `docs/audit.md` (recipient list L34, pass-through routes L67, knobs L159), `README.md`, `CLAUDE.md` (the service in the bespoke-server list; `faces` in the startup chain before `gliner`), `.env.example` (the six model variables and `FACE_GROUP_THRESHOLD`, commented).

- [ ] **Step 1: Write the docs**, including a short licence note: YuNet MIT, SFace Apache-2.0 (both OpenCV Zoo, `LICENSE` in each HF repo); InsightFace not used because its weights are non-commercial; fetch only the `.onnx` file and `LICENSE` of each repo (the repos also hold `example_outputs/` with photos of people).

- [ ] **Step 2: Check** `uv run pre-commit run --all-files` → all hooks pass.

- [ ] **Step 3: Commit**

```bash
git add docs README.md CLAUDE.md .env.example
git commit -m "docs: the faces service, its contract, shape and licences"
```

---

### Task 8: Release

- [ ] **Step 1:** `pyproject.toml` `version = "0.9.0"`; refresh `uv.lock` (`uv lock`), only the version line changes.
- [ ] **Step 2:** `uv run pre-commit run --all-files` and `uv run --group eval pytest eval/tests/ -q` → only the 6 pre-existing failures.
- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: release v0.9.0"
```

---

## Addendum: two-stage grouping

After the end-to-end runs, `faces_cluster.py` gained the join stage the
design's addendum describes: `group_tracks(..., threshold, join_threshold,
anchors)` returns the labels with the piece counts, `cluster_tracks` its
labels; `MIN_PERSON_SAMPLES = 2`, `JOIN_TOP_K = 3`, `JOIN_MARGIN = 0.10`.
`faces_server.py` reads `FACE_JOIN_THRESHOLD` (default 0.25, refused at
startup outside `[0, FACE_GROUP_THRESHOLD]`) and both compose files pass it.
Tests: `eval/tests/test_faces_cluster.py` (join between the thresholds, below
it, without a clear lead, top-three closeness, established people kept apart,
a person seen once takes no one, cannot-link against joined pieces, no
chaining, contiguous labels, the counts) and `eval/tests/test_faces_server.py`
(the variable, its bounds, that it reaches the grouping, the log line).

