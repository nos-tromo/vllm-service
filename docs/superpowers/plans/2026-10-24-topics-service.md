# Topics Fit Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a stateless CPU service that fits a BERTopic model to caller-supplied vectors and returns per-input topic labels plus per-topic keywords.

**Architecture:**
- `src/topics_fit.py` holds the deterministic fit, importing its heavy libraries lazily.
- `src/topics_server.py` holds the HTTP contract around it, so the contract can be tested with the fit stubbed.
- The image installs a separately locked closure (`docker/topics/`). One CPU image serves the full stack (behind a router pass-through) and a new `topics-only` shape.

**Tech Stack:** Python 3.11, FastAPI, BERTopic 0.17.4, umap-learn 0.5.12, hdbscan 0.8.44, scikit-learn, numpy, uv, Docker Compose, LiteLLM pass-through.

**Spec:** `docs/superpowers/specs/2026-10-24-topics-service-design.md`

## Global Constraints

**Pins**
- Python `>=3.11,<3.12`.
- Exact pins: `bertopic==0.17.4`, `umap-learn==0.5.12`, `hdbscan==0.8.44`.
- Ranges: `numpy>=2.0,<2.6`, `pandas>=2.2,<3`.

**What the image may contain**
- No `torch`, no `sentence-transformers`, no `plotly`.
- `pyyaml` and `packaging` are listed explicitly.

**Behaviour of the service**
- Deterministic: the same input bytes, settings and image give the same labels and keywords. No unseeded randomness.
- Stateless. It never logs texts, vectors or keywords; logs carry counts and timings only.

**Runtime environment**
- Thread variables `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` are all `1`.
- `NUMBA_CACHE_DIR=/tmp/numba-cache`, `JOBLIB_TEMP_FOLDER=/tmp`.

**Container hardening and naming**
- Hardening follows deploy ADR 0001: non-root uid 10001, `read_only: true`, `cap_drop: ALL`, `no-new-privileges`.
- Image tags: `vllm-service-topics` (full stack) and `vllm-service-topics-cpu` (`topics-only`). Host port in dev: `8008`. Alias: `topics-only`.

**Router**
- The pass-through has `auth: true` written out and no `headers` block.
- The router's `depends_on` uses `condition: service_started` for `topics`.

**Wording and data**
- Name things only "topics" / "topic-model fit".
- Use synthetic fixtures only.
- Use no absolute local paths in any committed file.

## Review Focus

Each line below is pinned by a test added to the task that owns the code.

1. **Degenerate vectors** (many identical rows, or a single dense blob). Expect a 200 with every label valid (`-1` or a topic), never a 500. → Task 1, `test_identical_vectors_do_not_crash`.
2. **Empty or whitespace texts for some, or all, points.** CountVectorizer's "empty vocabulary" must give topics with empty keywords, not a 500. → Task 1, `test_empty_texts_yield_empty_keywords`.
3. **Too few points for UMAP or HDBSCAN.** Expect a 422 naming the minimum, not a crash inside UMAP. → Task 1 `test_min_points` and Task 2 `test_too_few_points_is_422`.
4. **A second request during a fit.** Expect an immediate 503 with `Retry-After`, never a queued multi-hundred-MB upload. → Task 2, `test_busy_returns_503`.
5. **NaN or Inf in vectors, or a byte count that doesn't match `n·dim·4`.** Expect a 422 before any fit work. → Task 2, `test_non_finite_vectors_are_422` and `test_vector_size_mismatch_is_422`.

---

### Task 1: Locked topics project and deterministic fit core

**Files:**
- Create:
  - `docker/topics/pyproject.toml`
  - `docker/topics/uv.lock` (generated)
  - `src/topics_fit.py`
- Test: `eval/tests/test_topics_fit.py`

**Interfaces:**
- Produces, in `src/topics_fit.py`:
  - `FitParams` (frozen dataclass): `seed: int`, `min_topic_size: int`, `nr_topics: int`, `top_n_words: int`, `min_df: int`, `stop_words: tuple[str, ...]`.
  - `TopicInfo` (frozen dataclass): `id: int`, `size: int`, `keywords: list[tuple[str, float]]`.
  - `FitResult` (frozen dataclass): `labels: list[int]`, `topics: list[TopicInfo]`, `outliers: int`, `fit_s: float`.
  - `min_points(params: FitParams) -> int`, returning `max(16, 2 * params.min_topic_size)`.
  - `fit_topics(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult`. It raises `ValueError` when `len(texts) != len(vectors)` or `len(vectors) < min_points(params)`.
  - `versions() -> dict[str, str]`, using importlib.metadata for bertopic, umap-learn, hdbscan, scikit-learn, numpy and numba; a missing package reads `"absent"`.
  - `runtime() -> dict[str, object]`, holding `platform.machine()`, the python version, and the four thread variables.
  - `self_check() -> None`, which fits the synthetic set twice and raises `RuntimeError` if the labels or keywords differ.

- [ ] **Step 1: Create `docker/topics/pyproject.toml`**

```toml
[project]
name = "vllm-service-topics"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "bertopic==0.17.4",
    "umap-learn==0.5.12",
    "hdbscan==0.8.44",
    "numpy>=2.0,<2.6",
    "pandas>=2.2,<3",
    "scikit-learn>=1.6",
    "pyyaml>=6",
    "packaging>=24",
    "fastapi>=0.141.1",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.9",
    "pydantic>=2.0",
]

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.uv]
package = false
environments = ["sys_platform == 'linux'", "sys_platform == 'darwin'"]
# BERTopic hard-requires both, but imports neither without an embedding model or plots.
exclude-dependencies = ["sentence-transformers", "plotly"]
```

- [ ] **Step 2: Lock, and prove the exclusions held**

Run:

```bash
cd docker/topics
uv lock
grep -c 'name = "torch"' uv.lock
grep -c 'name = "sentence-transformers"' uv.lock
grep -c 'name = "plotly"' uv.lock
head -2 uv.lock
```

Expected:
- each `grep -c` prints `0`;
- the second line of `uv.lock` is `revision = 5`, the same revision as the root lock.
  - If it isn't, re-lock with the uv version that writes revision 5, and note which one.

- [ ] **Step 3: Write the failing tests in `eval/tests/test_topics_fit.py`**

Module setup:
- Skip the module unless `bertopic` is importable: `pytest.importorskip("bertopic")`. That way it runs only in the topics env.
- Insert `src/` on `sys.path`, as `test_clip_server.py` does.
- Fixture `blobs(n_per=60, dim=32, k=3, seed=7)`:
  - Gaussian blobs, L2-normalized, as float32.
  - Texts drawn from a per-blob synthetic vocabulary (`"alpha0 alpha1 …"`, `"beta0 …"`, `"gamma0 …"`).
- Tests:
  - `test_fit_is_deterministic`: two `fit_topics` calls on the same input give equal `labels` and equal `topics`.
  - `test_recovers_planted_blobs`: with `min_topic_size=10`, at least 2 topics are found, and ≥ 80% of each blob shares one label.
  - `test_keywords_come_from_texts`: each topic's top keyword starts with that blob's vocabulary prefix.
  - `test_min_points`:
    - `min_points(FitParams(..., min_topic_size=10))` is `20`;
    - fitting 15 points raises `ValueError` matching `"at least 20"`.
  - `test_identical_vectors_do_not_crash`: 80 identical rows. Every label is in `{-1} ∪ topic ids`, and no exception is raised.
  - `test_empty_texts_yield_empty_keywords`: all texts are `""`. The fit returns, and every topic's `keywords` is `[]`.
  - `test_no_network_during_fit`: patch `socket.socket.connect` to raise, then fit succeeds.
  - `test_no_torch_or_sentence_transformers`: `importlib.util.find_spec` is `None` for `torch`, `sentence_transformers` and `plotly`.
  - `test_self_check_passes`: `self_check()` returns `None`.

- [ ] **Step 4: Run them and confirm they fail**

```bash
uv run --project docker/topics --group dev pytest eval/tests/test_topics_fit.py -q
```

Expected: FAIL, `ModuleNotFoundError: topics_fit`.

- [ ] **Step 5: Implement `src/topics_fit.py`**

Import discipline:
- Import heavy modules (`bertopic`, `umap`, `hdbscan`, `sklearn`) **inside** `fit_topics`.
- At module level import only stdlib and numpy, so `topics_server` and its tests import without them.

Model configuration — exactly the spec:

```python
UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=params.seed)
hdbscan.HDBSCAN(
    min_cluster_size=params.min_topic_size,
    metric="euclidean",
    cluster_selection_method="eom",
    core_dist_n_jobs=1,
    prediction_data=False,
)
CountVectorizer(stop_words=list(params.stop_words) or None, min_df=params.min_df)
ClassTfidfTransformer(reduce_frequent_words=True)
BERTopic(
    embedding_model=None,
    nr_topics=params.nr_topics or None,
    top_n_words=params.top_n_words,
    calculate_probabilities=False,
    verbose=False,
)
```

Text handling — the empty-vocabulary fallback:
1. Before fitting, run `CountVectorizer(stop_words=…, min_df=…).fit(texts)`.
2. If that raises `ValueError` (empty vocabulary), fit instead with:
   - every text replaced by the sentinel `__topics_empty__`;
   - `stop_words=None`, `min_df=1`.
3. Drop the sentinel from every keyword list, so `keywords` comes back `[]`.

This leaves the clustering untouched and stays deterministic, which catching BERTopic's error mid-fit would not. Keyword scores are rounded to 6 decimals, so JSON round-trips are stable.

`self_check`:
- The dataset is fixed (`blobs` equivalent, `seed=7`), so it doesn't depend on the test file.

- [ ] **Step 6: Run them and confirm they pass**

```bash
uv run --project docker/topics --group dev pytest eval/tests/test_topics_fit.py -q
```

Expected: all PASS.

- [ ] **Step 7: Record timings (not committed)**

Time a fit on synthetic 1024-d blobs at 10k, 50k and 100k points with `min_topic_size=max(10, n//1000)`.

```bash
uv run --project docker/topics python -c '<inline timing script>'
```

Note the three wall times and the peak RSS (`/usr/bin/time -l` on macOS) for the PR body.

- [ ] **Step 8: Run the gates and commit**

```bash
uv run --locked pre-commit run --all-files
git add docker/topics/pyproject.toml docker/topics/uv.lock src/topics_fit.py eval/tests/test_topics_fit.py
git commit -m "feat(topics): deterministic topic-model fit core with its own locked closure"
```

---

### Task 2: HTTP contract

**Files:**
- Create: `src/topics_server.py`
- Test: `eval/tests/test_topics_server.py`

**Interfaces:**
- Consumes: from Task 1, `topics_fit.FitParams`, `FitResult`, `TopicInfo`, `fit_topics`, `min_points`, `versions`, `runtime` and `self_check`.
- Produces:
  - `app` (FastAPI, `title="vllm-service topics"`).
  - `POST /topics/fit` with form file fields `meta`, `texts` and `vectors`.
  - `GET /health`.
- Environment:
  - `TOPICS_MAX_POINTS`: default `250000`, read once into the module constant `MAX_POINTS: int`. The tests monkeypatch `topics_server.MAX_POINTS`. Fail fast at startup if it isn't a positive int, as in `embed_server._positive_int_env`.
  - `TOPICS_WARMUP`: default `true`; when true, call `topics_fit.self_check()` at import.

- [ ] **Step 1: Write the failing tests in `eval/tests/test_topics_server.py`**

Setup:
- Set `os.environ["TOPICS_WARMUP"] = "false"` before importing `topics_server`, and insert `src/` on the path.
- Monkeypatch `topics_server.topics_fit.fit_topics` with a fake that returns `FitResult(labels=[0]*n, topics=[TopicInfo(0, n, [("w", 1.0)])], outliers=0, fit_s=0.01)`.
- A helper `_post(meta, texts, vectors_bytes)` builds the multipart request with file tuples (`("meta.json", …, "application/json")`, and so on).

Tests:
- `test_fit_returns_contract`: n=40, dim=8.
  - The status is 200.
  - The keys are `labels`, `outliers`, `topics`, `versions`, `runtime` and `timings`.
  - `len(labels) == 40`.
  - `topics[0] == {"id": 0, "size": 40, "keywords": [["w", 1.0]]}`.
- `test_malformed_meta_is_400`: `meta` is not JSON → 400; `meta` is missing `"dim"` → 400.
- `test_too_many_points_is_413`: monkeypatch `topics_server.MAX_POINTS = 10`, n=11 → 413. The fake fit is never called.
- `test_vector_size_mismatch_is_422`: n·dim·4 − 4 bytes → 422.
- `test_non_finite_vectors_are_422`: one `NaN` → 422; one `inf` → 422.
- `test_text_count_mismatch_is_422`: n−1 texts → 422; texts not a list of str → 422.
- `test_too_few_points_is_422`: n = `min_points(params) - 1` → 422, and the detail contains the minimum.
- `test_missing_part_is_422`: no `vectors` part → 422.
- `test_busy_returns_503`:
  - Acquire `topics_server._fit_lock` in the test.
  - POST → 503 with the header `Retry-After: 30`.
  - Release the lock.
- `test_fit_failure_is_500`: the fake raises `RuntimeError` → 500, and the response body contains no text from the request.
- `test_health_reports_versions`: 200 and `{"status": "ok", "versions": {...}}`.

- [ ] **Step 2: Run them and confirm they fail**

```bash
uv run --locked pytest eval/tests/test_topics_server.py -q
```

Expected: FAIL, `ModuleNotFoundError: topics_server`.

- [ ] **Step 3: Implement `src/topics_server.py`**

Follow `src/vad_server.py`'s shape: module docstring with the contract, a `def` endpoint, and a module `_fit_lock = threading.Lock()`.

Order inside `POST /topics/fit`:
1. `_fit_lock.acquire(blocking=False)`; if that fails → 503 with the `Retry-After: 30` header.
2. Read and parse `meta` with a pydantic `FitMeta`. The bounds are:
   - `n >= 1`, `dim >= 1`
   - `1 <= top_n_words <= 30`
   - `min_topic_size >= 2`
   - `min_df >= 1`
   - `nr_topics >= 0`
   - `stop_words: list[str]` with at most 10000 entries
   - Errors → 400.
3. `n > MAX_POINTS` → 413.
4. Read `vectors`. If the length isn't `n*dim*4` → 422. Otherwise `np.frombuffer(…, "<f4").reshape(n, dim)`. If not finite everywhere → 422.
5. Read `texts`. It must be a JSON list of exactly `n` strings, else 422.
6. `n < min_points(params)` → 422, naming the minimum.
7. Call `fit_topics`. Any exception → 500 with a fixed message.
8. Release the lock in `finally`.

Logging: one INFO line, `topics fit n=%d dim=%d topics=%d outliers=%d fit_s=%.2f`. Never log texts, keywords or vectors.

- [ ] **Step 4: Run them and confirm they pass**

```bash
uv run --locked pytest eval/tests/test_topics_server.py -q
```

Expected: all PASS.

- [ ] **Step 5: Run the gates and commit**

```bash
uv run --locked pre-commit run --all-files
git add src/topics_server.py eval/tests/test_topics_server.py
git commit -m "feat(topics): POST /topics/fit contract with busy, size and input guards"
```

---

### Task 3: Image and dependency updates

**Files:**
- Create: `docker/Dockerfile.topics.cpu`
- Modify: `.github/dependabot.yml`, adding a uv entry for `/docker/topics`

**Interfaces:**
- Consumes: Tasks 1–2 (`docker/topics/uv.lock`, `src/topics_fit.py`, `src/topics_server.py`).
- Produces: an image whose `CMD` serves `topics_server:app` on port 8000 as uid 10001, with `curl` available for the health check.

- [ ] **Step 1: Write `docker/Dockerfile.topics.cpu`**

Use two stages on the pinned base `ghcr.io/astral-sh/uv:python3.11-trixie-slim@sha256:0f2d0e6d745eb7f9cc9bc3a8f3bec70f398b39f38c8a1c7fa2967ddbe91af8bd`. Both stages share it, so the interpreter path matches.

Builder stage:
1. Install `build-essential`.
2. `COPY docker/topics/pyproject.toml docker/topics/uv.lock /build/`.
3. `uv export --frozen --no-dev --no-emit-project --format requirements-txt -o /build/requirements.txt`.
4. `uv venv /opt/venv`.
5. `uv pip install --python /opt/venv/bin/python --no-deps --require-hashes --compile-bytecode -r /build/requirements.txt`.

Runtime stage:
1. Install `curl`.
2. `COPY --from=builder /opt/venv /opt/venv`.
3. `COPY src/topics_fit.py src/topics_server.py /app/`.
4. `ENV PATH=/opt/venv/bin:$PATH` plus the thread, numba and joblib variables from Global Constraints.
5. The build smoke: `RUN cd /app && NUMBA_CACHE_DIR=/tmp/build-numba python -c "import topics_fit; topics_fit.self_check(); print('topics build smoke: OK')"`.
6. Create the app user exactly as `Dockerfile.vad.cpu` does, minus the HF cache lines.
7. `USER 10001:10001`, `EXPOSE 8000`, `CMD ["uvicorn","--app-dir","/app","topics_server:app","--host","0.0.0.0","--port","8000"]`.

- [ ] **Step 2: Build it**

Docker Desktop must be running.

```bash
docker build -f docker/Dockerfile.topics.cpu -t vllm-service-topics-cpu:dev .
```

Expected:
- the build succeeds on linux/arm64, compiling hdbscan;
- the log contains `topics build smoke: OK`.

- [ ] **Step 3: Check determinism across CPU quotas**

Run the same synthetic fit (2000 points, dim 64, seed 7) under two quotas, printing the sha256 of `json.dumps(result.labels)`:

```bash
docker run --rm --cpus=2 vllm-service-topics-cpu:dev python -c '<print sha256 of labels>'
docker run --rm --cpus=8 vllm-service-topics-cpu:dev python -c '<print sha256 of labels>'
```

Expected: identical hashes.
- If they differ, set `NUMBA_CPU_NAME=generic` in the image and repeat.
- Record which setting held.

- [ ] **Step 4: Check the image's contents**

```bash
docker run --rm vllm-service-topics-cpu:dev python -c "import importlib.util as u; print([m for m in ('torch','sentence_transformers','plotly') if u.find_spec(m)])"
docker image ls vllm-service-topics-cpu:dev --format '{{.Size}}'
```

Expected: `[]`, and the size is noted for the PR body.

- [ ] **Step 5: Add the Dependabot entry**

Add a `uv` ecosystem entry for directory `/docker/topics`:
- weekly, with `cooldown.default-days: 7`;
- `ignore` semver-major and semver-minor updates for `bertopic`, `umap-learn`, `hdbscan`, `numba`, `llvmlite`, `pynndescent`, `numpy`, `scipy`, `scikit-learn` and `pandas`;
- a two-line comment: those bumps can change fit results, so they move deliberately. Security updates still open.

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile.topics.cpu .github/dependabot.yml
git commit -m "build(topics): CPU image from the locked closure with a determinism smoke"
```

---

### Task 4: Compose, router, `topics-only` shape and bundling

**Files:**
- Modify: `docker/compose.yaml`, `docker/litellm.config.yaml`, `scripts/bundle_images.sh`, `Makefile`, `.env.example`
- Create: `docker/compose.topics-only.yaml`, `docker/compose.topics-only.override.yaml`

**Interfaces:**
- Consumes: Task 3's image (`docker/Dockerfile.topics.cpu`).
- Produces:
  - router path `POST http://vllm-router:4000/topics/fit` (Bearer master key);
  - direct path `POST http://topics-only:8000/topics/fit` (no auth);
  - `make build-topics-only|bundle-topics-only|up-topics-only|up-dev-topics-only|stop-topics-only|down-topics-only`.

- [ ] **Step 1: Check the router for a path collision**

```bash
gh api -X GET search/code -f q='"/topics" repo:BerriAI/litellm path:litellm/proxy' --jq '.items[].path'
```

Expected: no FastAPI route registered at `/topics`.
- If there is one, use `/topic-model/fit` everywhere (contract, router, docs), and update the spec first.

- [ ] **Step 2: Add the full-stack service to `docker/compose.yaml`**

The `topics` service:
- `<<: *hardened`, `read_only: true`;
- image `vllm-service-topics:${VLLM_SERVICE_VERSION:-latest}`, built from `docker/Dockerfile.topics.cpu`;
- `logging: *default-logging`, `env_file: *env-file`;
- environment `<<: *no-proxy-env` and `TOPICS_MAX_POINTS: ${TOPICS_MAX_POINTS:-250000}`;
- volumes `media-tmp:/tmp` only;
- the healthcheck as in `vad`, with `start_period: 180s` for the warm-up;
- networks `vllm-net` and `inference-net`.

Also:
- Add `topics` to both strings in `x-no-proxy-env`.
- In `router.depends_on`, add `topics: {condition: service_started}` with a one-line comment: "A failed topics self-check must not keep inference down."

- [ ] **Step 3: Add the pass-through to `docker/litellm.config.yaml`**

Add it after the `/vad` entry, with a comment block in the style of the `/diarize` one:

```yaml
    - path: "/topics/fit"
      target: "http://topics:8000/topics/fit"
      auth: true
```

- [ ] **Step 4: Write `docker/compose.topics-only.yaml` and its override**

Base the file on `compose.vad-only.yaml`:
- Drop the `volume-permissions` service and the `huggingface-cache` volume; there are no weights.
- Service `topics-only` with image `vllm-service-topics-cpu`.
- Network alias `topics-only`.
- Keep the `media-tmp` volume.

The override publishes `"${TOPICS_HOST_PORT:-8008}:8000"`.

- [ ] **Step 5: Update bundling, the Makefile and `.env.example`**

- `scripts/bundle_images.sh`: add a `topics-only)` case and update the usage line.
- `Makefile`:
  - the header shape block;
  - the `.PHONY` line;
  - `COMPOSE_TOPICS_ONLY` and `COMPOSE_TOPICS_ONLY_DEV`;
  - the help lines;
  - the six targets, mirroring `*-vad-only`;
  - "seven CPU `-only` shapes" → "eight".
- `.env.example`: add `TOPICS_MAX_POINTS` and `TOPICS_HOST_PORT`, commented, with defaults.

- [ ] **Step 6: Validate the compose files and bring the shape up**

```bash
docker compose --env-file .env.example -f docker/compose.yaml config -q && echo OK
docker compose --env-file .env.example -f docker/compose.topics-only.yaml config -q && echo OK
make build-topics-only && make up-dev-topics-only
```

After about 3 minutes:

```bash
curl -sf localhost:8008/health
```

Expected:
- both `config` commands print `OK`;
- health returns `{"status":"ok","versions":{...}}`.

- [ ] **Step 7: Smoke-test against the live shape**

POST a synthetic 300-point multipart request to `localhost:8008/topics/fit` (a small Python `httpx` script, not committed).

Expected:
- a 200 whose labels have length 300;
- repeated twice, byte-identical JSON `labels` and `topics`.

Then `make down-topics-only`.

- [ ] **Step 8: Commit**

```bash
git add docker/compose.yaml docker/litellm.config.yaml docker/compose.topics-only.yaml docker/compose.topics-only.override.yaml scripts/bundle_images.sh Makefile .env.example
git commit -m "feat(topics): wire the service into the stack, the router and a topics-only shape"
```

---

### Task 5: Docs and version

**Files:**
- Modify: `docs/architecture.md`, `docs/api-reference.md`, `docs/deployment-shapes.md`, `docs/configuration.md`, `docs/airgap-bundles.md`, `docs/README.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`

**Interfaces:**
- Consumes: the contract from the spec, and the targets from Task 4.

- [ ] **Step 1: Update the reference docs**

- `docs/api-reference.md`:
  - a pass-through table row;
  - a `/topics/fit` section with both addresses, the parts table, the response, and the error table from the spec.
- `docs/deployment-shapes.md`:
  - "seven" → "eight";
  - both tables;
  - a `## Topics-only` section mirroring `## VAD-only`.
- `docs/architecture.md`: the service table and the container counts.
- `docs/configuration.md`: `TOPICS_MAX_POINTS` and `TOPICS_WARMUP`.
- `docs/airgap-bundles.md`: the target list and the count.
- `docs/README.md` and `README.md`: the service and shape lists.

- [ ] **Step 2: Update CLAUDE.md**

- The service list and the shape counts.
- One clause under the dependency rules: the topics image is the exception that installs from its own lock (`docker/topics/uv.lock`), because a fixed-seed fit needs exact pins.

- [ ] **Step 3: Bump the version**

In `pyproject.toml`, change `version = "0.5.10"` → `"0.6.0"`.

- [ ] **Step 4: Run the gates and commit**

```bash
uv run --locked pre-commit run --all-files
uv run --locked pytest eval/tests/test_topics_server.py -q
uv run --project docker/topics --group dev pytest eval/tests/test_topics_fit.py -q
git add docs README.md CLAUDE.md pyproject.toml
git commit -m "docs(topics): document the fit service and the topics-only shape; 0.6.0"
```

Expected: all green.

The six `test_diarize_server_gate.py` failures are pre-existing stub drift, outside this change. Leave them as they are.
