# gliner Self-Heal Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gliner service recover unattended from a wedged Ray Serve deployment (the Ray 2.55.1 rank-consistency bug) by supervising `gliner.serve` with an in-container watchdog that self-exits on a failed NER probe, backed by a real restart policy and a functional healthcheck.

**Architecture:** A POSIX-sh script (`gliner-watchdog`) becomes the container's entrypoint wrapper: it launches `python -m gliner.serve` as a child, waits out a startup grace period, then probes `POST /gliner` on a fixed interval. After N consecutive failures it kills the child and exits non-zero, so Docker's `restart: unless-stopped` recreates the container with a fresh embedded Ray (which clears the wedge — verified during the incident). The Docker healthcheck switches from the shallow `/-/healthz` to the same functional probe so `docker ps` reflects real NER serving.

**Tech Stack:** POSIX shell, `curl` (already in both gliner images), Docker Compose, Ray Serve / `gliner.serve` (upstream, unchanged). Test harness: Bash + Python stdlib `http.server` (dev/CI only).

## Global Constraints

- **`scripts/gliner_watchdog.sh` MUST be POSIX `sh`** — it runs as the container's `/bin/sh` entrypoint (Debian `dash` on the CUDA image, bookworm on the CPU image). No bashisms.
- **No new Python module.** The repo's `src/*.py` are pyrefly-strict + ruff-linted; the watchdog is shell precisely to avoid that surface. (The *test* harness may be Bash + Python, since it runs on dev/CI, not in the image.)
- **Probe is fixed:** `POST http://localhost:8000/gliner` with body `{"text":"ping","labels":["person"],"threshold":0.5}`, success = HTTP 2xx (`curl -sf`). gliner always runs `--port 8000`.
- **Default knobs:** `NER_WATCHDOG_ENABLED=true`, `GRACE_S=180`, `INTERVAL_S=30`, `FAILURES=3`, `TIMEOUT_S=10`, `WEBHOOK_URL` unset.
- **Both compose shapes stay in sync:** `docker/compose.yaml` (full CUDA stack) and `docker/compose.gliner-only.yaml` (CPU dev shape) get the same command + healthcheck changes.
- **Do not touch** the existing `gliner[serve]` pin or the build-time `gliner.serve.server` bind-host patch in either Dockerfile.
- **Docker build context is the repo root** (`build.context: ..`), so Dockerfile `COPY` paths are repo-root-relative (e.g. `scripts/gliner_watchdog.sh`).
- **All commits end with the trailer:**
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work happens on branch `feat/gliner-self-heal-watchdog` (already created; the design spec is committed there).

---

### Task 1: Watchdog script + standalone test harness

**Files:**
- Create: `scripts/gliner_watchdog.sh`
- Create (test): `scripts/test_gliner_watchdog.sh`

**Interfaces:**
- Produces: an executable supervisor invoked as `gliner-watchdog <cmd> [args…]`. It runs `<cmd> [args…]` as a child and manages its lifecycle. Exit codes: `0` on SIGTERM/SIGINT (clean shutdown); non-zero when it decides the child must be restarted or the child died on its own.
- Consumes (env): `NER_WATCHDOG_ENABLED|GRACE_S|INTERVAL_S|FAILURES|TIMEOUT_S|WEBHOOK_URL` (see Global Constraints).

- [ ] **Step 1: Write the failing test harness**

Create `scripts/test_gliner_watchdog.sh`:

```bash
#!/usr/bin/env bash
# Standalone tests for scripts/gliner_watchdog.sh. No GPU / no Ray:
# a stdlib HTTP stub stands in for gliner.serve; its health is toggled via a
# flag file. Uses condition-polling (not fixed sleeps) to stay deterministic.
# Run: bash scripts/test_gliner_watchdog.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
WATCHDOG="$HERE/gliner_watchdog.sh"
STUB="$(mktemp /tmp/gliner_stub.XXXXXX.py)"
FLAG="$(mktemp -u /tmp/gliner_unhealthy.XXXXXX)"
PASS=0; FAIL=0

cat >"$STUB" <<'PY'
import http.server, os
FLAG = os.environ["STUB_UNHEALTHY_FLAG"]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        self.rfile.read(n)
        self.send_response(503 if os.path.exists(FLAG) else 200)
        self.end_headers()
        self.wfile.write(b'{"entities":[]}')
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", 8000), H).serve_forever()
PY

port_free() { ! curl -sf -m1 -X POST -d '{}' http://localhost:8000/gliner >/dev/null 2>&1; }

reset() {           # ensure no stub is bound to :8000 and health flag cleared
  rm -f "$FLAG"
  for _ in $(seq 1 20); do port_free && return 0; sleep 0.5; done
  return 0
}

wait_healthy() {    # poll the stub directly until it answers 2xx, up to ~10s
  for _ in $(seq 1 20); do
    curl -sf -m1 -X POST -H 'content-type: application/json' \
      -d '{"text":"x","labels":["person"],"threshold":0.5}' \
      http://localhost:8000/gliner >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  return 1
}

check() {           # $1=name $2=expected $3=actual
  if [ "$2" = "$3" ]; then echo "PASS: $1 (exit $3)"; PASS=$((PASS+1));
  else echo "FAIL: $1 (expected $2, got $3)"; FAIL=$((FAIL+1)); fi
}

export STUB_UNHEALTHY_FLAG="$FLAG"
export NER_WATCHDOG_GRACE_S=2 NER_WATCHDOG_INTERVAL_S=1 \
       NER_WATCHDOG_FAILURES=2 NER_WATCHDOG_TIMEOUT_S=2

# A: healthy, then SIGTERM -> clean exit 0
reset; sh "$WATCHDOG" python3 "$STUB" & WD=$!
wait_healthy || { echo "FAIL: A setup (stub never healthy)"; FAIL=$((FAIL+1)); }
sleep 2                                   # let the watchdog arm
kill -TERM "$WD"; wait "$WD"; check "SIGTERM -> exit 0" 0 "$?"

# B: healthy then unhealthy -> exit 1 (steady-state restart)
reset; sh "$WATCHDOG" python3 "$STUB" & WD=$!
wait_healthy || { echo "FAIL: B setup"; FAIL=$((FAIL+1)); }
sleep 2; touch "$FLAG"                     # flip stub to 503 after arming
wait "$WD"; check "unhealthy -> exit 1" 1 "$?"

# C: child dies on its own -> watchdog exits non-zero (propagates)
reset; sh "$WATCHDOG" python3 "$STUB" & WD=$!
wait_healthy || { echo "FAIL: C setup"; FAIL=$((FAIL+1)); }
sleep 2; pkill -P "$WD" -f "$STUB"         # kill ONLY the supervised child
wait "$WD"; C=$?
if [ "$C" -ne 0 ]; then echo "PASS: child death -> exit $C"; PASS=$((PASS+1));
else echo "FAIL: child death -> expected non-zero, got 0"; FAIL=$((FAIL+1)); fi

# D: never healthy -> fail-fast exit 1 after 2x grace
reset; touch "$FLAG"                       # stub answers 503 from the start
sh "$WATCHDOG" python3 "$STUB" & WD=$!
wait "$WD"; check "never ready -> exit 1" 1 "$?"

reset; rm -f "$STUB"
echo "----"; echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash scripts/test_gliner_watchdog.sh; echo "rc=$?"`
Expected: FAIL — every scenario errors because `scripts/gliner_watchdog.sh` does not exist yet (`sh: …/gliner_watchdog.sh: No such file or directory`), final line `PASS=0 FAIL=4`, `rc=1`.

- [ ] **Step 3: Write the watchdog**

Create `scripts/gliner_watchdog.sh`:

```sh
#!/bin/sh
# gliner-watchdog — supervise `gliner.serve` and self-exit when the Ray Serve
# deployment stops serving NER, so Docker's restart policy recreates the
# container (a fresh embedded Ray clears the rank-consistency wedge,
# ray-project/ray#63862). POSIX sh only; runs as the container entrypoint.
# Design: docs/superpowers/specs/2026-06-25-gliner-self-heal-watchdog-design.md
set -u

# Escape hatch: behave exactly like the pre-watchdog entrypoint.
if [ "${NER_WATCHDOG_ENABLED:-true}" != "true" ]; then
  exec "$@"
fi

GRACE_S="${NER_WATCHDOG_GRACE_S:-180}"
INTERVAL_S="${NER_WATCHDOG_INTERVAL_S:-30}"
MAX_FAILURES="${NER_WATCHDOG_FAILURES:-3}"
TIMEOUT_S="${NER_WATCHDOG_TIMEOUT_S:-10}"
WEBHOOK_URL="${NER_WATCHDOG_WEBHOOK_URL:-}"
PROBE_URL="http://localhost:8000/gliner"
SLEEP_PID=""

log() {   # $1=message  $2=level (default info)
  printf '{"level":"%s","svc":"gliner-watchdog","msg":"%s"}\n' "${2:-info}" "$1" >&2
}

alert() { # $1=message
  log "$1" error
  if [ -n "$WEBHOOK_URL" ]; then
    curl -sf -m "$TIMEOUT_S" -X POST -H 'content-type: application/json' \
      -d "{\"text\":\"gliner watchdog: $1\"}" "$WEBHOOK_URL" >/dev/null 2>&1 || true
  fi
}

probe() {
  curl -sf -m "$TIMEOUT_S" -X POST -H 'content-type: application/json' \
    -d '{"text":"ping","labels":["person"],"threshold":0.5}' \
    "$PROBE_URL" >/dev/null 2>&1
}

# Interruptible sleep: backgrounded so a trapped signal fires immediately.
# A foreground `sleep` would defer the trap until it elapsed -> SIGKILL/137.
nap() {
  sleep "$1" &
  SLEEP_PID=$!
  wait "$SLEEP_PID" 2>/dev/null || true
  SLEEP_PID=""
}

# Launch the supervised process (the passed argv: python -m gliner.serve ...).
"$@" &
CHILD_PID=$!

shutdown() {
  trap '' TERM INT
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null || true
  kill -TERM "$CHILD_PID" 2>/dev/null || true
  wait "$CHILD_PID" 2>/dev/null || true
  exit 0
}
trap shutdown TERM INT

# --- Readiness: wait for the first successful probe, bounded by 2x grace. ---
deadline=$(( GRACE_S * 2 ))
waited=0
ready=false
while : ; do
  if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    wait "$CHILD_PID"; exit $?      # child exited on its own during startup
  fi
  if probe; then ready=true; break; fi
  [ "$waited" -ge "$deadline" ] && break
  nap "$INTERVAL_S"
  waited=$(( waited + INTERVAL_S ))
done

if [ "$ready" != "true" ]; then
  alert "gliner did not become ready within ${deadline}s; exiting for restart"
  kill -TERM "$CHILD_PID" 2>/dev/null || true
  wait "$CHILD_PID" 2>/dev/null || true
  exit 1
fi
log "gliner ready; watchdog armed (interval=${INTERVAL_S}s failures=${MAX_FAILURES})"

# --- Steady state: probe periodically; restart after N consecutive failures. ---
failures=0
while kill -0 "$CHILD_PID" 2>/dev/null; do
  nap "$INTERVAL_S"
  if probe; then
    failures=0
  else
    failures=$(( failures + 1 ))
    log "probe failed (${failures}/${MAX_FAILURES})" warning
    if [ "$failures" -ge "$MAX_FAILURES" ]; then
      alert "gliner unresponsive after ${MAX_FAILURES} consecutive probes; exiting for restart"
      kill -TERM "$CHILD_PID" 2>/dev/null || true
      wait "$CHILD_PID" 2>/dev/null || true
      exit 1
    fi
  fi
done

# Child exited on its own during steady state; propagate its exit status.
wait "$CHILD_PID"
exit $?
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `chmod +x scripts/gliner_watchdog.sh && bash scripts/test_gliner_watchdog.sh; echo "rc=$?"`
Expected: four `PASS:` lines, final `PASS=4 FAIL=0`, `rc=0`.

- [ ] **Step 5: Lint the script**

Run: `sh -n scripts/gliner_watchdog.sh && { command -v shellcheck >/dev/null && shellcheck -s sh scripts/gliner_watchdog.sh || echo "shellcheck not installed; sh -n ok"; }`
Expected: no syntax errors; shellcheck clean if present.

- [ ] **Step 6: Commit**

```bash
git add scripts/gliner_watchdog.sh scripts/test_gliner_watchdog.sh
git commit -m "feat(gliner): in-container self-heal watchdog + standalone tests"
```

---

### Task 2: Bake the watchdog into both gliner images

**Files:**
- Modify: `docker/Dockerfile.gliner.cuda` (insert before `ENV TOKENIZERS_PARALLELISM=true`, currently line 31)
- Modify: `docker/Dockerfile.gliner.cpu` (insert before `ENV TOKENIZERS_PARALLELISM=true`, currently line 41)

**Interfaces:**
- Consumes: `scripts/gliner_watchdog.sh` from Task 1.
- Produces: `/usr/local/bin/gliner-watchdog` (executable) present in both images.

- [ ] **Step 1: Add the COPY + build-time syntax smoke test to the CUDA image**

In `docker/Dockerfile.gliner.cuda`, immediately before `ENV TOKENIZERS_PARALLELISM=true`, insert:

```dockerfile
# In-container self-heal watchdog: supervises gliner.serve and self-exits on a
# failed NER probe so the container restarts (clears the Ray Serve rank wedge,
# ray-project/ray#63862). See docs/superpowers/specs/2026-06-25-gliner-self-heal-watchdog-design.md
COPY scripts/gliner_watchdog.sh /usr/local/bin/gliner-watchdog
RUN chmod +x /usr/local/bin/gliner-watchdog && sh -n /usr/local/bin/gliner-watchdog
```

- [ ] **Step 2: Add the identical block to the CPU image**

In `docker/Dockerfile.gliner.cpu`, immediately before `ENV TOKENIZERS_PARALLELISM=true`, insert the **same** four lines as Step 1 (verbatim).

- [ ] **Step 3: Build both images to verify the COPY + syntax check pass**

Run:
```bash
docker build -f docker/Dockerfile.gliner.cuda -t wd-test-cuda .
docker build -f docker/Dockerfile.gliner.cpu  -t wd-test-cpu  .
```
Expected: both builds succeed; the `RUN … sh -n …` layer passes (a syntax error in the script would fail the build here).

- [ ] **Step 4: Verify the script is present and executable in each image**

Run:
```bash
docker run --rm --entrypoint sh wd-test-cuda -c 'test -x /usr/local/bin/gliner-watchdog && echo CUDA-OK'
docker run --rm --entrypoint sh wd-test-cpu  -c 'test -x /usr/local/bin/gliner-watchdog && echo CPU-OK'
```
Expected: `CUDA-OK` and `CPU-OK`.

- [ ] **Step 5: Commit**

```bash
git add docker/Dockerfile.gliner.cuda docker/Dockerfile.gliner.cpu
git commit -m "build(gliner): install self-heal watchdog into both gliner images"
```

---

### Task 3: Wire watchdog + deep healthcheck + restart into the full CUDA stack

**Files:**
- Modify: `docker/compose.yaml` (the `gliner:` service)

**Interfaces:**
- Consumes: `/usr/local/bin/gliner-watchdog` (Task 2).
- Produces: a gliner service that self-restarts on a wedge and reports true health.

- [ ] **Step 1: Wrap the entrypoint command with the watchdog**

In `docker/compose.yaml`, in the `gliner:` service `command:`, change the final line from:

```yaml
        exec "$@"
```
to:
```yaml
        exec /usr/local/bin/gliner-watchdog "$@"
```

- [ ] **Step 2: Replace the shallow healthcheck with the functional probe**

In the same `gliner:` service, replace the `healthcheck:` block:

```yaml
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/-/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 60
      start_period: 120s
```
with:
```yaml
    healthcheck:
      test: ["CMD-SHELL", "curl -sf -m 10 -X POST -H 'content-type: application/json' -d '{\"text\":\"ping\",\"labels\":[\"person\"],\"threshold\":0.5}' http://localhost:8000/gliner || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 180s
```

- [ ] **Step 3: Add a real restart directive**

In the same `gliner:` service, add a top-level `restart:` key (the existing `deploy: *cuda-deploy`'s `restart_policy` is Swarm-only and ignored by `docker compose up`). Add this line adjacent to the other top-level service keys (e.g. just after `deploy: *cuda-deploy`):

```yaml
    restart: unless-stopped
```

- [ ] **Step 4: Validate compose syntax**

Run: `docker compose -f docker/compose.yaml config >/dev/null && echo CONFIG-OK`
Expected: `CONFIG-OK` (no YAML/interpolation errors).

- [ ] **Step 5: Bring up the stack and verify real health + end-to-end NER**

Run:
```bash
docker compose -f docker/compose.yaml up -d --build gliner router
# wait for gliner to report healthy under the NEW deep healthcheck
for _ in $(seq 1 40); do
  s=$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose -f docker/compose.yaml ps -q gliner)" 2>/dev/null)
  echo "health=$s"; [ "$s" = "healthy" ] && break; sleep 10
done
# end-to-end NER through the router (the path docint uses)
docker exec "$(docker compose -f docker/compose.yaml ps -q gliner)" \
  curl -sf -X POST -H 'content-type: application/json' \
  -d '{"text":"Angela Merkel visited Berlin.","labels":["person","loc"],"threshold":0.5}' \
  http://localhost:8000/gliner
```
Expected: `health=healthy` within the loop, and a JSON `{"entities":[…]}` with `Angela Merkel`/`Berlin`.

- [ ] **Step 6: Verify clean shutdown (no 137)**

Run:
```bash
docker compose -f docker/compose.yaml stop gliner
docker inspect -f '{{.State.ExitCode}}' "$(docker compose -f docker/compose.yaml ps -aq gliner)"
```
Expected: exit code `0` (the watchdog's SIGTERM trap), not `137`.

- [ ] **Step 7: Commit**

```bash
git add docker/compose.yaml
git commit -m "feat(gliner): watchdog entrypoint, deep healthcheck, restart policy (full stack)"
```

---

### Task 4: Apply the same wiring to the CPU `gliner-only` shape

**Files:**
- Modify: `docker/compose.gliner-only.yaml` (the `gliner:` service)

**Interfaces:**
- Consumes: `/usr/local/bin/gliner-watchdog` (Task 2). `restart: unless-stopped` already present here — do not duplicate.

- [ ] **Step 1: Wrap the entrypoint command with the watchdog**

In `docker/compose.gliner-only.yaml`, change the `command:`'s final `exec "$@"` to:

```yaml
        exec /usr/local/bin/gliner-watchdog "$@"
```

- [ ] **Step 2: Replace the shallow healthcheck with the functional probe**

Replace the `healthcheck.test` line:

```yaml
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/-/healthz || exit 1"]
```
with:
```yaml
      test: ["CMD-SHELL", "curl -sf -m 10 -X POST -H 'content-type: application/json' -d '{\"text\":\"ping\",\"labels\":[\"person\"],\"threshold\":0.5}' http://localhost:8000/gliner || exit 1"]
```
Leave `interval`/`timeout`/`retries`/`start_period` as they are (this shape already uses `start_period: 180s`); do not add a `restart:` key (already present).

- [ ] **Step 3: Validate compose syntax**

Run: `docker compose -f docker/compose.gliner-only.yaml config >/dev/null && echo CONFIG-OK`
Expected: `CONFIG-OK`.

- [ ] **Step 4: Bring it up and verify health + NER (CPU)**

Run:
```bash
docker compose -f docker/compose.gliner-only.yaml up -d --build
gid="$(docker compose -f docker/compose.gliner-only.yaml ps -q gliner)"
for _ in $(seq 1 40); do
  s=$(docker inspect -f '{{.State.Health.Status}}' "$gid" 2>/dev/null)
  echo "health=$s"; [ "$s" = "healthy" ] && break; sleep 10
done
docker exec "$gid" curl -sf -X POST -H 'content-type: application/json' \
  -d '{"text":"Berlin","labels":["loc"],"threshold":0.5}' http://localhost:8000/gliner
```
Expected: `health=healthy`, then JSON containing `Berlin`/`loc`.

- [ ] **Step 5: Commit**

```bash
git add docker/compose.gliner-only.yaml
git commit -m "feat(gliner): watchdog entrypoint + deep healthcheck (gliner-only CPU shape)"
```

---

### Task 5: Document the knobs and behavior

**Files:**
- Modify: `.env.example` (add `NER_WATCHDOG_*` near the existing `NER_*` block)
- Modify: `README.md` and `CLAUDE.md` (note the watchdog)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the watchdog knobs to `.env.example`**

In `.env.example`, immediately after the existing `NER_*` knob comments (the block ending with `# NER_SHM_SIZE=3gb`), add:

```bash
# --- gliner self-heal watchdog (supervises gliner.serve; restarts the
# container when the Ray Serve deployment wedges, ray-project/ray#63862) ---
# NER_WATCHDOG_ENABLED=true            # false -> run gliner.serve directly (no watchdog)
# NER_WATCHDOG_GRACE_S=180             # readiness grace before arming (hard cap 2x)
# NER_WATCHDOG_INTERVAL_S=30           # probe interval
# NER_WATCHDOG_FAILURES=3              # consecutive failures before restart (~90s)
# NER_WATCHDOG_TIMEOUT_S=10            # per-probe timeout (a hang counts as failure)
# NER_WATCHDOG_WEBHOOK_URL=            # optional alert target (Slack-style {"text":...}); unset = log only
```

- [ ] **Step 2: Note the watchdog in `README.md` and `CLAUDE.md`**

Add, wherever the gliner service is described in each file, a sentence such as:

```markdown
The gliner container is supervised by `scripts/gliner_watchdog.sh` (installed as
`/usr/local/bin/gliner-watchdog`): it probes `POST /gliner` and self-exits after
`NER_WATCHDOG_FAILURES` consecutive failures so `restart: unless-stopped` recreates
the container, recovering from the Ray Serve rank-consistency wedge
(ray-project/ray#63862). The Docker healthcheck uses the same functional probe;
set `NER_WATCHDOG_ENABLED=false` to disable. Knobs: `NER_WATCHDOG_*` in `.env.example`.
```

- [ ] **Step 3: Run repo lint (pre-commit) and confirm docs/grep**

Run:
```bash
pre-commit run --all-files || true   # shell/docs changes shouldn't trip ruff/pyrefly (Python-only)
grep -q NER_WATCHDOG_ENABLED .env.example && echo ENV-OK
grep -q gliner-watchdog README.md CLAUDE.md && echo DOCS-OK
```
Expected: pre-commit passes (or only reformats unrelated files), `ENV-OK`, `DOCS-OK`.

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md CLAUDE.md
git commit -m "docs(gliner): document NER_WATCHDOG_* knobs and self-heal behavior"
```

---

### Task 6: Integration verification — induce a wedge, confirm auto-recovery

**Files:** none (verification task; produces no code).

**Interfaces:** Consumes the running full-stack gliner from Task 3.

- [ ] **Step 1: Bring up gliner and confirm healthy**

Run:
```bash
docker compose -f docker/compose.yaml up -d --build gliner
gid="$(docker compose -f docker/compose.yaml ps -q gliner)"
for _ in $(seq 1 40); do
  s=$(docker inspect -f '{{.State.Health.Status}}' "$gid"); echo "health=$s"
  [ "$s" = "healthy" ] && break; sleep 10
done
```
Expected: `health=healthy`.

- [ ] **Step 2: Induce an unserviceable state**

Kill the Ray Serve replica/proxy worker inside the container so `/gliner` stops responding while the `gliner.serve` parent (the watchdog's child) stays alive:

```bash
docker exec "$gid" sh -c 'pkill -f "ServeReplica" || pkill -f "ray::ServeController" || pkill -f "proxy"'
```
(Any of these forces the deployment to stop serving without exiting the main process — the exact wedge the watchdog targets.)

- [ ] **Step 3: Observe the watchdog restart the container**

Run:
```bash
# Watch RestartCount climb and health return. Allow up to ~FAILURES*INTERVAL + restart.
for _ in $(seq 1 30); do
  rc=$(docker inspect -f '{{.RestartCount}}' "$gid")
  s=$(docker inspect -f '{{.State.Health.Status}}' "$gid")
  echo "restarts=$rc health=$s"; sleep 10
done
docker logs "$gid" 2>&1 | grep -i 'gliner-watchdog' | tail -5
```
Expected: `restarts` increments (≥1) and `health` returns to `healthy`; watchdog logs show the `"exiting for restart"` alert line.

- [ ] **Step 4: Confirm NER works again end-to-end**

Run:
```bash
docker exec "$gid" curl -sf -X POST -H 'content-type: application/json' \
  -d '{"text":"Angela Merkel visited Berlin.","labels":["person","loc"],"threshold":0.5}' \
  http://localhost:8000/gliner
```
Expected: JSON `{"entities":[…]}` with `Angela Merkel`/`Berlin`.

- [ ] **Step 5: Record the result in the spec (verification note)**

Append a short "Verified <date>: wedge injection → auto-restart → recovery confirmed (RestartCount N→N+1, clean stop exit 0)" note to the design spec's end, then commit:

```bash
git add docs/superpowers/specs/2026-06-25-gliner-self-heal-watchdog-design.md
git commit -m "docs(gliner): record self-heal integration verification"
```

---

## Self-Review

**1. Spec coverage:**
- Watchdog (detect→self-exit→restart) → Task 1 (script) + Tasks 3/4 (wiring). ✓
- Deep `/gliner` healthcheck replacing `/-/healthz` → Tasks 3 & 4. ✓
- Real `restart: unless-stopped` (CUDA) → Task 3 Step 3; CPU already has it (noted Task 4). ✓
- Both shapes covered → Tasks 3 (CUDA) + 4 (CPU). ✓
- Config knobs `NER_WATCHDOG_*` + defaults → Task 1 (consumed) + Task 5 (documented). ✓
- Signal handling / no-137 → Task 1 Step 3 (trap) + Task 3 Step 6 (verified). ✓
- Alerting (log + optional webhook) → Task 1 `alert()`; channel deferred per spec. ✓
- Testing (standalone shell, build smoke, manual integration) → Task 1, Task 2 Step 3, Task 6. ✓
- Rollout/revert (`NER_WATCHDOG_ENABLED=false`) → Task 1 escape hatch + Task 5 docs. ✓
- Tracking #63862 → recorded in spec; no code task needed. ✓
No gaps.

**2. Placeholder scan:** No TBD/TODO; all code blocks are complete (watchdog, test, dockerfile, compose, env). Webhook URL intentionally blank (operator-supplied), not a placeholder. ✓

**3. Type/name consistency:** Env var names identical across script, `.env.example`, and docs (`NER_WATCHDOG_ENABLED|GRACE_S|INTERVAL_S|FAILURES|TIMEOUT_S|WEBHOOK_URL`). Probe body identical in watchdog, healthcheck (both shapes), and tests. Installed path `/usr/local/bin/gliner-watchdog` consistent across Dockerfiles and both compose commands. ✓
