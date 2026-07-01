# gliner self-heal watchdog — design

- **Date:** 2026-06-25
- **Status:** Approved (design); implementation pending
- **Component:** vllm-service — `gliner` backend (full CUDA stack + `gliner-only` CPU shape)
- **Related:** docint NER outage 2026-06-25 (`Remote NER call failed: 500 … /gliner`); upstream Ray Serve bug [ray-project/ray#63862](https://github.com/ray-project/ray/issues/63862)

## Summary

Add an **in-container watchdog** to the gliner service that detects a wedged Ray Serve
deployment and self-exits, so Docker's restart policy recreates the container (a fresh
embedded Ray clears the wedge). Replace the shallow `/-/healthz` healthcheck with a
**functional NER probe**, and add a real `restart:` directive to the full-stack compose.
This turns a silent, manual-recovery outage into unattended recovery in ~90 s, with an alert.

## Background

### The failure mode

gliner runs the upstream `python -m gliner.serve` module, which serves the model behind
**Ray Serve**. On 2026-06-25 the Ray Serve controller wedged in a tight loop:

```
controller -- Found active keys without ranks: {'…'}. This should never happen. Please report this as a bug.
controller -- Error executing function _check_rank_consistency_impl: Rank system is in an invalid state.
```

In this state the Serve replica never becomes healthy, so nothing listens on `gliner:8000`.
The LiteLLM router's `/gliner` pass-through then gets **connection-refused → 500**, which
docint surfaces as a flood of `Remote NER call failed` warnings (it degrades gracefully).
Recovery required a manual `docker compose down && up`.

This is a **known, confirmed Ray Serve bug on our exact version (Ray 2.55.1)** — issues
[#63862](https://github.com/ray-project/ray/issues/63862) (open) and
[#64103](https://github.com/ray-project/ray/issues/64103) (closed dup). Trigger: the
ServeController is killed/restarted and, on restart, its replica-rank bookkeeping is left
inconsistent. The replica-rank feature was introduced in Ray 2.50.0.

### Why it is neither detected nor recovered today

1. **Detection gap.** Both compose shapes healthcheck Ray Serve's `/-/healthz` (proxy
   liveness), which stays `200` while the deployment replica is wedged — so Docker keeps
   gliner marked *healthy* straight through the outage.
2. **Recovery gap.** `docker compose` never restarts an *unhealthy* container at runtime
   (`depends_on: service_healthy` only gates startup). And in the full CUDA `compose.yaml`,
   gliner's only restart directive is `deploy.restart_policy`, which `docker compose up`
   **ignores** (Swarm-only) — so even a hard exit would not be restarted.

### Why not fix it upstream instead

- **Upgrade Ray:** no released fix. 2.55.1 is the latest release; the only relevant merged
  fix (PR #63139) is master/nightly-only and, per the reporter, the symptom still
  reproduces on nightly. Rejected.
- **Downgrade below the feature:** the last release without replica-ranks is `ray-2.49.2`
  (2025-09-19) — a ~6-minor-version regression, and the new `gliner.serve` module (≈v0.2.27)
  is untested on Ray that old. Disproportionate risk for a feature we don't use (we run a
  single replica). Rejected.
- **autoheal sidecar:** viable, but adds a container and a `/var/run/docker.sock` mount
  (privilege surface) plus an image to vendor into the offline build. Rejected in favor of
  the self-contained in-container watchdog.

Self-heal is therefore the primary mitigation; the upstream bug is tracked for a future
forward upgrade.

## Goals / Non-goals

**Goals**
- Detect a wedged (running-but-not-serving) gliner and recover **unattended** in ~90 s.
- Make `docker ps` health reflect *real* NER serving, not just proxy liveness.
- Emit an alert when a restart fires (auto-restart fixes serving, but documents ingested
  during the blip have empty entity metadata and must be re-ingested — a human must know).
- Apply uniformly to the full CUDA stack and the `gliner-only` CPU dev shape.
- No new containers, no docker-socket exposure, no new image dependency.

**Non-goals**
- Fixing the upstream Ray Serve rank bug.
- Backfilling NER for documents ingested during an outage (separate docint-side concern).
- Multi-replica / rank-aware behavior (we deploy `--num-replicas 1`).

## Design

### Component — `scripts/gliner_watchdog.sh`

A POSIX-sh supervisor (matches the repo's shell-script convention; no new Python module, so
no pyrefly/ruff strict-typing burden). `COPY`'d into both gliner images as
`/usr/local/bin/gliner-watchdog` (`chmod +x`). Each compose builder's final `exec "$@"`
becomes `exec /usr/local/bin/gliner-watchdog "$@"`, so the watchdog is the container's main
process and `gliner.serve` is its child.

Lifecycle:
1. If `NER_WATCHDOG_ENABLED != true`, `exec "$@"` (today's behavior — a no-rebuild escape hatch).
2. Launch the passed argv (`python -m gliner.serve …`) as a background child.
3. Trap `SIGTERM`/`SIGINT` → forward to child → `exit 0` (clean shutdown; removes the `137`
   seen on `compose down`).
4. **Readiness wait** (bounded by `2 × GRACE`): poll `/gliner`; on first success, proceed to
   steady-state. If the child dies, propagate its exit code. If never ready within the cap,
   alert + exit non-zero (fail-fast a broken boot).
5. **Steady-state monitor:** poll `/gliner` every `INTERVAL`; reset a failure counter on
   success; after `FAILURES` consecutive failures, alert, kill the child, and **exit
   non-zero** → Docker `restart: unless-stopped` recreates the container.

Illustrative pseudocode (final script will be shellcheck-clean):

```sh
#!/bin/sh
set -u
[ "${NER_WATCHDOG_ENABLED:-true}" = "true" ] || exec "$@"
GRACE=${NER_WATCHDOG_GRACE_S:-180}; INTERVAL=${NER_WATCHDOG_INTERVAL_S:-30}
MAXFAIL=${NER_WATCHDOG_FAILURES:-3}; TIMEOUT=${NER_WATCHDOG_TIMEOUT_S:-10}
WEBHOOK=${NER_WATCHDOG_WEBHOOK_URL:-}

probe() {
  curl -sf -m "$TIMEOUT" -X POST -H 'content-type: application/json' \
    -d '{"text":"ping","labels":["person"],"threshold":0.5}' \
    http://localhost:8000/gliner >/dev/null 2>&1
}
alert() {                                   # $1 = message
  echo "{\"level\":\"error\",\"svc\":\"gliner-watchdog\",\"msg\":\"$1\"}" >&2
  [ -n "$WEBHOOK" ] && curl -sf -m "$TIMEOUT" -X POST -H 'content-type: application/json' \
    -d "{\"text\":\"gliner watchdog: $1\"}" "$WEBHOOK" >/dev/null 2>&1 || true
}

"$@" & CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; exit 0' TERM INT

waited=0; ready=false
while [ "$waited" -lt $((GRACE * 2)) ]; do
  kill -0 "$CHILD" 2>/dev/null || { wait "$CHILD"; exit $?; }
  probe && { ready=true; break; }
  sleep "$INTERVAL"; waited=$((waited + INTERVAL))
done
[ "$ready" = true ] || { alert "not ready within $((GRACE*2))s; restarting"; kill -TERM "$CHILD" 2>/dev/null; exit 1; }

fails=0
while kill -0 "$CHILD" 2>/dev/null; do
  if probe; then fails=0; else
    fails=$((fails + 1))
    [ "$fails" -ge "$MAXFAIL" ] && { alert "unresponsive after $MAXFAIL probes; restarting"; kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; exit 1; }
  fi
  sleep "$INTERVAL"
done
wait "$CHILD"; exit $?
```

### Detection probe + healthcheck

The probe is a functional NER call: `POST /gliner {"text":"ping","labels":["person"],"threshold":0.5}`,
expecting `200`. It catches the wedge (connection-refused / 503 / hang) that `/-/healthz` misses.

The **Docker healthcheck** in both shapes changes to the same deep probe, so `docker ps`
shows true health and the router's `depends_on: gliner: service_healthy` gates startup on
real NER readiness:

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -sf -m 10 -X POST -H 'content-type: application/json' -d '{\"text\":\"ping\",\"labels\":[\"person\"],\"threshold\":0.5}' http://localhost:8000/gliner || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 180s
```

The watchdog and the healthcheck probe independently; both calls are sub-second on a
3-word input, so the duplicate load is negligible. (Optional future refinement: have the
watchdog write a heartbeat file the healthcheck reads, to probe once.)

### Configuration (env, `NER_WATCHDOG_*`)

| Variable | Default | Purpose |
|---|---|---|
| `NER_WATCHDOG_ENABLED` | `true` | `false` → `exec` gliner directly (today's behavior, no rebuild) |
| `NER_WATCHDOG_GRACE_S` | `180` | Readiness grace before arming (mirrors `start_period`); hard cap `2×` |
| `NER_WATCHDOG_INTERVAL_S` | `30` | Probe interval |
| `NER_WATCHDOG_FAILURES` | `3` | Consecutive failures before restart (≈90 s of downtime) |
| `NER_WATCHDOG_TIMEOUT_S` | `10` | Per-probe timeout (a hang counts as a failure) |
| `NER_WATCHDOG_WEBHOOK_URL` | _(unset)_ | Optional alert target; unset → log-only |

Documented in `.env.example` next to the existing `NER_*` knobs.

### Shutdown / signals

The watchdog forwards `SIGTERM`/`SIGINT` to the child and exits `0`, giving gliner a clean
shutdown on `docker stop` / `compose down` (eliminating the `137` exit observed during the
incident).

### Alerting

On restart the watchdog always emits a structured error log line; if `NER_WATCHDOG_WEBHOOK_URL`
is set it also POSTs a JSON message. Payload shape is a generic `{"text": "…"}` (Slack
incoming-webhook compatible); the concrete channel/URL is operator-supplied and out of scope
for this change.

## Files touched

| File | Change |
|---|---|
| `scripts/gliner_watchdog.sh` | **new** — the supervisor |
| `docker/Dockerfile.gliner.cuda` | `COPY` script to `/usr/local/bin/gliner-watchdog`, `chmod +x` |
| `docker/Dockerfile.gliner.cpu` | same `COPY` + `chmod` |
| `docker/compose.yaml` | gliner `command` final `exec` → watchdog; deep healthcheck; add top-level `restart: unless-stopped` |
| `docker/compose.gliner-only.yaml` | gliner `command` final `exec` → watchdog; deep healthcheck (`restart:` already present) |
| `.env.example` | document `NER_WATCHDOG_*` |
| `README.md` / `CLAUDE.md` | note the watchdog + recovery behavior |

## Testing & verification

Repo norm is "no unit-test suite; lint (ruff/pyrefly) + in-Dockerfile build smoke tests."

- **Standalone shell test:** run `gliner_watchdog.sh` against a toggleable stub HTTP server
  (e.g. `python -m http.server` fronting a flag file). Assert it (a) ignores failures during
  grace, (b) exits non-zero after `FAILURES` consecutive failures, (c) exits `0` on `SIGTERM`,
  (d) propagates the child's exit code when the child dies on its own. No GPU required.
- **Static check:** `sh -n scripts/gliner_watchdog.sh` (and `shellcheck` if added to pre-commit).
- **Manual integration:** build + `compose up gliner`; confirm healthy; induce an
  unserviceable state (`docker exec` and kill the Ray Serve replica/proxy process so
  `/gliner` stops responding while the `gliner.serve` parent — the watchdog's child — stays
  alive); confirm the watchdog detects it, restarts the container, and NER recovers (re-run
  the standard `POST /gliner` probe); confirm `compose down` exits `0`, not `137`.

## Rollout & revert

- Ships in the gliner image + compose; no API or data changes.
- **Instant revert without rebuild:** set `NER_WATCHDOG_ENABLED=false` → the entrypoint
  `exec`s gliner exactly as today.

## Risks & mitigations

- **False-positive restart loop** under transient slowness → require `FAILURES` consecutive
  failures + per-probe timeout; Docker restart backoff; alert surfaces a recurring wedge.
- **Premature kill during slow model load** → readiness grace (`2×GRACE`) before arming.
- **Probe load on GPU** → 3-word input, 30 s interval → sub-second, negligible.
- **Masking a real recurring bug** → every restart alerts/logs; track upstream #63862.

## Future work

- Track [ray-project/ray#63862](https://github.com/ray-project/ray/issues/63862). When a Ray
  release ships a **verified** fix, evaluate a forward upgrade and whether the watchdog can be
  relaxed (kept as defense-in-depth or removed).
- Optional: heartbeat-file shared between watchdog and healthcheck to probe once.

## Verification (2026-06-26)

Wedge injected by killing the Ray Serve ProxyActor and immediately seizing port 8000 with a
503-returning stub server, keeping the `python -m gliner.serve` child alive so the probe-failure
path was exercised (Ray transparently respawns replica/proxy within ~2 s, making a raw
`pkill ServeReplica` insufficient; port seizure was required to sustain 3 consecutive failures).
RestartCount transition: 1 → 2 (baseline was 0 → 1 from an earlier child-death path test).
Recovery time: container restarted and returned to `healthy` in under 30 s after watchdog exit.
Alert line observed in `docker logs`: `{"level":"error","svc":"gliner-watchdog","msg":"gliner unresponsive after 3 consecutive probes; exiting for restart"}`.
End-to-end NER confirmed post-recovery: `POST /gliner {"text":"Angela Merkel visited Berlin.","labels":["person","loc"]}` → `Angela Merkel (person, 0.992)`, `Berlin (loc, 0.969)`.
