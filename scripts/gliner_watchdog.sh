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
