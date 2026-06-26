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
