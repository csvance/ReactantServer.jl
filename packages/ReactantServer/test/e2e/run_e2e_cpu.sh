#!/usr/bin/env bash
# CPU end-to-end test of the supervised single-container layout, with no containers: the
# ReactantServerNode supervisor spawns two CPU workers plus the embedded gateway as host
# subprocesses, the client drives scale4 through the gateway, and SIGTERM shuts the tree down.
# Verifies worker synthesis (no workers list in the node file), the env-only embedded gateway,
# prefixed log multiplexing, and graceful shutdown. Needs no GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
E2E_DIR="packages/ReactantServer/test/e2e"

WORK="$(mktemp -d)"
SUP_PID=""
cleanup() {
    # TERM, not KILL: the supervisor forwards SIGTERM to its children and waits for them;
    # SIGKILL would orphan every worker and the gateway.
    if [[ -n "$SUP_PID" ]] && kill -0 "$SUP_PID" 2>/dev/null; then
        kill -TERM "$SUP_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SUP_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SUP_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
BASE_PORT="$(free_port)"
METRICS_BASE_PORT="$(free_port)"
GW_PORT="$(free_port)"
GW_METRICS_PORT="$(free_port)"
# Generous default: the first run precompiles ReactantServer under the preference override
# below (a different preferences hash forces a full Reactant recompile, CPU-only and slow).
READY_TIMEOUT="${E2E_CPU_READY_TIMEOUT:-1800}"

echo "== [1/5] generating scale4 bundle =="
julia --project=packages/ReactantServer "$E2E_DIR/gen_scale4.jl"

mkdir -p "$WORK/models"
cp -r "$E2E_DIR/models/scale4" "$WORK/models/scale4"

echo "== [2/5] writing node file and Reactant preference override =="
# No workers list: the supervisor synthesizes them (REACTANT_CPU_WORKERS below).
cat > "$WORK/node.yaml" << EOF
model_repo: $WORK/models
base_port: $BASE_PORT
metrics_base_port: $METRICS_BASE_PORT
global:
  cache_dir: $WORK/cache
  runtime:
    backend: cpu
  endpoints:
    host: 127.0.0.1
EOF

# The worker project's LocalPreferences.toml pins Reactant's persistent compile cache to
# /var/cache/reactant-compile (a Docker volume mount point), which is not writable on a dev
# host. Prepend a load-path entry that disables the persistent cache for every child.
mkdir -p "$WORK/prefs"
cat > "$WORK/prefs/Project.toml" << 'EOF'
[extras]
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
EOF
cat > "$WORK/prefs/LocalPreferences.toml" << 'EOF'
[Reactant]
persistent_cache_enabled = false
EOF
export JULIA_LOAD_PATH="$WORK/prefs:@:@v#.#:@stdlib"

echo "== [3/5] starting supervisor (2 CPU workers + embedded gateway) =="
REACTANT_GPUS=0 \
REACTANT_CPU_WORKERS=2 \
REACTANT_WORKSPACE_ROOT="$REPO_ROOT" \
REACTANT_GATEWAY_LISTEN_GRPC="127.0.0.1:$GW_PORT" \
REACTANT_GATEWAY_LISTEN_METRICS="127.0.0.1:$GW_METRICS_PORT" \
julia --handle-signals=no --project=packages/ReactantServerNode -e '
    using ReactantServerNode
    exit(ReactantServerNode.supervise(ARGS[1]; runtime_dir = ARGS[2]))
' "$WORK/node.yaml" "$WORK/run" > "$WORK/supervisor.log" 2>&1 &
SUP_PID=$!

echo "   waiting for gateway /readyz on :$GW_METRICS_PORT (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/readyz" >/dev/null 2>&1; do
    if ! kill -0 "$SUP_PID" 2>/dev/null; then
        echo "ERROR: supervisor exited early; log:"
        tail -80 "$WORK/supervisor.log"
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "ERROR: gateway not ready within ${READY_TIMEOUT}s; supervisor and gateway lines:"
        grep -E "^\[(supervisor|gateway)\]" "$WORK/supervisor.log" | tail -120
        exit 1
    fi
    sleep 2
done
echo "   gateway ready."

echo "== [4/5] running CPU e2e client =="
set +e
julia --project=packages/ReactantServerGateway "$E2E_DIR/client_cpu.jl" "$GW_PORT" "$BASE_PORT"
client_rc=$?
set -e
if ((client_rc != 0)); then
    echo "== client failed (rc=${client_rc}); recent supervisor log: =="
    tail -120 "$WORK/supervisor.log"
    exit "$client_rc"
fi

# The gateway's /metrics aggregates every worker's export; workers self-tag their series.
metrics="$(curl -fsS "http://127.0.0.1:${GW_METRICS_PORT}/metrics")"
for tag in 'worker="worker0"' 'worker="worker1"' 'gateway_worker_metrics_up'; do
    if ! grep -qF "$tag" <<< "$metrics"; then
        echo "ERROR: aggregated /metrics is missing $tag"
        exit 1
    fi
done
echo "   aggregated /metrics carries per-worker series."

echo "== [5/5] SIGTERM shutdown =="
kill -TERM "$SUP_PID"
sup_rc=0
wait "$SUP_PID" || sup_rc=$?
if ((sup_rc != 0)); then
    echo "ERROR: supervisor exited $sup_rc on SIGTERM; recent log:"
    tail -80 "$WORK/supervisor.log"
    exit 1
fi

# Smoke-check the multiplexed log: every child line carries its prefix.
for tag in worker0 worker1 gateway supervisor; do
    if ! grep -q "^\[$tag\] " "$WORK/supervisor.log"; then
        echo "ERROR: no [$tag] lines in the supervisor log"
        tail -40 "$WORK/supervisor.log"
        exit 1
    fi
done

echo "CPU e2e: PASS (workers synthesized, gateway routed, clean shutdown)"
