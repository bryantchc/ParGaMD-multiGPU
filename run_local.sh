#!/usr/bin/env bash
# Local-workstation launcher for ParGaMD on a single host.
#
# Replaces the cluster ZMQ master/client split with WESTPA's processes
# work manager: one w_run, N forked workers.
#
# Override defaults via env:
#   WORKERS_GPU0=4 CUDA_VISIBLE_DEVICES=0 ./run_local.sh
#   USE_MPS=1 WORKERS_GPU0=12 ./run_local.sh        # opt-in to MPS
#
# Restart-safe: only calls init.sh if west.h5 is missing; Ctrl-C cleans up.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
: "${CUDA_VISIBLE_DEVICES:=0}"     # nvidia-smi -L to confirm GPU index<->card
export CUDA_VISIBLE_DEVICES
WORKERS_GPU0=${WORKERS_GPU0:-8}     # 5090 (32 GB) - chignolin uses ~300-500 MB/walker; 8 ≈ paper's GH200 sweet spot
WORKERS_GPU1=${WORKERS_GPU1:-6}     # 5070 (12 GB)
CONDA_ENV=${CONDA_ENV:-pargamd}

# MPS modes:
#   USE_MPS=auto  (default) detect a live sudo-mode MPS daemon at
#                 /tmp/nvidia-mps. If present → use it (= mode 2).
#                 If absent → no MPS (= mode 0). Zero friction:
#                     sudo nvidia-cuda-mps-control -d   # once per session
#                 then every ./run_local.sh runs at full MPS speed.
#                 Stop with `echo quit | sudo nvidia-cuda-mps-control`.
#   USE_MPS=0     force no MPS. Workers serialize on the GPU via the
#                 CUDA driver. ~50% efficient at 4 workers, but correct.
#   USE_MPS=1     user-mode MPS — this script starts a per-GPU daemon.
#                 KNOWN BROKEN on Blackwell + GaMD: concurrent context
#                 creation triggers OverflowError in setStepCount on
#                 loadCheckpoint. Kept only for upstream-bug repro.
#   USE_MPS=2     force external sudo-mode MPS. Fails loud if the
#                 daemon isn't running. Use when you want to be sure
#                 MPS is active and not silently fall back.
USE_MPS=${USE_MPS:-auto}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
cd "$(dirname "$(readlink -f "$0")")"
export WEST_SIM_ROOT="$PWD"

# shellcheck disable=SC1090
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
command -v w_run >/dev/null || { echo "[run_local] w_run not on PATH; bad conda env?"; exit 1; }

# Each walker is a single-GPU OpenMM job; CPU threading is wasted and
# contends with the other walkers.
export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Worker counts and per-walker MPS thread share
# ---------------------------------------------------------------------------
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_WORKERS=0
for gpuid in "${DEVICES[@]}"; do
    case "$gpuid" in
        0) TOTAL_WORKERS=$(( TOTAL_WORKERS + WORKERS_GPU0 )) ;;
        1) TOTAL_WORKERS=$(( TOTAL_WORKERS + WORKERS_GPU1 )) ;;
        *) TOTAL_WORKERS=$(( TOTAL_WORKERS + 4 )) ;;
    esac
done
THREAD_PCT_GPU0=$(( 100 / WORKERS_GPU0 ))
THREAD_PCT_GPU1=$(( 100 / WORKERS_GPU1 ))

# Resolve USE_MPS=auto to a concrete mode by probing for a live sudo daemon.
# We probe the system pipe dir AND verify the recorded PID is actually a
# running process — a leftover socket file alone isn't enough. The daemon
# is root-owned (started by sudo), so we use /proc/$pid (visible cross-uid)
# rather than kill -0 (returns EPERM for processes we don't own).
mps_alive() {
    [ -S /tmp/nvidia-mps/control ] || return 1
    local pidfile=/tmp/nvidia-mps/nvidia-cuda-mps-control.pid
    [ -f "$pidfile" ] || return 1
    local pid; pid=$(cat "$pidfile" 2>/dev/null)
    [ -n "$pid" ] && [ -d "/proc/$pid" ]
}
if [ "$USE_MPS" = "auto" ]; then
    if mps_alive; then
        USE_MPS=2
        echo "[run_local] auto: live sudo MPS daemon detected, using it"
    else
        USE_MPS=0
        echo "[run_local] auto: no sudo MPS daemon at /tmp/nvidia-mps; running without MPS"
        echo "[run_local]       (start with 'sudo nvidia-cuda-mps-control -d' for ~5x speedup)"
    fi
fi

echo "[run_local] WEST_SIM_ROOT=$WEST_SIM_ROOT  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  workers=$TOTAL_WORKERS  USE_MPS=$USE_MPS"

# ---------------------------------------------------------------------------
# MPS lifecycle (per-GPU pipe dir = isolated control socket per daemon)
# ---------------------------------------------------------------------------
start_mps() {  # $1 gpuid  $2 thread%
    local pipe="/tmp/nvidia-mps-$USER-$1"
    local logd="/tmp/nvidia-log-$USER-$1"
    mkdir -p "$pipe" "$logd"
    CUDA_VISIBLE_DEVICES="$1" CUDA_MPS_PIPE_DIRECTORY="$pipe" \
        CUDA_MPS_LOG_DIRECTORY="$logd" nvidia-cuda-mps-control -d
    for _ in {1..10}; do [ -S "$pipe/control" ] && break; sleep 0.5; done
    echo "set_default_active_thread_percentage $2" \
        | CUDA_MPS_PIPE_DIRECTORY="$pipe" nvidia-cuda-mps-control
}
stop_mps() {
    local pipe="/tmp/nvidia-mps-$USER-$1"
    [ -d "$pipe" ] || return 0
    echo quit | CUDA_MPS_PIPE_DIRECTORY="$pipe" nvidia-cuda-mps-control 2>/dev/null || true
}
cleanup() {
    echo "[run_local] cleanup..."
    # SIGINT to w_run so WESTPA finishes the current segment and writes a
    # clean iteration boundary. Without this, killing run_local.sh leaves
    # w_run + workers orphaned on the GPU.
    if [ -n "${WRUN_PID:-}" ] && kill -0 "$WRUN_PID" 2>/dev/null; then
        echo "[run_local] sending SIGINT to w_run (PID $WRUN_PID); workers will drain..."
        kill -INT "$WRUN_PID" 2>/dev/null || true
        # Give WESTPA up to 90 s to flush west.h5 cleanly.
        for _ in {1..90}; do
            kill -0 "$WRUN_PID" 2>/dev/null || break
            sleep 1
        done
        # Escalate if it's still alive.
        if kill -0 "$WRUN_PID" 2>/dev/null; then
            echo "[run_local] w_run did not exit on SIGINT; sending SIGTERM"
            pkill -TERM -P "$WRUN_PID" 2>/dev/null || true
            kill -TERM "$WRUN_PID" 2>/dev/null || true
        fi
    fi
    [ -n "${LOG_PID:-}" ] && kill "$LOG_PID" 2>/dev/null || true
    if [ "$USE_MPS" = "1" ]; then
        for gpuid in "${DEVICES[@]}"; do stop_mps "$gpuid"; done
    fi
    # USE_MPS=2: do not touch the user-managed sudo daemon.
}
trap cleanup EXIT INT TERM

case "$USE_MPS" in
    1)
        for gpuid in "${DEVICES[@]}"; do
            case "$gpuid" in
                0) start_mps "$gpuid" "$THREAD_PCT_GPU0" ;;
                1) start_mps "$gpuid" "$THREAD_PCT_GPU1" ;;
                *) start_mps "$gpuid" 25 ;;
            esac
        done
        FIRST_GPU="${DEVICES[0]}"
        export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-$USER-$FIRST_GPU"
        export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-$USER-$FIRST_GPU"
        ;;
    2)
        # Connect to an externally-managed (sudo-started) MPS daemon at
        # the default system pipe dir. Verify it's actually running.
        if [ ! -S /tmp/nvidia-mps/control ]; then
            echo "[run_local] USE_MPS=2 but /tmp/nvidia-mps/control is missing." >&2
            echo "[run_local] Start the daemon in your terminal first:" >&2
            echo "[run_local]   sudo nvidia-cuda-mps-control -d" >&2
            exit 1
        fi
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
        export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
        echo "[run_local] using existing sudo-mode MPS at $CUDA_MPS_PIPE_DIRECTORY"
        ;;
    *)
        # USE_MPS=0 (default) or anything else: no MPS. Make sure no stray
        # MPS env from the parent shell routes us to a phantom daemon.
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
        ;;
esac

# ---------------------------------------------------------------------------
# WESTPA init or restart
# ---------------------------------------------------------------------------
# Warn (don't block) if west.cfg still has the out-of-the-box [-inf, inf]
# placeholder bins. Single-bin WE collapses to plain GaMD — almost certainly
# not what you want for a real system. Hard-erroring would block legitimate
# smoke tests and the existing chignolin run, so this is advisory.
if grep -qE "^\s*-\s*\['-inf',\s*'inf'\]" west.cfg; then
    echo "[run_local] WARNING: west.cfg has placeholder [-inf, inf] bins."
    echo "[run_local]          Real WE sampling needs tuned bin edges along your pcoord axes."
    echo "[run_local]          Edit west.cfg → system.system_options.bins.boundaries before production."
fi

if [ ! -f west.h5 ]; then
    echo "[run_local] west.h5 missing; running init.sh"
    ./init.sh
else
    echo "[run_local] west.h5 present; restarting in place"
fi
# shellcheck disable=SC1091
source env.sh

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used \
           --format=csv -l 10 > gpu_util.log &
LOG_PID=$!

# Launch w_run in the background so we can capture its PID for the cleanup
# trap; then `wait` so the script blocks until w_run exits or a signal hits.
w_run --work-manager=processes --n-workers="$TOTAL_WORKERS" "$@" &
WRUN_PID=$!
wait "$WRUN_PID" || true
echo "[run_local] w_run exited."
