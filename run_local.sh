#!/usr/bin/env bash
# Local-workstation ParGaMD launcher — ZMQ work manager edition.
#
# Spawns one ZMQ master + N independent worker processes, each pinned
# to a specific GPU via CUDA_VISIBLE_DEVICES. Workers connect to the
# master over TCP/ZMQ, so the same script works for:
#   - this workstation (master + workers all on one host)
#   - a single beefy cloud box (8 GPUs, master + 8 workers, one per GPU)
#   - a multi-node cluster (master on one node, workers on N others)
#     with the master's host info shared via $SERVER_INFO on a shared FS
#
# Override defaults via env:
#   WORKERS_GPU0=8 WORKERS_GPU1=4 ./run_local.sh
#   USE_MPS=2 CUDA_VISIBLE_DEVICES=0,1 ./run_local.sh
#
# Restart-safe: only calls init.sh if west.h5 is missing.
# Trap-based cleanup: SIGINT to master and workers; sudo MPS untouched.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
: "${CUDA_VISIBLE_DEVICES:=0,1}"          # default both GPUs visible
export CUDA_VISIBLE_DEVICES
WORKERS_GPU0=${WORKERS_GPU0:-8}            # 5090 (32 GB) — chignolin uses ~300-500 MB/walker
WORKERS_GPU1=${WORKERS_GPU1:-4}            # 5070 (12 GB) — fewer walkers, lower per-GPU memory
CONDA_ENV=${CONDA_ENV:-pargamd}

# MPS modes (same as single-host processes launcher):
#   USE_MPS=auto  detect /tmp/nvidia-mps; use it if alive, else no-MPS
#   USE_MPS=0     no MPS (workers serialize on each GPU)
#   USE_MPS=2     require external sudo-mode MPS daemon at /tmp/nvidia-mps
#   USE_MPS=1     user-mode MPS — DO NOT USE on Blackwell (gamdRunner OverflowError)
USE_MPS=${USE_MPS:-auto}

# ZMQ heartbeat tuning (inherited if set in env.sh)
: "${WM_ZMQ_MASTER_HEARTBEAT:=50}"
: "${WM_ZMQ_WORKER_HEARTBEAT:=50}"
: "${WM_ZMQ_TIMEOUT_FACTOR:=100}"
export WM_ZMQ_MASTER_HEARTBEAT WM_ZMQ_WORKER_HEARTBEAT WM_ZMQ_TIMEOUT_FACTOR

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
cd "$(dirname "$(readlink -f "$0")")"
export WEST_SIM_ROOT="$PWD"
SERVER_INFO="$WEST_SIM_ROOT/west_zmq_info.json"
export SERVER_INFO
rm -f "$SERVER_INFO"   # stale info file from a previous run will mislead workers

# shellcheck disable=SC1090
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
command -v w_run >/dev/null || { echo "[run_local] w_run not on PATH; bad conda env?"; exit 1; }

export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Worker plan
# ---------------------------------------------------------------------------
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
declare -A WORKERS_PER_GPU
TOTAL_WORKERS=0
for gpuid in "${DEVICES[@]}"; do
    case "$gpuid" in
        0) WORKERS_PER_GPU[$gpuid]=$WORKERS_GPU0 ;;
        1) WORKERS_PER_GPU[$gpuid]=$WORKERS_GPU1 ;;
        *) WORKERS_PER_GPU[$gpuid]=4 ;;
    esac
    TOTAL_WORKERS=$(( TOTAL_WORKERS + WORKERS_PER_GPU[$gpuid] ))
done

# ---------------------------------------------------------------------------
# MPS auto-detect
# ---------------------------------------------------------------------------
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

if [ "$USE_MPS" = "2" ]; then
    # Per-GPU sudo MPS pipe dirs preferred (when /tmp/nvidia-mps-${gpuid}/control
    # exists for every visible GPU). Falls back to a single shared daemon at
    # /tmp/nvidia-mps if those aren't present.
    PER_GPU_MPS=1
    for gpuid in "${DEVICES[@]}"; do
        [ -S "/tmp/nvidia-mps-${gpuid}/control" ] || PER_GPU_MPS=0
    done
    if [ "$PER_GPU_MPS" = "1" ]; then
        echo "[run_local] using per-GPU sudo MPS daemons (/tmp/nvidia-mps-{${CUDA_VISIBLE_DEVICES}})"
        # CUDA_MPS_PIPE_DIRECTORY will be set per-worker in the spawn loop below.
    elif [ -S /tmp/nvidia-mps/control ]; then
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
        export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
        echo "[run_local] using shared sudo-mode MPS at $CUDA_MPS_PIPE_DIRECTORY"
    else
        echo "[run_local] USE_MPS=2 but no sudo daemon found at /tmp/nvidia-mps or /tmp/nvidia-mps-{0,1,...}." >&2
        echo "[run_local] Start a daemon: sudo nvidia-cuda-mps-control -d" >&2
        exit 1
    fi
elif [ "$USE_MPS" = "1" ]; then
    echo "[run_local] USE_MPS=1 (user-mode) is broken on Blackwell. Use sudo MPS (=2) instead." >&2
    exit 1
else
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY 2>/dev/null || true
fi

echo "[run_local] WEST_SIM_ROOT=$WEST_SIM_ROOT"
echo "[run_local] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  total workers=$TOTAL_WORKERS  USE_MPS=$USE_MPS"
for gpuid in "${DEVICES[@]}"; do
    echo "[run_local]   GPU $gpuid: ${WORKERS_PER_GPU[$gpuid]} workers"
done

# ---------------------------------------------------------------------------
# Bin warning + WESTPA init
# ---------------------------------------------------------------------------
if grep -qE "^\s*-\s*\['-inf',\s*'inf'\]" west.cfg; then
    echo "[run_local] WARNING: west.cfg has placeholder [-inf, inf] bins."
    echo "[run_local]          Real WE sampling needs tuned bin edges."
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
# Cleanup trap
# ---------------------------------------------------------------------------
WORKER_PIDS=()
MASTER_PID=""
LOG_PID=""

cleanup() {
    echo "[run_local] cleanup..."
    # Master first — once it exits, workers see disconnect and shut down on their own.
    if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "[run_local] sending SIGINT to ZMQ master (PID $MASTER_PID); workers will drain..."
        kill -INT "$MASTER_PID" 2>/dev/null || true
        for _ in {1..90}; do
            kill -0 "$MASTER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$MASTER_PID" 2>/dev/null && kill -TERM "$MASTER_PID" 2>/dev/null || true
    fi
    # Backstop — any worker that didn't disconnect cleanly.
    for wpid in "${WORKER_PIDS[@]}"; do
        kill -INT "$wpid" 2>/dev/null || true
    done
    sleep 2
    for wpid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$wpid" 2>/dev/null || true
    done
    [ -n "$LOG_PID" ] && kill "$LOG_PID" 2>/dev/null || true

    # Reap orphaned per-segment children. When a worker dies hard, its
    # runseg.sh + python gamdRunner descendants get reparented to init
    # and end up stuck on the MPS Unix socket (__skb_wait_for_more_packets);
    # only SIGKILL clears them. Scope by $WEST_SIM_ROOT so we don't touch
    # unrelated processes on the host.
    pkill -KILL -f "$WEST_SIM_ROOT/westpa_scripts/runseg.sh" 2>/dev/null || true
    pkill -KILL -f "$WEST_SIM_ROOT/common_files/gamdRunner"  2>/dev/null || true

    # Clear stale runtime files so the next launch starts clean.
    rm -f "$SERVER_INFO" /tmp/pargamd-cuda-init.lock 2>/dev/null || true

    # USE_MPS=2: do not touch the user-managed sudo daemon.
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Launch ZMQ master in background
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used \
           --format=csv -l 10 > gpu_util.log &
LOG_PID=$!

echo "[run_local] starting ZMQ master..."
w_run --work-manager=zmq \
      --n-workers=0 \
      --zmq-mode=master \
      --zmq-write-host-info="$SERVER_INFO" \
      --zmq-comm-mode=tcp \
      "$@" \
      &> "west_master.log" &
MASTER_PID=$!

# Wait up to 60 s for the master to publish $SERVER_INFO.
for _ in $(seq 1 60); do
    [ -e "$SERVER_INFO" ] && break
    kill -0 "$MASTER_PID" 2>/dev/null || { echo "[run_local] master died before publishing $SERVER_INFO; see west_master.log"; exit 1; }
    sleep 1
done
if [ ! -e "$SERVER_INFO" ]; then
    echo "[run_local] ZMQ master did not publish host info within 60 s; aborting." >&2
    exit 1
fi
echo "[run_local] master ready (PID $MASTER_PID); host info in $SERVER_INFO"

# ---------------------------------------------------------------------------
# Launch one worker process per (gpu, slot)
# ---------------------------------------------------------------------------
for gpuid in "${DEVICES[@]}"; do
    n_workers=${WORKERS_PER_GPU[$gpuid]}
    for ((w=1; w<=n_workers; w++)); do
        (
          # Unified-namespace approach: workers inherit the master's
          # CUDA_VISIBLE_DEVICES (e.g. "0,1") so the driver and any single
          # shared MPS daemon see a consistent device numbering across all
          # processes. Each worker just signals which physical GPU to target
          # for the segment via TARGET_GPU; runseg.sh forwards this to
          # gamdRunner via the -d flag rather than via CUDA_VD restriction.
          export TARGET_GPU="$gpuid"
          exec w_run --work-manager=zmq \
                     --n-workers=1 \
                     --zmq-mode=client \
                     --zmq-read-host-info="$SERVER_INFO" \
                     --zmq-comm-mode=tcp \
                     &> "west_worker_gpu${gpuid}_w${w}.log"
        ) &
        WORKER_PIDS+=($!)
    done
done
echo "[run_local] launched ${#WORKER_PIDS[@]} workers; waiting for master to exit..."

# ---------------------------------------------------------------------------
# Block until master finishes (success, end-of-iterations, or signal)
# ---------------------------------------------------------------------------
wait "$MASTER_PID" || true
echo "[run_local] master exited."
