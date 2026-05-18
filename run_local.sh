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
#   WORKERS_GPU0=8 WORKERS_GPU1=4 ./run_local.sh             # default MPS=auto
#   USE_MPS=1 CUDA_VISIBLE_DEVICES=0,1 ./run_local.sh        # preferred multi-GPU
#   USE_MPS=2 CUDA_VISIBLE_DEVICES=0,1 ./run_local.sh        # requires sudo daemon
#
# Multi-GPU verdict (consumer Blackwell, RTX 5090 + 5070):
#   USE_MPS=1 is preferred. It starts one user-mode MPS daemon per GPU at
#   /tmp/nvidia-mps-$USER-$GPUID and pins each worker to its own daemon.
#   No cross-GPU MPS routing ambiguity (which the shared sudo daemon at
#   /tmp/nvidia-mps suffers from), and we verified both GPUs at 100% peak
#   util under load. See BLACKWELL_NOTES.md for the full story.
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

# MPS modes:
#   USE_MPS=auto  detect /tmp/nvidia-mps; use it if alive (=2), else =0
#   USE_MPS=0     no MPS (workers serialize on each GPU; very slow)
#   USE_MPS=1     RECOMMENDED for multi-GPU. Script starts one user-mode
#                 daemon per visible GPU at /tmp/nvidia-mps-$USER-$GPUID;
#                 each worker is pinned to its GPU's daemon. Gives true
#                 multi-GPU execution with no MPS routing ambiguity.
#   USE_MPS=2     external sudo daemon at /tmp/nvidia-mps. Works but may
#                 route all "GPU 1" work to GPU 0 under load (one shared
#                 server per uid). Fine for single-GPU; suspect for multi.
USE_MPS=${USE_MPS:-auto}

# ZMQ heartbeat tuning (inherited if set in env.sh)
: "${WM_ZMQ_MASTER_HEARTBEAT:=50}"
: "${WM_ZMQ_WORKER_HEARTBEAT:=50}"
: "${WM_ZMQ_TIMEOUT_FACTOR:=100}"
export WM_ZMQ_MASTER_HEARTBEAT WM_ZMQ_WORKER_HEARTBEAT WM_ZMQ_TIMEOUT_FACTOR

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Use BASH_SOURCE (not readlink -f) so symlinks from a run dir into the
# repo resolve to the RUN DIR (symlink location), not the repo (symlink
# target). The run dir is what owns env.sh, west.cfg, west.h5, etc.
cd "$(dirname "${BASH_SOURCE[0]}")"
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
    # User-mode MPS: this script starts one daemon per visible GPU at
    # /tmp/nvidia-mps-$USER-$GPUID. Each worker picks the pipe dir for its
    # TARGET_GPU in the spawn loop below, so kernels route to a daemon that
    # only knows about that physical GPU (no cross-GPU MPS routing
    # ambiguity).
    #
    # Was previously hard-disabled with the comment "broken on Blackwell"
    # because we attributed the OverflowError-in-setStepCount to user-mode
    # MPS context creation. We now know that error is a GPU-side scalar
    # readback corruption fixed by the stepCount workaround in
    # runtime/gamd/runners.py, so user-mode MPS is back on the table.
    start_mps() {  # $1 gpuid  $2 thread%
        local pipe="/tmp/nvidia-mps-$USER-$1"
        local logd="/tmp/nvidia-log-$USER-$1"
        mkdir -p "$pipe" "$logd"
        # If a leftover daemon from a prior crashed run is still listening
        # on the pipe, reuse it (the `-d` invocation would fail with
        # "An instance of this daemon is already running"). Probe with a
        # cheap get_server_list query.
        if ! echo get_server_list \
                 | CUDA_MPS_PIPE_DIRECTORY="$pipe" nvidia-cuda-mps-control \
                   >/dev/null 2>&1; then
            CUDA_VISIBLE_DEVICES="$1" CUDA_MPS_PIPE_DIRECTORY="$pipe" \
                CUDA_MPS_LOG_DIRECTORY="$logd" nvidia-cuda-mps-control -d
            for _ in {1..10}; do [ -S "$pipe/control" ] && break; sleep 0.5; done
        fi
        echo "set_default_active_thread_percentage $2" \
            | CUDA_MPS_PIPE_DIRECTORY="$pipe" nvidia-cuda-mps-control >/dev/null
    }
    for gpuid in "${DEVICES[@]}"; do
        nw=${WORKERS_PER_GPU[$gpuid]}
        thread_pct=$(( nw > 0 ? 100 / nw : 100 ))
        echo "[run_local] starting user-mode MPS on GPU $gpuid (thread%=$thread_pct)"
        start_mps "$gpuid" "$thread_pct"
    done
    # Don't export a global CUDA_MPS_PIPE_DIRECTORY — workers set it per-GPU
    # in the spawn loop. Make sure no inherited value leaks in.
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY 2>/dev/null || true
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
    pkill -KILL -f "${PARGAMD_RUNTIME_DIR:-$WEST_SIM_ROOT/runtime}/gamdRunner" 2>/dev/null || true

    # Clear stale runtime files so the next launch starts clean.
    rm -f "$SERVER_INFO" /tmp/pargamd-cuda-init.lock 2>/dev/null || true

    # USE_MPS=1: stop our per-GPU user-mode daemons. USE_MPS=2: leave the
    # user-managed sudo daemon alone.
    if [ "$USE_MPS" = "1" ]; then
        for gpuid in "${DEVICES[@]}"; do
            local pipe="/tmp/nvidia-mps-$USER-$gpuid"
            [ -d "$pipe" ] || continue
            echo quit | CUDA_MPS_PIPE_DIRECTORY="$pipe" nvidia-cuda-mps-control 2>/dev/null || true
        done
    fi
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
          # Per-GPU MPS pipe routing: under USE_MPS=1 (per-GPU user-mode
          # daemons) and USE_MPS=2 with per-GPU sudo daemons at
          # /tmp/nvidia-mps-${gpuid}, point each worker at the daemon for
          # its physical GPU. Otherwise leave the global setting (shared
          # sudo daemon at /tmp/nvidia-mps, or no MPS).
          #
          # Per-GPU MPS daemons were started with CUDA_VISIBLE_DEVICES=$gpuid,
          # so each daemon exposes only ONE device, renumbered as device
          # index 0 in client space. Restrict the worker's CVD to match
          # and target device 0 (=this daemon's only physical GPU).
          # For the shared MPS=2 daemon (sees both GPUs), keep the global
          # CVD=0,1 and target device $gpuid as usual.
          if [ "$USE_MPS" = "1" ]; then
              export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-$USER-$gpuid"
              export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-$USER-$gpuid"
              # Daemon was started with CVD=$gpuid → exposes its 1 GPU as
              # device index 0 in client namespace. Client CVD must match
              # that renumbered index (NOT the host's physical $gpuid),
              # otherwise driver returns CUDA_ERROR_NO_DEVICE.
              export CUDA_VISIBLE_DEVICES="0"
              export TARGET_GPU="0"
          elif [ "$USE_MPS" = "2" ] && [ -S "/tmp/nvidia-mps-${gpuid}/control" ]; then
              export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${gpuid}"
              export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${gpuid}"
              export CUDA_VISIBLE_DEVICES="0"
              export TARGET_GPU="0"
          else
              export TARGET_GPU="$gpuid"
          fi
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
