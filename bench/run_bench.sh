#!/usr/bin/env bash
# bench/run_bench.sh — sweep (WORKERS_GPU0, WORKERS_GPU1) distributions and
# report aggregate GaMD throughput for each. Use this AFTER new_run.sh (or
# equilibrate.sh) has produced $WEST_SIM_ROOT/gamd_restart.checkpoint,
# before launching a long production run, to pick the worker split that
# maximizes ns/day for your system on this hardware.
#
# Each worker runs gamdRunner directly (no WESTPA, no walker tree) against
# its own scratch dir seeded with the same checkpoint run_local.sh's iter 1
# would use. The seed checkpoint is restored before each segment so every
# worker does identical work — that makes the throughput comparison fair.
#
# Knobs (env):
#   BENCH_CONFIGS         space-separated "W0,W1" tuples to sweep
#                         default: a reasonable sweep on a 5090+5070 box
#   BENCH_SEGS_PER_WORKER N gamdRunner invocations per worker per config
#                         default: 3 (≈1 min per config on the 5090)
#   BENCH_SEG_NS          per-segment simulated ns (must match input.xml's
#                         extension-steps × dt). default: 0.1 (= 100 ps)
#   USE_MPS               1 (per-GPU user-mode, default) or 0 (no MPS)
#
# Output: a markdown-style table to stdout AND to bench/run-<ts>/results.txt.

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
# BASH_SOURCE (not readlink -f) so symlink path resolves to the run dir's
# bench/ symlink, not the repo's actual bench/. We want the run dir as cwd.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
WEST_SIM_ROOT="$PWD"
export WEST_SIM_ROOT
# shellcheck disable=SC1091
source env.sh
: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
: "${PARGAMD_RUNTIME_DIR:?PARGAMD_RUNTIME_DIR is unset; check env.sh}"
: "${PARGAMD_SYSTEM_DIR:?PARGAMD_SYSTEM_DIR is unset; check env.sh}"
[ -f "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7" ] \
    || { echo "[bench] no $PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7" >&2; exit 1; }
[ -f "gamd_restart.checkpoint" ] \
    || { echo "[bench] no gamd_restart.checkpoint in $PWD — run ./equilibrate.sh first" >&2; exit 1; }

CONFIGS=${BENCH_CONFIGS:-"4,0 8,0 12,0 16,0 0,4 4,4 6,4 8,4 12,4 12,6 6,6"}
SEGS_PER_WORKER=${BENCH_SEGS_PER_WORKER:-3}
SEG_NS=${BENCH_SEG_NS:-0.1}
USE_MPS=${USE_MPS:-1}

BENCH_DIR="bench/run-$(date +%s)"
mkdir -p "$BENCH_DIR"
RESULTS="$BENCH_DIR/results.txt"

# Single-threaded per worker — same as run_local.sh
export OPENMM_CPU_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[bench] sim root:           $WEST_SIM_ROOT"
echo "[bench] system:             $SYSTEM_NAME"
echo "[bench] segs per worker:    $SEGS_PER_WORKER"
echo "[bench] ns per segment:     $SEG_NS"
echo "[bench] configs to sweep:   $CONFIGS"
echo "[bench] MPS mode:           $USE_MPS"
echo "[bench] scratch + results:  $BENCH_DIR"
echo

# ---------------------------------------------------------------------------
# MPS daemon lifecycle (per-GPU user-mode, same recipe as run_local.sh)
# ---------------------------------------------------------------------------
start_mps() { # $1 gpuid
    local pipe="/tmp/nvidia-mps-$USER-$1" logd="/tmp/nvidia-log-$USER-$1"
    mkdir -p "$pipe" "$logd"
    CUDA_VISIBLE_DEVICES="$1" CUDA_MPS_PIPE_DIRECTORY="$pipe" \
        CUDA_MPS_LOG_DIRECTORY="$logd" nvidia-cuda-mps-control -d
    for _ in {1..10}; do [ -S "$pipe/control" ] && break; sleep 0.5; done
}
stop_all_mps() {
    for g in 0 1; do
        local pipe="/tmp/nvidia-mps-$USER-$g"
        [ -d "$pipe" ] || continue
        echo quit | CUDA_MPS_PIPE_DIRECTORY="$pipe" nvidia-cuda-mps-control 2>/dev/null || true
    done
}
set_thread_pct() { # $1 gpuid  $2 pct
    echo "set_default_active_thread_percentage $2" \
        | CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-$USER-$1" \
            nvidia-cuda-mps-control >/dev/null
}

# Clean up daemons + any straggler bench gamdRunners on exit
cleanup_bench() {
    [ "$USE_MPS" = "1" ] && stop_all_mps
    pkill -KILL -f "$BENCH_DIR/" 2>/dev/null || true
}
trap cleanup_bench EXIT INT TERM

if [ "$USE_MPS" = "1" ]; then
    for g in 0 1; do start_mps "$g"; done
fi

# ---------------------------------------------------------------------------
# Per-worker scratch + driver
# ---------------------------------------------------------------------------
seed_worker_dir() { # $1 dir
    local d="$1"
    mkdir -p "$d"
    ln -sf "$PARGAMD_SYSTEM_DIR/$SYSTEM_NAME.parm7" "$d/topology.parm7"
    ln -sf "$PARGAMD_SYSTEM_DIR/$SYSTEM_NAME.rst7"  "$d/coordinates.rst7"
    ln -sf "$WEST_SIM_ROOT/gamd-restart.dat"        "$d/gamd-restart.dat"
    cp    "$WEST_SIM_ROOT/input.xml"                "$d/input.xml"
    # Point gamdRunner's output dir at the scratch dir itself.
    sed -i 's|<directory>.*</directory>|<directory>.</directory>|' "$d/input.xml"
}

# Runs in a subshell ( ... ) & — exit status surfaces as wait's $?.
# Writes its own elapsed seconds to .elapsed in $1 so the aggregator can
# compute per-worker throughput (the "WE keeps every worker fed" upper
# bound) in addition to the wall-clock-to-last-worker metric.
run_worker() { # $1 dir  $2 mps_pipe ("" if no MPS)
    cd "$1"
    if [ -n "$2" ]; then
        export CUDA_MPS_PIPE_DIRECTORY="$2"
        export CUDA_MPS_LOG_DIRECTORY="${2/nvidia-mps/nvidia-log}"
        # Daemon exposes its 1 GPU as device index 0 in client namespace.
        export CUDA_VISIBLE_DEVICES=0
    fi
    local t0=$SECONDS
    for ((s=1; s<=SEGS_PER_WORKER; s++)); do
        # Fresh seed each segment → identical work per segment.
        cp "$WEST_SIM_ROOT/gamd_restart.checkpoint" \
           gamd_restart.checkpoint
        if ! python "$PARGAMD_RUNTIME_DIR/gamdRunner" \
                -p CUDA -d 0 -r xml input.xml > "seg${s}.out" 2>&1; then
            echo "FAIL seg=$s dir=$1" >&2
            return 1
        fi
        # gamd.log empty = workaround/fallback path → not a real run
        [ -s gamd.log ] || { echo "FAIL empty-gamd.log seg=$s dir=$1" >&2; return 1; }
    done
    echo $((SECONDS - t0)) > .elapsed
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
# wall_ns_day:    aggregate from total simulated ns / wall-to-last-worker
#                 (what you'd see if WE could NOT keep workers fed past
#                 their assigned segments — pessimistic / lower bound).
# fed_ns_day:     sum of per-worker (own_segs × seg_ns / own_elapsed),
#                 i.e. each worker's intrinsic rate × N workers (what WE
#                 production should approach if it keeps every worker
#                 fed — optimistic / upper bound).
# Real production lands between these, typically near fed_ns_day.
printf "| %-6s | %4s | %4s | %5s | %5s | %12s | %12s | %4s |\n" \
    config W0 W1 g0_s g1_s "wall_ns_day" "fed_ns_day" "ok" | tee "$RESULTS"
printf "|--------|------|------|-------|-------|--------------|--------------|------|\n" \
    | tee -a "$RESULTS"

for cfg in $CONFIGS; do
    W0=${cfg%,*}; W1=${cfg#*,}
    TOTAL=$((W0+W1))
    [ "$TOTAL" -eq 0 ] && continue
    [ "$W0" -gt 0 ] || [ "$W1" -gt 0 ] || continue

    # Per-GPU MPS thread share = 100% / (workers on that GPU)
    if [ "$USE_MPS" = "1" ]; then
        [ "$W0" -gt 0 ] && set_thread_pct 0 $((100/W0))
        [ "$W1" -gt 0 ] && set_thread_pct 1 $((100/W1))
    fi

    cdir="$BENCH_DIR/${W0}_${W1}"
    rm -rf "$cdir"; mkdir -p "$cdir"
    for ((w=1; w<=TOTAL; w++)); do seed_worker_dir "$cdir/w$w"; done

    PIPE0="/tmp/nvidia-mps-$USER-0"
    PIPE1="/tmp/nvidia-mps-$USER-1"
    [ "$USE_MPS" = "1" ] || { PIPE0=""; PIPE1=""; }

    pids=(); wi=0; OK=0
    for ((w=1; w<=W0; w++)); do
        wi=$((wi+1)); ( run_worker "$cdir/w$wi" "$PIPE0" ) & pids+=($!)
    done
    for ((w=1; w<=W1; w++)); do
        wi=$((wi+1)); ( run_worker "$cdir/w$wi" "$PIPE1" ) & pids+=($!)
    done

    start=$SECONDS
    for p in "${pids[@]}"; do wait "$p" && OK=$((OK+1)) || true; done
    wall=$((SECONDS - start))
    [ "$wall" -lt 1 ] && wall=1

    # Per-GPU max elapsed (= when last worker on that GPU finished). Shows
    # the GPU 0 vs GPU 1 finish-time gap that drives the wall_ns_day vs
    # fed_ns_day delta.
    g0_max=0; g1_max=0; fed_ns_day_sum=0
    wi=0
    for ((w=1; w<=W0; w++)); do
        wi=$((wi+1)); el=$(cat "$cdir/w$wi/.elapsed" 2>/dev/null || echo 0)
        [ "$el" -gt "$g0_max" ] && g0_max=$el
        [ "$el" -gt 0 ] && fed_ns_day_sum=$(awk "BEGIN{print $fed_ns_day_sum + $SEGS_PER_WORKER * $SEG_NS * 86400 / $el}")
    done
    for ((w=1; w<=W1; w++)); do
        wi=$((wi+1)); el=$(cat "$cdir/w$wi/.elapsed" 2>/dev/null || echo 0)
        [ "$el" -gt "$g1_max" ] && g1_max=$el
        [ "$el" -gt 0 ] && fed_ns_day_sum=$(awk "BEGIN{print $fed_ns_day_sum + $SEGS_PER_WORKER * $SEG_NS * 86400 / $el}")
    done

    agg_ns_done=$(awk "BEGIN{print $OK * $SEGS_PER_WORKER * $SEG_NS}")
    wall_nsday=$(awk "BEGIN{printf \"%.1f\", $agg_ns_done * 86400 / $wall}")
    fed_nsday=$(awk "BEGIN{printf \"%.1f\", $fed_ns_day_sum}")

    printf "| %-6s | %4d | %4d | %5d | %5d | %12s | %12s | %d/%d |\n" \
        "$cfg" "$W0" "$W1" "$g0_max" "$g1_max" "$wall_nsday" "$fed_nsday" "$OK" "$TOTAL" \
        | tee -a "$RESULTS"
done

echo
echo "[bench] full table also in $RESULTS"
