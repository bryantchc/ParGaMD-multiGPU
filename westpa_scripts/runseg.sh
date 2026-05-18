#!/bin/bash
set -x  # Print commands for debugging

# Source env.sh defensively so SYSTEM_NAME and conda are guaranteed to be
# set even when this script is invoked outside the run_local.sh chain.
if [ -n "$WEST_SIM_ROOT" ] && [ -f "$WEST_SIM_ROOT/env.sh" ]; then
    source "$WEST_SIM_ROOT/env.sh"
fi

##############################################################################
# 1) Rely on HPC to set CUDA_VISIBLE_DEVICES
##############################################################################
# echo "[DEBUG] HPC provided CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
# nvidia-smi

##############################################################################
# 2) Create and move into the simulation directory
##############################################################################
mkdir -pv "$WEST_CURRENT_SEG_DATA_REF"
cd "$WEST_CURRENT_SEG_DATA_REF" || exit 1

##############################################################################
# 3) Link necessary files (topology, coords, XML)
##############################################################################
# Honor $SYSTEM_NAME / $PARGAMD_{RUNTIME,SYSTEM}_DIR from env.sh (sourced
# above). Hard error if unset — silent fallbacks used to mask misconfigured
# swaps.
# We symlink the system-specific parm7/rst7 to GENERIC names inside the seg
# dir so input.xml can be system-agnostic (it always references topology.parm7
# and coordinates.rst7). To swap systems entirely, scaffold a new run dir
# with new_run.sh --system <other>.
: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
: "${PARGAMD_RUNTIME_DIR:?PARGAMD_RUNTIME_DIR is unset; check env.sh}"
: "${PARGAMD_SYSTEM_DIR:?PARGAMD_SYSTEM_DIR is unset; check env.sh}"
ln -sfv "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7" ./topology.parm7
ln -sfv "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.rst7"  ./coordinates.rst7
ln -sfv "$WEST_SIM_ROOT/gamd-restart.dat"          ./gamd-restart.dat
ln -sfv "$WEST_SIM_ROOT/input.xml"                 ./input.xml

# Fix XML output directory to current location
sed -i 's|<directory>.*</directory>|<directory>.</directory>|' input.xml

##############################################################################
# 4) Handle GaMD checkpoint for first vs. subsequent iteration, with retry
##############################################################################
# The Langevin integrator + GaMD bias is intrinsically stochastic. ~5-10% of
# segments hit a "Particle coordinate is NaN" within the first ~250 steps
# from a bad random-noise draw, even when the parent state is fine. Without
# retry, WESTPA's strict propagation policy kills the whole run on the first
# segment failure (probability of clean iteration over 24 walkers is ~0.13).
#
# We retry up to MAX_RETRIES, each time:
#   - Restoring the seed checkpoint locally (gamdRunner overwrites it on
#     failure with a NaN-state checkpoint, useless for retry).
#   - Bumping the integrator's <random-seed> in input.xml. Different seeds
#     are extremely unlikely to all produce NaN from the same parent state.
#   - Falling out of the loop on first success (non-empty output_restart.dcd).
GAMD_RUNNER="$PARGAMD_RUNTIME_DIR/gamdRunner"
GAMD_CHECKPOINT_SEED="$WEST_SIM_ROOT/gamd_restart.checkpoint"
if [ "$WEST_CURRENT_ITER" -eq 1 ]; then
    PARENT_CHECKPOINT="$GAMD_CHECKPOINT_SEED"
else
    PARENT_CHECKPOINT="$WEST_PARENT_DATA_REF/gamd_restart.checkpoint"
fi

MAX_RETRIES=3
SUCCESS=0
for attempt in $(seq 1 $MAX_RETRIES); do
    cp -L "$PARENT_CHECKPOINT" ./gamd_restart.checkpoint

    if [ "$attempt" -gt 1 ]; then
        # Vary the seed on retry. Mix $RANDOM with PID and attempt to avoid
        # collisions across the 4 concurrent workers. Bash $RANDOM is 16-bit;
        # we square it for a wider range that fits in OpenMM's int seed.
        SEED=$(( (RANDOM * RANDOM + $$ + attempt) % 2000000000 ))
        sed -i "s|<random-seed>[0-9-]*</random-seed>|<random-seed>${SEED}</random-seed>|" input.xml
        echo "[runseg] iter=$WEST_CURRENT_ITER seg=$WEST_CURRENT_SEG_ID attempt=$attempt with random-seed=$SEED" >&2
    fi

    # GPU pinning: TARGET_GPU is set in run_local.sh's per-worker spawn
    # (each worker subshell exports its assigned GPU). We forward it to
    # gamdRunner via -d so every worker's CUDA_VISIBLE_DEVICES can stay
    # unrestricted ("0,1"), giving the driver and shared MPS daemon a
    # consistent namespace across all worker processes.
    GAMD_GPU_FLAG=""
    if [ -n "${TARGET_GPU:-}" ]; then
        GAMD_GPU_FLAG="-d $TARGET_GPU"
    fi
    # NOTE: an earlier revision wrapped this invocation in a flock-based
    # CUDA-init serialization to mitigate cross-GPU context-creation
    # contention on Blackwell (the Context_setStepCount OverflowError).
    # Empirically the flock did NOT fix the underlying bug and just
    # added ~CUDA_INIT_HOLD_S × N_workers of overhead per iter on hardware
    # that didn't need it. The retry-skip below is the lighter, hardware-
    # neutral safety net we kept.
    GAMD_OUT="gamdRunner_attempt${attempt}.out"
    python "$GAMD_RUNNER" -p CUDA $GAMD_GPU_FLAG -r xml input.xml > "$GAMD_OUT" 2>&1

    # Blackwell multi-GPU short-circuit: this specific error is environmental
    # (concurrent CUDA contexts across GPUs corrupt loadCheckpoint deserialization)
    # and is byte-identical across retries — different random seeds don't help
    # because the failure is *before* integration starts. Burning two more
    # attempts just adds ~6 s of wasted lock-contended GPU time per frozen seg
    # and amplifies the cross-GPU concurrency pressure for everyone else.
    if grep -q "OverflowError.*Context_setStepCount" "$GAMD_OUT" 2>/dev/null; then
        echo "[runseg] iter=$WEST_CURRENT_ITER seg=$WEST_CURRENT_SEG_ID attempt=$attempt: Blackwell multi-GPU OverflowError; skipping remaining retries" >&2
        break
    fi

    # Stricter validity check than `-s`: a late-NaN gamdRunner can write a
    # DCD header (~96 bytes) plus a frame or two before crashing, producing
    # a non-empty but truncated file that downstream MDAnalysis can't read
    # ("OSError: opened empty file. No frames are saved", or wrong-shape
    # pcoord arrays).
    #
    # Read the DCD's NSET field (frame count) from header offset 8 via
    # pure-stdlib struct. System-agnostic — no atom-count or size threshold
    # baked in, works for any system that produces 100 frames per WE segment.
    # ~10ms per check (vs ~1s for an MDAnalysis Universe load).
    EXPECTED_FRAMES=100
    DCD_FRAMES=$(python -c "
import struct, sys
try:
    with open('output_restart.dcd','rb') as f:
        f.read(8)  # leading 4-byte block size + 'CORD' magic
        sys.stdout.write(str(struct.unpack('<i', f.read(4))[0]))
except Exception:
    sys.stdout.write('0')
" 2>/dev/null)
    if [ "${DCD_FRAMES:-0}" -ge "$EXPECTED_FRAMES" ]; then
        SUCCESS=1
        break
    fi
    echo "[runseg] iter=$WEST_CURRENT_ITER seg=$WEST_CURRENT_SEG_ID attempt=$attempt failed (DCD has ${DCD_FRAMES:-0} frames < $EXPECTED_FRAMES expected)" >&2
done

if [ "$SUCCESS" -ne 1 ]; then
    # Freeze-fallback: all $MAX_RETRIES attempts hit NaN, so the parent
    # state is deterministically NaN-prone (probably an atom on the edge
    # of an energy cliff that any forward integration pushes over). Rather
    # than killing the entire WE run via WESTPA's strict propagation policy,
    # we copy the parent's trajectory + checkpoint forward — the walker
    # "stays in place" this iteration and gets resampled next iteration.
    # Some sampling efficiency is lost; grep "FROZEN" in seg_logs/ to
    # identify and quantify these events for any final analysis.
    echo "[runseg] FROZEN iter=$WEST_CURRENT_ITER seg=$WEST_CURRENT_SEG_ID: $MAX_RETRIES retries all hit NaN. Walker stays in place." >&2
    cp -L "$PARENT_CHECKPOINT" ./gamd_restart.checkpoint
    if [ -s "$WEST_PARENT_DATA_REF/output_restart.dcd" ]; then
        # Iter > 1: parent has a 100-frame DCD from its own runseg.sh execution.
        cp -L "$WEST_PARENT_DATA_REF/output_restart.dcd" ./output_restart.dcd
    elif [ -f "$WEST_PARENT_DATA_REF/output_restart.rst7" ]; then
        # Iter 1: parent is the basis state (single-frame .rst7). Synthesize
        # a 100-frame DCD by replicating the rst7 coordinates so downstream
        # MDAnalysis pcoord computation produces 100 (identical) rows.
        python <<EOF
import MDAnalysis as mda
u = mda.Universe("topology.parm7", "$WEST_PARENT_DATA_REF/output_restart.rst7", format="INPCRD")
with mda.Writer("output_restart.dcd", u.atoms.n_atoms) as W:
    for _ in range(100):
        W.write(u.atoms)
EOF
    else
        echo "Error: parent state has neither DCD nor rst7 at $WEST_PARENT_DATA_REF; cannot freeze walker"
        ls -lh "$WEST_PARENT_DATA_REF" 2>/dev/null
        exit 1
    fi
    SUCCESS=1
fi

##############################################################################
# 6) Compute pcoord via $WEST_SIM_ROOT/cv_*.py modules
##############################################################################
# The shared-Universe dispatcher loads MDAnalysis once and runs every
# user-defined CV (cv_0.py, cv_1.py, ...) per frame. Edit those files in
# the run dir to change what's being sampled; pcoord_ndim in west.cfg
# must match the number of cv_*.py files.
REF_PDB="$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.pdb"
[ -f "$REF_PDB" ] || REF_PDB="-"
python "$WEST_SIM_ROOT/westpa_scripts/_pcoord_dispatch.py" \
    topology.parm7 output_restart.dcd "$REF_PDB" "$WEST_SIM_ROOT" \
    > "$WEST_PCOORD_RETURN"

if [ ! -s "$WEST_PCOORD_RETURN" ]; then
    echo "Error: pcoord computation produced empty output." >&2
    exit 1
fi

##############################################################################
# 8) Optional debugging
##############################################################################
#if [ -n "\$SEG_DEBUG" ]; then
#    echo "Preview of $WEST_PCOORD_RETURN:"
#    head -v "$WEST_PCOORD_RETURN"
#fi

echo "Segment run completed successfully."
exit 0
