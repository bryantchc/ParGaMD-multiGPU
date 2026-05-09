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
# Honor $SYSTEM_NAME from env.sh (sourced above). Hard error if unset —
# a silent chignolin fallback used to mask misconfigured swaps.
# We symlink the system-specific parm7/rst7 to GENERIC names inside the seg
# dir so input.xml can be system-agnostic (it always references topology.parm7
# and coordinates.rst7). To swap systems: drop new <name>.{parm7,rst7,pdb}
# into common_files/ and change SYSTEM_NAME in env.sh.
: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
ln -sfv "$WEST_SIM_ROOT/common_files/${SYSTEM_NAME}.parm7" ./topology.parm7
ln -sfv "$WEST_SIM_ROOT/common_files/${SYSTEM_NAME}.rst7" ./coordinates.rst7
ln -sfv "$WEST_SIM_ROOT/common_files/gamd-restart.dat" .
ln -sfv "$WEST_SIM_ROOT/common_files/input.xml" .

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
GAMD_RUNNER="$WEST_SIM_ROOT/common_files/gamdRunner"
GAMD_CHECKPOINT_SEED="$WEST_SIM_ROOT/common_files/gamd_restart.checkpoint"
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

    # If TARGET_GPU is set (by run_local.sh's per-worker spawn), pass the
    # device index to gamdRunner directly via -d. This keeps every worker's
    # CUDA_VISIBLE_DEVICES unrestricted (e.g. "0,1") so the driver and any
    # shared MPS daemon see a unified namespace across all worker processes,
    # while still routing each segment to its assigned physical GPU.
    GAMD_GPU_FLAG=""
    if [ -n "${TARGET_GPU:-}" ]; then
        GAMD_GPU_FLAG="-d $TARGET_GPU"
    fi
    # CUDA context-creation serialization. On Blackwell with concurrent
    # gamdRunner processes spanning multiple GPUs, simultaneous CUDA context
    # init triggers OverflowError in Context_setStepCount (cousin to the
    # CUDA_ERROR_NOT_FOUND that REST2 sees and mitigates with launch
    # staggering). We hold a global flock for ~CUDA_INIT_HOLD_S seconds
    # while gamdRunner does its context init in the background, then
    # release. The simulation itself continues in parallel (lock is held
    # only for the init window), so steady-state throughput is preserved;
    # the lock just spaces out the simultaneous context creations.
    : "${CUDA_INIT_HOLD_S:=1.0}"
    : "${CUDA_INIT_LOCK:=/tmp/pargamd-cuda-init.lock}"
    GAMD_OUT="gamdRunner_attempt${attempt}.out"
    exec {LOCKFD}>"$CUDA_INIT_LOCK"
    flock -x "$LOCKFD"
    python "$GAMD_RUNNER" -p CUDA $GAMD_GPU_FLAG -r xml input.xml > "$GAMD_OUT" 2>&1 &
    GAMD_PID=$!
    sleep "$CUDA_INIT_HOLD_S"
    flock -u "$LOCKFD"
    exec {LOCKFD}>&-
    # gamdRunner is now past context init; wait for it to finish the rest
    # of the simulation outside the critical section.
    wait "$GAMD_PID"

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
# 6) Post-processing with MDAnalysis: compute RMSD and Rg
##############################################################################
RMSD_FILE="rmsd_ca.xvg"
RG_FILE="rg_ca.xvg"

# Create a Python script on-the-fly to run MDAnalysis
cat << EOF > mdanalysis_rmsd_rg.py
#!/usr/bin/env python

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np

# Load the system. topology.parm7 is a per-segment symlink set up at the top
# of runseg.sh; reference PDB is system-specific via \$SYSTEM_NAME.
u = mda.Universe("topology.parm7", "output_restart.dcd")
ref = mda.Universe("${WEST_SIM_ROOT}/common_files/${SYSTEM_NAME}.pdb")

# Select only CA atoms
mobile_ca = u.select_atoms("name CA")
ref_ca    = ref.select_atoms("name CA")

rmsd_list = []
rg_list   = []

for ts in u.trajectory:
    # mass-weighted best-fit RMSD
    # 'weights=mobile_ca.masses' ensures mass weighting
    # center=True and superposition=True => do best-fit alignment
    rmsd_value = rms.rmsd(
        mobile_ca.positions,
        ref_ca.positions,
        center=True,
        superposition=True,
        weights=mobile_ca.masses
    )
    # Radius of gyration (all CA positions)
    rg_value = mobile_ca.radius_of_gyration()
    rmsd_list.append(rmsd_value)
    rg_list.append(rg_value)

# Write out data in a format similar to cpptraj .xvg
with open("${RMSD_FILE}", "w") as f:
    f.write("# frame RMSD_CA(Angstrom)\n")
    for i, val in enumerate(rmsd_list):
        f.write(f"{i} {val}\n")

with open("${RG_FILE}", "w") as f:
    f.write("# frame Rg_CA(Angstrom)\n")
    for i, val in enumerate(rg_list):
        f.write(f"{i} {val}\n")
EOF

# Run the Python script
python mdanalysis_rmsd_rg.py
if [ $? -ne 0 ]; then
    echo "Error: MDAnalysis RMSD/Rg calculation failed."
    exit 1
fi

##############################################################################
# 7) Write final RMSD and Rg data to $WEST_PCOORD_RETURN
##############################################################################
if [ -f "$RMSD_FILE" ] && [ -f "$RG_FILE" ]; then
    > "$WEST_PCOORD_RETURN"
    # Extract second column from lines >1 of each file, then paste them
    paste <(awk 'NR>1 {print $2}' "$RMSD_FILE") \
          <(awk 'NR>1 {print $2}' "$RG_FILE") >> "$WEST_PCOORD_RETURN"
else
    echo "Error: Missing $RMSD_FILE or $RG_FILE"
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
