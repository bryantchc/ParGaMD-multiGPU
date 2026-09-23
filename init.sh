#!/bin/bash


set -x
# Set up simulation environment


source ~/.bashrc

# 1. Hook Conda into this shell script
eval "$(conda shell.bash hook)"
conda init
conda  activate pargamd

source env.sh
: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
: "${PARGAMD_SYSTEM_DIR:?PARGAMD_SYSTEM_DIR is unset; check env.sh}"


# Clean up from previous/ failed runs
rm -rf seg_logs istates west.h5 *.log
mkdir -p seg_logs istates

# traj_segs is usually a SYMLINK into a scratch pool (see
# templates/env.sh.template). `rm -rf traj_segs` would delete the link and the
# following mkdir would silently recreate it as a real directory on whatever
# filesystem the run dir lives on -- typically the boot drive -- while orphaning
# the old data in the pool. So clear the link's TARGET and keep the link.
if [ -L traj_segs ]; then
    TRAJ_TARGET="$(readlink -f traj_segs)"
    case "$TRAJ_TARGET" in
        ""|"/"|"$HOME") echo "[init] refusing to clear traj_segs target '$TRAJ_TARGET'"; exit 1 ;;
    esac
    echo "[init] traj_segs -> $TRAJ_TARGET (preserving symlink, clearing contents)"
    mkdir -p "$TRAJ_TARGET"
    find "$TRAJ_TARGET" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
else
    rm -rf traj_segs
    : "${PARGAMD_SCRATCH_ROOT:=/scratch/pool}"
    RUN_NAME="$(basename "$WEST_SIM_ROOT")"
    if [ -d "$PARGAMD_SCRATCH_ROOT" ] && [ -w "$PARGAMD_SCRATCH_ROOT" ]; then
        TRAJ_TARGET="$PARGAMD_SCRATCH_ROOT/$RUN_NAME/traj_segs"
        mkdir -p "$TRAJ_TARGET"
        ln -sfn "$TRAJ_TARGET" traj_segs
        echo "[init] traj_segs -> $TRAJ_TARGET"
    else
        mkdir -p traj_segs
        echo "[init] WARNING: $PARGAMD_SCRATCH_ROOT not usable; traj_segs is LOCAL"
    fi
fi

BSTATE_FILE="$WEST_SIM_ROOT/bstates/bstates.txt"
N_BSTATES=$(grep -cE '^[[:space:]]*[^#[:space:]]' "$BSTATE_FILE" 2>/dev/null || true)
N_BSTATES=${N_BSTATES:-0}

if [ "$N_BSTATES" -le 1 ]; then
    # Single-structure run. Refresh bstate0's coords symlink to match the
    # current SYSTEM_NAME; without this, leftover stale coords in bstate0/ get
    # loaded against the (possibly new) topology and die on an atom-count
    # mismatch.
    mkdir -p "$WEST_SIM_ROOT/bstates/bstate0"
    ln -sfv "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.rst7" "$WEST_SIM_ROOT/bstates/bstate0/output_restart.rst7"
    SEGS_PER_STATE="${SEGS_PER_STATE:-5}"
else
    # Multi-structure run (e.g. built by westpa_scripts/make_bstates.py). Leave
    # the harvested basis states completely alone -- do NOT scaffold bstate0,
    # which would add a stray state that is not listed in bstates.txt. One
    # segment per state by default, since there are already many states.
    echo "[init] $N_BSTATES basis states listed in $BSTATE_FILE; leaving them untouched"
    SEGS_PER_STATE="${SEGS_PER_STATE:-1}"
fi

# Pre-flight the basis-state weights.
#
# WESTPA asserts abs(1 - sum(weights)) < EPS * (n_segments + n_active_bins) in
# sim_manager.report_bin_statistics -- for ~100 segments that is a budget of
# only ~2.6e-14. Weights written with too few significant digits overshoot it:
# 194 probabilities written as %.12e summed to 1.0000000000000273 (2.7e-14) and
# killed w_run at iteration 1.
#
# w_init does have its own renormalisation fallback ("Normalization check failed
# at w_init, explicitly renormalizing"), but in practice it did not persist --
# the error in west.h5 came back essentially unchanged. And the failure surfaces
# later, in w_run, as a bare AssertionError inside the BIN MAPPER, which points
# nowhere near the real cause. So check here, where the message can be useful.
#
# make_bstates.py writes exactly-normalised, shortest-round-trip weights;
# `make_bstates.py --refresh-weights` repairs an existing bstates/ in place
# without re-copying any checkpoints.
python - "$BSTATE_FILE" <<'PREFLIGHT'
import sys
import numpy as np
vals = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line and not line.startswith('#'):
        vals.append(float(line.split()[1]))
if not vals:
    sys.exit("[init] ERROR: no basis states found in %s" % sys.argv[1])
w = np.array(vals, dtype=np.float64)
err = abs(1.0 - w.sum())
tol = np.finfo(np.float64).eps * (len(w) + 32)
print("[init] bstate weights: n=%d  |1-sum|=%.3e  tolerance=%.3e" % (len(w), err, tol))
if err > tol:
    sys.exit("[init] ERROR: basis-state weights do not sum to 1 closely enough.\n"
             "[init] Repair with:  python westpa_scripts/make_bstates.py --refresh-weights")
PREFLIGHT

# Set pointer to bstate and tstate
BSTATE_ARGS="--bstate-file $BSTATE_FILE"
#TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstate.file"

# Run w_init
w_init \
  $BSTATE_ARGS \
  $TSTATE_ARGS \
  --segs-per-state "$SEGS_PER_STATE" \
  --work-manager=threads "$@"
