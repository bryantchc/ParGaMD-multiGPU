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
rm -rf traj_segs seg_logs istates west.h5 *.log  # added log removal
mkdir   seg_logs traj_segs istates

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
