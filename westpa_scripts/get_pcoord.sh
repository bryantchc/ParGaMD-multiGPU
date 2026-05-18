#!/bin/bash
##############################################################################
# MDAnalysis-based pcoord driver for WESTPA. Computes:
#   - RMSD of CA atoms (mass-weighted, best-fit alignment)
#   - Radius of gyration of CA atoms (mass-weighted)
#
# Called by WESTPA for both basis states (single-frame .rst7) and any
# externally-supplied structures. Per-segment pcoords are produced
# directly by runseg.sh, so this script primarily serves bstates.
#
# Inputs (in $WEST_STRUCT_DATA_REF):
#   - output_restart.dcd  -> trajectory (preferred if present)
#   - output_restart.rst7 -> single-frame restart (basis state fallback)
#
# Topology and reference structure are pulled from $PARGAMD_SYSTEM_DIR.
##############################################################################

# Source env.sh defensively so manual invocations (e.g. pre-flight pcoord
# checks before w_init) get SYSTEM_NAME and the conda env without the user
# having to source it themselves. Under WESTPA the env is already inherited
# from run_local.sh, but sourcing again is idempotent.
if [ -n "$WEST_SIM_ROOT" ] && [ -f "$WEST_SIM_ROOT/env.sh" ]; then
    source "$WEST_SIM_ROOT/env.sh"
fi

# 1. Optionally enable debugging
if [ -n "$SEG_DEBUG" ]; then
    set -x
    env | sort
fi

# 2. Move to the segment directory
cd "$WEST_STRUCT_DATA_REF" || {
    echo "Error: Could not cd to \$WEST_CURRENT_SEG_DATA_REF=$WEST_STRUCT_DATA_REF" >&2
    exit 1
}

# 3. Decide which trajectory file to feed the dispatcher. Prefer the
#    segment DCD; fall back to the basis-state .rst7.
: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
: "${PARGAMD_SYSTEM_DIR:?PARGAMD_SYSTEM_DIR is unset; check env.sh}"

if [ -f output_restart.dcd ]; then
    TRAJ=output_restart.dcd
elif [ -f output_restart.rst7 ]; then
    TRAJ=output_restart.rst7
else
    echo "Error: neither output_restart.dcd nor output_restart.rst7 in $PWD" >&2
    exit 1
fi

TOPO="$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7"
REF_PDB="$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.pdb"
[ -f "$REF_PDB" ] || REF_PDB="-"

# 4. Run the shared-Universe dispatcher; it reads cv_*.py from $WEST_SIM_ROOT.
python "$WEST_SIM_ROOT/westpa_scripts/_pcoord_dispatch.py" \
    "$TOPO" "$TRAJ" "$REF_PDB" "$WEST_SIM_ROOT" \
    > "$WEST_PCOORD_RETURN"

if [ ! -s "$WEST_PCOORD_RETURN" ]; then
    echo "Error: pcoord computation produced empty output." >&2
    exit 1
fi

if [ -n "$SEG_DEBUG" ]; then
    echo "Preview of \$WEST_PCOORD_RETURN:"
    head -v "$WEST_PCOORD_RETURN"
fi
exit 0
