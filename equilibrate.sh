#!/usr/bin/env bash
# equilibrate.sh — run Phase 1 GaMD equilibration in this run dir.
#
# Reads equilibration/input.xml (per-run, editable), produces
# equilibration/out/{gamd_restart.checkpoint, gamd-restart.dat}, then wires
# them into the run dir as $WEST_SIM_ROOT/{gamd_restart.checkpoint,gamd-restart.dat}.
#
# This is the equilibration half of new_run.sh, factored out so you can
# re-equilibrate without re-scaffolding (e.g. after editing
# equilibration/input.xml). Refuses to clobber an existing seed without
# --force.
#
# Wallclock: ~60 minutes on a 5090 for chignolin-sized systems.

set -euo pipefail

# Use BASH_SOURCE (not readlink -f) so the run dir's symlink → repo
# resolves to the run dir, not the repo.
cd "$(dirname "${BASH_SOURCE[0]}")"
WEST_SIM_ROOT="$PWD"
export WEST_SIM_ROOT
# shellcheck disable=SC1091
source env.sh
: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
: "${PARGAMD_RUNTIME_DIR:?PARGAMD_RUNTIME_DIR is unset; check env.sh}"
: "${PARGAMD_SYSTEM_DIR:?PARGAMD_SYSTEM_DIR is unset; check env.sh}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

EQ_DIR="$WEST_SIM_ROOT/equilibration"
EQ_OUT="$EQ_DIR/out"
SEED_CKPT="$EQ_OUT/gamd_restart.checkpoint"
SEED_DAT="$EQ_OUT/gamd-restart.dat"

if [ -s "$SEED_CKPT" ] && [ "$FORCE" -ne 1 ]; then
    echo "[equilibrate] seed checkpoint exists at $SEED_CKPT" >&2
    echo "[equilibrate] pass --force to redo equilibration (60 min)" >&2
    exit 0
fi

# Wire equilibration/'s topology + coords symlinks to the current SYSTEM_NAME.
# input.xml in $EQ_DIR is already system-agnostic (refers to topology.parm7
# and coordinates.rst7) so only these two symlinks change between systems.
mkdir -p "$EQ_DIR" "$EQ_OUT"
ln -sfn "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7" "$EQ_DIR/topology.parm7"
ln -sfn "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.rst7"  "$EQ_DIR/coordinates.rst7"

if [ ! -f "$EQ_DIR/input.xml" ]; then
    echo "[equilibrate] missing $EQ_DIR/input.xml — was this run dir scaffolded with new_run.sh?" >&2
    exit 1
fi

echo "[equilibrate] starting Phase 1 GaMD equilibration for SYSTEM_NAME=$SYSTEM_NAME"
echo "[equilibrate] wallclock ~60 min on a 5090…"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

(
    cd "$EQ_DIR"
    python "$PARGAMD_RUNTIME_DIR/gamdRunner_init" -p CUDA xml input.xml \
        | tee equilibration.log
)

if [ ! -s "$SEED_CKPT" ]; then
    echo "[equilibrate] FAILED — $SEED_CKPT missing/empty" >&2
    echo "[equilibrate] inspect $EQ_DIR/equilibration.log" >&2
    exit 1
fi

# Wire the seed into the run dir at the paths runseg.sh / bench expect.
ln -sfn equilibration/out/gamd_restart.checkpoint "$WEST_SIM_ROOT/gamd_restart.checkpoint"
ln -sfn equilibration/out/gamd-restart.dat        "$WEST_SIM_ROOT/gamd-restart.dat"

echo "[equilibrate] done. Seed:"
ls -L "$WEST_SIM_ROOT/gamd_restart.checkpoint" "$WEST_SIM_ROOT/gamd-restart.dat"
