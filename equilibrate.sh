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
# Flags:
#   --force    redo equilibration even though a seed checkpoint exists
#   --resume   CONTINUE an interrupted equilibration from its checkpoint
#
# --resume exists because an equilibration is many hours and the machine will
# sometimes need rebooting. gamdRunner_init takes -r, which loadCheckpoint()s
# equilibration/out/gamd_restart.checkpoint and reads stepCount back off the
# integrator. OpenMM checkpoints carry integrator global variables, so the
# accumulated GaMD statistics (Vmax, Vmin, Vavg, sigmaV, k0, stepCount) are all
# restored -- it is a genuine resume, not a restart. This only continues to the
# original nstlim while extension-steps is 0, which is the case for the
# equilibration templates; a non-zero value switches gamdRunner into extension
# mode (that is what WE segments use) and --resume would then EXTEND rather
# than continue, so it is refused below.
#
# Stopping safely first matters: the gamd runtime installs no signal handlers
# and saveCheckpoint() writes in place, so a signal during a checkpoint write
# truncates it. Use ./stop_gamd.sh <run_dir>, then
# ./verify_gamd_checkpoint.sh <run_dir> before relying on it.
#
# Wallclock: ~60 minutes on a 5090 for chignolin-sized systems. A solvated
# nucleoprotein complex of ~270k atoms takes ~16 h for the 41.5M-step protocol.

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
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --force)  FORCE=1 ;;
        --resume) RESUME=1 ;;
        *) echo "[equilibrate] unknown argument: $arg" >&2
           echo "[equilibrate] usage: equilibrate.sh [--force] [--resume]" >&2
           exit 1 ;;
    esac
done
if [ "$FORCE" -eq 1 ] && [ "$RESUME" -eq 1 ]; then
    echo "[equilibrate] --force and --resume are mutually exclusive" >&2
    exit 1
fi

EQ_DIR="$WEST_SIM_ROOT/equilibration"
EQ_OUT="$EQ_DIR/out"
SEED_CKPT="$EQ_OUT/gamd_restart.checkpoint"
SEED_DAT="$EQ_OUT/gamd-restart.dat"

# Wire equilibration/'s topology + coords symlinks to the current SYSTEM_NAME.
# input.xml in $EQ_DIR is already system-agnostic (refers to topology.parm7
# and coordinates.rst7) so only these two symlinks change between systems.
mkdir -p "$EQ_DIR" "$EQ_OUT"
ln -sfn "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7" "$EQ_DIR/topology.parm7"
ln -sfn "$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.rst7"  "$EQ_DIR/coordinates.rst7"

# ---------------------------------------------------------------------------
# Run equilibration (skip if seed exists and no --force)
# ---------------------------------------------------------------------------
if [ "$RESUME" -eq 1 ]; then
    if [ ! -s "$SEED_CKPT" ]; then
        echo "[equilibrate] --resume needs an existing $SEED_CKPT" >&2
        exit 1
    fi
    # extension-steps != 0 turns gamdRunner's restart path into an EXTENSION
    # run (nstlim := stepCount + extension_steps), which is not what --resume
    # means. Refuse rather than silently run the wrong length.
    EXT=$(sed -n 's:.*<extension-steps>\([0-9]*\)</extension-steps>.*:\1:p' \
          "$EQ_DIR/input.xml" | head -1)
    if [ -n "${EXT:-}" ] && [ "$EXT" -ne 0 ]; then
        echo "[equilibrate] refusing --resume: extension-steps=$EXT in input.xml" >&2
        echo "[equilibrate] with a non-zero value gamdRunner EXTENDS instead of continuing." >&2
        exit 1
    fi
    echo "[equilibrate] RESUMING equilibration for SYSTEM_NAME=$SYSTEM_NAME"
    echo "[equilibrate] checkpoint: $SEED_CKPT"
    : "${CUDA_VISIBLE_DEVICES:=0}"
    export CUDA_VISIBLE_DEVICES
    (
        cd "$EQ_DIR"
        python "$PARGAMD_RUNTIME_DIR/gamdRunner_init" -r -p CUDA xml input.xml \
            | tee -a equilibration.log
    )
    if [ ! -s "$SEED_CKPT" ]; then
        echo "[equilibrate] FAILED after resume — $SEED_CKPT missing/empty" >&2
        exit 1
    fi
elif [ -s "$SEED_CKPT" ] && [ "$FORCE" -ne 1 ]; then
    echo "[equilibrate] seed checkpoint exists at $SEED_CKPT (use --force to redo)"
    echo "[equilibrate] proceeding to pcoord computation only"
else
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
fi

# Wire the seed into the run dir at the paths runseg.sh / bench expect.
ln -sfn equilibration/out/gamd_restart.checkpoint "$WEST_SIM_ROOT/gamd_restart.checkpoint"
ln -sfn equilibration/out/gamd-restart.dat        "$WEST_SIM_ROOT/gamd-restart.dat"

echo "[equilibrate] seed wired:"
ls -L "$WEST_SIM_ROOT/gamd_restart.checkpoint" "$WEST_SIM_ROOT/gamd-restart.dat"

# ---------------------------------------------------------------------------
# Pre-compute basis-state pcoord and write bstates.txt
# ---------------------------------------------------------------------------
# The basis state is the system's input .rst7 (not the equilibrated structure
# — WESTPA branches walkers from this geometry, biased forward by the seed
# GaMD checkpoint). We compute its pcoord here so bstates.txt is ready for
# w_init the moment ./run_local.sh runs. Folded in from the old setup.sh.
echo "[equilibrate] computing basis-state pcoord…"
mkdir -p "$WEST_SIM_ROOT/bstates/bstate0"
ln -sfn "../../system/${SYSTEM_NAME}.rst7" \
    "$WEST_SIM_ROOT/bstates/bstate0/output_restart.rst7"

PCOORD_TMP=$(mktemp)
WEST_SIM_ROOT="$WEST_SIM_ROOT" \
WEST_STRUCT_DATA_REF="$WEST_SIM_ROOT/bstates/bstate0" \
WEST_PCOORD_RETURN="$PCOORD_TMP" \
    bash "$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh"

if [ -s "$PCOORD_TMP" ]; then
    read -r PC0 PC1 < "$PCOORD_TMP"
    printf "0 1 bstate0 %s %s\n" "$PC0" "$PC1" > "$WEST_SIM_ROOT/bstates/bstates.txt"
    echo "[equilibrate] bstates.txt: $(cat "$WEST_SIM_ROOT/bstates/bstates.txt")"
else
    echo "[equilibrate] WARN: pcoord computation produced no output — bstates.txt left unchanged" >&2
fi
rm -f "$PCOORD_TMP"

echo "[equilibrate] done."
