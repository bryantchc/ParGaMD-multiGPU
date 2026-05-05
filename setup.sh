#!/usr/bin/env bash
# One-shot system setup: equilibration + seed wiring + pcoord pre-compute.
#
# Usage:
#   ./setup.sh <name>           # full setup for common_files/<name>.{parm7,rst7,pdb}
#   ./setup.sh <name> --force   # re-run even if a seed checkpoint already exists
#
# Performs WORKSTATION.md steps 3, 4, 6 (and writes SYSTEM_NAME into env.sh).
# Steps deliberately NOT automated:
#   - Step 7 (west.cfg bin tuning) — science decision, requires human judgment
#   - Step 8 (wiping west.h5)      — destructive, hard rule from project brief
#
# Wallclock: ~60 minutes (Phase 1 GaMD equilibration dominates).

set -euo pipefail

SYS_NAME=${1:-}
FORCE=0
[ "${2:-}" = "--force" ] && FORCE=1

if [ -z "$SYS_NAME" ]; then
    echo "Usage: $0 <name> [--force]" >&2
    echo "  Expects common_files/<name>.{parm7,rst7,pdb} to exist." >&2
    exit 1
fi

cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
for ext in parm7 rst7 pdb; do
    f="common_files/${SYS_NAME}.${ext}"
    if [ ! -f "$f" ]; then
        echo "[setup] missing $f — drop your system files into common_files/ first" >&2
        exit 1
    fi
done

if [ -f west.h5 ] && [ "$FORCE" -ne 1 ]; then
    echo "[setup] west.h5 exists; refusing to re-setup over a live run." >&2
    echo "        Move it aside (mv west.h5 traj_segs seg_logs istates ...) then retry," >&2
    echo "        or pass --force if you really mean it." >&2
    exit 1
fi

CKPT="common_files/equilibration/out/gamd_restart.checkpoint"
if [ -f "$CKPT" ] && [ "$FORCE" -ne 1 ]; then
    # Check whether the existing checkpoint is for this same system. The
    # equilibration's topology.parm7 symlink records what was last equilibrated.
    cur_top=$(readlink common_files/equilibration/topology.parm7 2>/dev/null || echo "")
    if [ "$cur_top" = "../${SYS_NAME}.parm7" ]; then
        echo "[setup] seed checkpoint already exists for $SYS_NAME; skipping equilibration." >&2
        echo "        Pass --force to redo it (60 min)." >&2
        SKIP_EQUIL=1
    else
        echo "[setup] $CKPT exists but was built from $cur_top, not ${SYS_NAME}.parm7." >&2
        echo "        Pass --force to overwrite, or move the existing seed aside first." >&2
        exit 1
    fi
fi
SKIP_EQUIL=${SKIP_EQUIL:-0}

# ---------------------------------------------------------------------------
# 1. Set SYSTEM_NAME in env.sh (step 2 of the manual recipe)
# ---------------------------------------------------------------------------
sed -i "s|SYSTEM_NAME:-[A-Za-z0-9_-]*|SYSTEM_NAME:-${SYS_NAME}|" env.sh
echo "[setup] env.sh now exports SYSTEM_NAME=${SYS_NAME}"

# Source env.sh so the rest of this script (and get_pcoord.sh below) sees it.
# shellcheck disable=SC1091
source env.sh

# ---------------------------------------------------------------------------
# 2. Phase 1 GaMD equilibration (step 3)  — ~60 min on a 5090
# ---------------------------------------------------------------------------
if [ "$SKIP_EQUIL" -ne 1 ]; then
    echo "[setup] Phase 1 GaMD equilibration starting (~60 min)…"
    pushd common_files/equilibration >/dev/null
    ln -sfn "../${SYS_NAME}.parm7" topology.parm7
    ln -sfn "../${SYS_NAME}.rst7"  coordinates.rst7
    : "${CUDA_VISIBLE_DEVICES:=0}"
    export CUDA_VISIBLE_DEVICES
    python ../gamdRunner_init -p CUDA xml input.xml | tee equilibration.log
    popd >/dev/null

    if [ ! -s "$CKPT" ]; then
        echo "[setup] equilibration finished but $CKPT is missing or empty." >&2
        echo "        Inspect common_files/equilibration/equilibration.log." >&2
        exit 1
    fi
    echo "[setup] equilibration done; checkpoint at $CKPT"
fi

# ---------------------------------------------------------------------------
# 3. Wire seed checkpoint symlinks (step 4)
# ---------------------------------------------------------------------------
pushd common_files >/dev/null
ln -sfn equilibration/out/gamd_restart.checkpoint gamd_restart.checkpoint
ln -sfn equilibration/out/gamd-restart.dat        gamd-restart.dat
popd >/dev/null
echo "[setup] seed symlinks wired:"
ls -L common_files/gamd_restart.checkpoint common_files/gamd-restart.dat

# ---------------------------------------------------------------------------
# 4. Recompute basis-state pcoord and write bstates.txt (step 6)
# ---------------------------------------------------------------------------
# bstate0/output_restart.rst7 normally is refreshed by init.sh, but the
# pcoord pre-flight runs before that, so refresh it here too.
ln -sfn "../../common_files/${SYS_NAME}.rst7" bstates/bstate0/output_restart.rst7

PCOORD_TMP=$(mktemp)
WEST_SIM_ROOT="$ROOT" \
WEST_STRUCT_DATA_REF="$ROOT/bstates/bstate0" \
WEST_PCOORD_RETURN="$PCOORD_TMP" \
    bash westpa_scripts/get_pcoord.sh

if [ ! -s "$PCOORD_TMP" ]; then
    echo "[setup] pcoord computation produced no output — check the error above." >&2
    exit 1
fi

# bstates.txt format: <state_id> <weight> <auxref> <pcoord_dim_0> <pcoord_dim_1> ...
read -r RMSD RG < "$PCOORD_TMP"
printf "0 1 bstate0 %s %s\n" "$RMSD" "$RG" > bstates/bstates.txt
echo "[setup] bstates.txt updated: $(cat bstates/bstates.txt)"
rm -f "$PCOORD_TMP"

# ---------------------------------------------------------------------------
# 5. Done — flag remaining manual gates
# ---------------------------------------------------------------------------
cat <<EOF

[setup] Setup complete for SYSTEM_NAME=${SYS_NAME}.

Remaining manual steps before production:
  7. Edit west.cfg bin boundaries to bracket your system's pcoord range.
     Current basis pcoord: RMSD=${RMSD} Å, Rg=${RG} Å.
     Out-of-the-box [-inf, inf] bins do NOT give meaningful WE sampling.
  8. (If switching from a previous system) move stale WE state aside:
       mv west.h5 traj_segs seg_logs istates .pre_refactor_backup/
     This is destructive; left manual on purpose.

Then: ./run_local.sh
EOF
