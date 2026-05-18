#!/usr/bin/env bash
# new_run.sh — scaffold a fresh ParGaMD run dir, then equilibrate.
#
# Usage:
#   ./new_run.sh <run_dir> --system <name> [--system-from <path>] [--skip-equil] [--seed-from <dir>]
#
# Creates <run_dir> as a self-contained simulation:
#   <run_dir>/
#     env.sh, west.cfg, input.xml, bstates.txt    ← copied from templates/
#     equilibration/input.xml                      ← copied from templates/
#     system/<name>.{parm7,rst7,pdb}               ← copied from --system-from
#     runtime/        westpa_scripts/  bench/      ← symlinks to this repo
#     equilibrate.sh  run_local.sh    init.sh      ← symlinks to this repo
#
# Then runs equilibration (~60 min) and pre-computes the basis-state pcoord
# so the run is one ./run_local.sh away from production.
#
# Flags:
#   --system <name>       (required) the SYSTEM_NAME that env.sh exports
#   --system-from <path>  defaults to ../pargamd_systems/<name>/
#                         expects <name>.parm7, <name>.rst7, <name>.pdb there
#   --skip-equil          scaffold only, don't run equilibration
#   --seed-from <dir>     reuse an existing equilibration's outputs
#                         (must contain gamd_restart.checkpoint, gamd-restart.dat)
#                         — saves 60 min when re-running the same system

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
RUN_DIR=""
SYSTEM=""
SYSTEM_FROM=""
SKIP_EQUIL=0
SEED_FROM=""

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}
[ $# -ge 1 ] || usage
RUN_DIR="$1"; shift
while [ $# -gt 0 ]; do
    case "$1" in
        --system)       SYSTEM="$2"; shift 2 ;;
        --system-from)  SYSTEM_FROM="$2"; shift 2 ;;
        --skip-equil)   SKIP_EQUIL=1; shift ;;
        --seed-from)    SEED_FROM="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "[new_run] unknown arg: $1" >&2; usage ;;
    esac
done
[ -n "$SYSTEM" ] || { echo "[new_run] --system <name> required" >&2; exit 1; }
[ -n "$SYSTEM_FROM" ] || SYSTEM_FROM="$(dirname "$REPO_DIR")/pargamd_systems/$SYSTEM"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
for ext in parm7 rst7 pdb; do
    [ -f "$SYSTEM_FROM/${SYSTEM}.${ext}" ] \
        || { echo "[new_run] missing $SYSTEM_FROM/${SYSTEM}.${ext}" >&2; exit 1; }
done
if [ -e "$RUN_DIR" ]; then
    echo "[new_run] $RUN_DIR already exists; refusing to overwrite" >&2
    exit 1
fi
if [ -n "$SEED_FROM" ]; then
    [ -f "$SEED_FROM/gamd_restart.checkpoint" ] \
        || { echo "[new_run] --seed-from $SEED_FROM missing gamd_restart.checkpoint" >&2; exit 1; }
    [ -f "$SEED_FROM/gamd-restart.dat" ] \
        || { echo "[new_run] --seed-from $SEED_FROM missing gamd-restart.dat" >&2; exit 1; }
fi

echo "[new_run] scaffolding $RUN_DIR for SYSTEM_NAME=$SYSTEM"
echo "[new_run]   system files from: $SYSTEM_FROM"
echo "[new_run]   code dir (this repo): $REPO_DIR"

# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------
mkdir -p "$RUN_DIR/system" "$RUN_DIR/bstates/bstate0" "$RUN_DIR/equilibration"

# System files — copy by default (self-contained run dir). Users who want
# library-style sharing can `mv` the copies and re-symlink by hand.
cp "$SYSTEM_FROM/${SYSTEM}.parm7" "$RUN_DIR/system/${SYSTEM}.parm7"
cp "$SYSTEM_FROM/${SYSTEM}.rst7"  "$RUN_DIR/system/${SYSTEM}.rst7"
cp "$SYSTEM_FROM/${SYSTEM}.pdb"   "$RUN_DIR/system/${SYSTEM}.pdb"

# Symlinks to the repo for code dirs / scripts.
ln -s "$REPO_DIR/runtime"         "$RUN_DIR/runtime"
ln -s "$REPO_DIR/westpa_scripts"  "$RUN_DIR/westpa_scripts"
ln -s "$REPO_DIR/bench"           "$RUN_DIR/bench"
ln -s "$REPO_DIR/run_local.sh"    "$RUN_DIR/run_local.sh"
ln -s "$REPO_DIR/equilibrate.sh"  "$RUN_DIR/equilibrate.sh"
ln -s "$REPO_DIR/init.sh"         "$RUN_DIR/init.sh"

# Templates → run dir, with SYSTEM_NAME substitution.
sed "s|__SYSTEM_NAME__|$SYSTEM|g" "$REPO_DIR/templates/env.sh.template"   > "$RUN_DIR/env.sh"
chmod +x "$RUN_DIR/env.sh"
cp "$REPO_DIR/templates/west.cfg.template"                "$RUN_DIR/west.cfg"
cp "$REPO_DIR/templates/input.xml.template"               "$RUN_DIR/input.xml"
cp "$REPO_DIR/templates/equilibration_input.xml.template" "$RUN_DIR/equilibration/input.xml"
cp "$REPO_DIR/templates/bstates.txt.template"             "$RUN_DIR/bstates/bstates.txt"

echo "[new_run] scaffold done"

# ---------------------------------------------------------------------------
# Equilibration (optional)
# ---------------------------------------------------------------------------
if [ -n "$SEED_FROM" ]; then
    echo "[new_run] importing seed from $SEED_FROM (skipping equilibration)"
    mkdir -p "$RUN_DIR/equilibration/out"
    cp "$SEED_FROM/gamd_restart.checkpoint" "$RUN_DIR/equilibration/out/"
    cp "$SEED_FROM/gamd-restart.dat"        "$RUN_DIR/equilibration/out/"
    ( cd "$RUN_DIR" && \
      ln -sfn equilibration/out/gamd_restart.checkpoint gamd_restart.checkpoint && \
      ln -sfn equilibration/out/gamd-restart.dat        gamd-restart.dat )
elif [ "$SKIP_EQUIL" -eq 1 ]; then
    echo "[new_run] --skip-equil: scaffold only, run ./equilibrate.sh inside the run dir when ready"
else
    ( cd "$RUN_DIR" && ./equilibrate.sh )
fi

# ---------------------------------------------------------------------------
# Pre-compute basis-state pcoord (only when we actually have a seed wired)
# ---------------------------------------------------------------------------
if [ -f "$RUN_DIR/gamd_restart.checkpoint" ]; then
    echo "[new_run] computing basis-state pcoord…"
    ( cd "$RUN_DIR" && \
      ln -sfn "$SYSTEM_FROM/${SYSTEM}.rst7" "bstates/bstate0/output_restart.rst7" || \
        ln -sfn "system/${SYSTEM}.rst7"     "bstates/bstate0/output_restart.rst7" )
    PCOORD_TMP=$(mktemp)
    WEST_SIM_ROOT="$RUN_DIR" \
    WEST_STRUCT_DATA_REF="$RUN_DIR/bstates/bstate0" \
    WEST_PCOORD_RETURN="$PCOORD_TMP" \
        bash "$RUN_DIR/westpa_scripts/get_pcoord.sh"
    if [ -s "$PCOORD_TMP" ]; then
        read -r RMSD RG < "$PCOORD_TMP"
        printf "0 1 bstate0 %s %s\n" "$RMSD" "$RG" > "$RUN_DIR/bstates/bstates.txt"
        echo "[new_run] bstates.txt: $(cat "$RUN_DIR/bstates/bstates.txt")"
    else
        echo "[new_run] WARN: pcoord computation produced no output — bstates.txt left as default" >&2
    fi
    rm -f "$PCOORD_TMP"
fi

cat <<EOF

[new_run] Run dir ready: $RUN_DIR

Remaining manual steps before production:
  1. Edit $RUN_DIR/west.cfg bin boundaries to bracket your system's pcoord
     range. The default [0–8 Å] grid is chignolin-tuned and will not give
     meaningful WE sampling for other systems.
  2. Review $RUN_DIR/input.xml (per-segment GaMD params) for your system —
     dt, sigma0, T, extension-steps may need adjustment.

Then:
  cd $RUN_DIR
  ./run_local.sh           # or with WORKERS_GPU0= WORKERS_GPU1= overrides
EOF
