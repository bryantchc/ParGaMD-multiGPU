#!/usr/bin/env bash
# strip_iter.sh — strip solvent from segment DCDs to recover disk space.
#
# Each segment's output_restart.dcd is run through cpptraj with $STRIP_MASK
# (default: water + standard counterions, keeping catalytic divalents like
# Mg2+ / Zn2+ / Ca2+). On success the original DCD is replaced with the
# stripped one and a .stripped marker file is written next to it so
# repeated runs are idempotent.
#
# A stripped topology ($SYSTEM_NAME.stripped.parm7) is produced on first
# invocation in two places: alongside the full topology in $PARGAMD_SYSTEM_DIR,
# AND in traj_segs/ so post-hoc analysis scripts find it next to the data.
#
# Usage:
#   ./strip_iter.sh --iter <N>          strip a single completed iteration
#   ./strip_iter.sh --finalize          strip EVERY iter not yet marked stripped
#                                       (use after your final WESTPA iteration —
#                                        bypasses the N-2 safety lag)
#   ./strip_iter.sh --all               alias for --finalize
#
#   ./strip_iter.sh --prune-checkpoints [--keep-every N] [--behind M] [--dry-run]
#                                       DELETE gamd_restart.checkpoint from old
#                                       iterations, keeping every Nth. IRREVERSIBLE.
#                                       See "Checkpoint pruning" below.
#
# Checkpoint pruning
# ------------------
# Checkpoints dominate a stripped run's footprint: a solvated checkpoint is
# ~49 MB per segment, so a 550-walker iteration costs ~27 GB in checkpoints
# against ~12 GB in stripped trajectories. Pruning old ones is usually the
# largest single disk win available to a finished or long-running simulation.
#
# This is deliberately NOT part of --iter/--finalize and never fires from
# post_iter.sh. Stripping is LOSSLESS with respect to restartability -- solvent
# leaves the DCD but the checkpoint still holds full positions, velocities and
# GaMD state. Deleting a checkpoint is IRREVERSIBLE: it permanently removes the
# ability to restart or re-seed a new run from that iteration, which is exactly
# how Ultra_RL2.rna.preloop3 was built (basis states harvested from iterations
# 103-121 of a finished run). That asset is worth keeping some of, so the
# default policy thins history rather than clearing it.
#
# Safety rails, all unconditional:
#   - only ever touches traj_segs/; bstates/ is never walked
#   - never prunes the newest --behind iterations (default 20). runseg.sh reads
#     only the PARENT's checkpoint (iter N-1), so 20 is far beyond what
#     propagation needs -- the margin is for crash recovery, not correctness.
#   - never prunes iteration 1
#   - keeps every --keep-every'th iteration (default 10) as re-seeding anchors
#   - --dry-run reports the reclaim total and deletes nothing
#
# Restart safety: WESTPA's runseg.sh freeze-fallback in iter N reads iter
# N-1's output_restart.dcd. So during a live run, post_iter.sh only fires
# stripping for iter N-2 — iters N-1 and N remain full-atom to keep the
# fallback safe across one Ctrl-C / resume cycle. When you're truly done
# (no more iters planned, ready for analysis), `--finalize` is the safe
# way to recover the disk space from those held-back iters.
#
# Concurrency: per-iter lock file (.stripping.lock) prevents the WESTPA
# background hook and a manual invocation from racing on the same iter.

set -euo pipefail

# BASH_SOURCE (not readlink -f) so the run dir's symlink → repo
# resolves to the run dir, not the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
WEST_SIM_ROOT="$PWD"
export WEST_SIM_ROOT
# shellcheck disable=SC1091
[ -f env.sh ] && source env.sh

: "${SYSTEM_NAME:?SYSTEM_NAME is unset; check env.sh}"
: "${PARGAMD_SYSTEM_DIR:?PARGAMD_SYSTEM_DIR is unset; check env.sh}"
STRIP_MASK="${STRIP_MASK:-:WAT,Na+,Cl-,K+}"

command -v cpptraj >/dev/null \
    || { echo "[strip] cpptraj not on PATH (need AmberTools)" >&2; exit 1; }

FULL_TOPO="$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.parm7"
STRIPPED_TOPO="$PARGAMD_SYSTEM_DIR/${SYSTEM_NAME}.stripped.parm7"
TS_STRIPPED_TOPO="$WEST_SIM_ROOT/traj_segs/${SYSTEM_NAME}.stripped.parm7"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
MODE=""; ITER=""
KEEP_EVERY="${PRUNE_CHECKPOINTS_KEEP_EVERY:-10}"
BEHIND="${PRUNE_CHECKPOINTS_BEHIND:-20}"
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --iter)              MODE=iter; ITER="$2"; shift 2 ;;
        --finalize|--all)    MODE=finalize; shift ;;
        --prune-checkpoints) MODE=prune; shift ;;
        --keep-every)        KEEP_EVERY="$2"; shift 2 ;;
        --behind)            BEHIND="$2"; shift 2 ;;
        --dry-run|-n)        DRY_RUN=1; shift ;;
        -h|--help)           usage ;;
        *) echo "[strip] unknown arg: $1" >&2; usage ;;
    esac
done
[ -n "$MODE" ] || { echo "[strip] need --iter <N>, --finalize or --prune-checkpoints" >&2; usage; }

if [ "$MODE" = prune ]; then
    case "$KEEP_EVERY" in ''|*[!0-9]*) echo "[prune] --keep-every must be a positive integer" >&2; exit 1 ;; esac
    case "$BEHIND"     in ''|*[!0-9]*) echo "[prune] --behind must be a non-negative integer" >&2; exit 1 ;; esac
    [ "$KEEP_EVERY" -ge 1 ] || { echo "[prune] --keep-every must be >= 1" >&2; exit 1; }
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Generate the stripped topology once per system. Writes to the canonical
# location next to the original parm7, and to traj_segs/ for analysis.
ensure_stripped_topo() {
    [ -f "$STRIPPED_TOPO" ] && {
        # Mirror into traj_segs/ if missing (e.g., first iter after a restart)
        [ -f "$TS_STRIPPED_TOPO" ] || cp -f "$STRIPPED_TOPO" "$TS_STRIPPED_TOPO" 2>/dev/null || true
        return 0
    }
    echo "[strip $(date +%H:%M:%S)] generating stripped topology"
    local tmp; tmp=$(mktemp -d)
    cat > "$tmp/topology.cpptraj" <<EOF
parm $FULL_TOPO
parmstrip $STRIP_MASK
parmwrite out $STRIPPED_TOPO
quit
EOF
    if ! cpptraj -i "$tmp/topology.cpptraj" > "$tmp/cpptraj.out" 2>&1; then
        echo "[strip] ERROR: stripped topology generation failed" >&2
        sed 's/^/[strip cpptraj] /' "$tmp/cpptraj.out" >&2
        rm -rf "$tmp"; return 1
    fi
    # Surface any cpptraj warnings (don't fail on them, just log)
    grep -iE "warning|error" "$tmp/cpptraj.out" \
        | sed 's/^/[strip cpptraj-topo] /' || true
    mkdir -p "$(dirname "$TS_STRIPPED_TOPO")"
    cp -f "$STRIPPED_TOPO" "$TS_STRIPPED_TOPO"
    rm -rf "$tmp"
    [ -f "$STRIPPED_TOPO" ]
}

# Strip a single segment's DCD. Atomic-ish:
#   - cpptraj writes to output_restart.stripped.dcd
#   - on success, mv over output_restart.dcd and touch .stripped marker
#   - on failure, leave original untouched
strip_segment() {  # $1 seg_dir
    local seg="$1"
    [ -f "$seg/.stripped" ]            && return 0   # idempotent
    [ -s "$seg/output_restart.dcd" ]   || return 0   # nothing to strip

    local tmp; tmp=$(mktemp -d)
    cat > "$tmp/strip.cpptraj" <<EOF
parm $FULL_TOPO
trajin $seg/output_restart.dcd
strip $STRIP_MASK
trajout $seg/output_restart.stripped.dcd dcd
go
quit
EOF
    if ! cpptraj -i "$tmp/strip.cpptraj" > "$tmp/cpptraj.out" 2>&1; then
        echo "[strip] ERROR: cpptraj failed for $seg" >&2
        sed 's/^/[strip cpptraj] /' "$tmp/cpptraj.out" >&2
        rm -rf "$tmp"
        return 1
    fi
    if [ ! -s "$seg/output_restart.stripped.dcd" ]; then
        echo "[strip] ERROR: cpptraj produced no output for $seg" >&2
        sed 's/^/[strip cpptraj] /' "$tmp/cpptraj.out" >&2
        rm -rf "$tmp"
        return 1
    fi
    # Surface non-fatal cpptraj warnings so the user knows
    local warns
    warns=$(grep -iE "warning|error" "$tmp/cpptraj.out" || true)
    [ -n "$warns" ] && echo "$warns" | sed "s|^|[strip cpptraj-warn $seg] |"

    mv -f "$seg/output_restart.stripped.dcd" "$seg/output_restart.dcd"
    touch "$seg/.stripped"
    rm -rf "$tmp"
}

# Strip every segment in a given iteration. Locks the iter dir to prevent
# concurrent strip processes (WESTPA hook racing with manual invocation).
strip_iter_dir() {  # $1 iter_number
    local iter="$1"
    local iter_dir="traj_segs/$(printf '%06d' "$iter")"
    if [ ! -d "$iter_dir" ]; then
        echo "[strip $(date +%H:%M:%S)] iter $iter: no $iter_dir, skipping"
        return 0
    fi

    local lock="$iter_dir/.stripping.lock"
    if ! ( set -o noclobber; echo $$ > "$lock" ) 2>/dev/null; then
        local holder; holder=$(cat "$lock" 2>/dev/null || echo "?")
        echo "[strip $(date +%H:%M:%S)] iter $iter: already locked by pid $holder, skipping"
        return 0
    fi
    trap 'rm -f "$lock"' RETURN

    if ! ensure_stripped_topo; then
        echo "[strip] ERROR: cannot strip iter $iter without stripped topology" >&2
        rm -f "$lock"; trap - RETURN
        return 1
    fi

    local total=0 already=0 done_now=0 failed=0
    local start_b=0 end_b=0
    for seg in "$iter_dir"/*/; do
        [ -d "$seg" ] || continue
        total=$((total+1))
        if [ -f "$seg/.stripped" ]; then
            already=$((already+1)); continue
        fi
        local orig_size; orig_size=$(stat -c%s "$seg/output_restart.dcd" 2>/dev/null || echo 0)
        start_b=$((start_b + orig_size))
        if strip_segment "$seg"; then
            done_now=$((done_now+1))
            local new_size; new_size=$(stat -c%s "$seg/output_restart.dcd" 2>/dev/null || echo 0)
            end_b=$((end_b + new_size))
        else
            failed=$((failed+1))
            end_b=$((end_b + orig_size))
        fi
    done
    rm -f "$lock"; trap - RETURN

    local savings_mb=$(( (start_b - end_b) / 1024 / 1024 ))
    echo "[strip $(date +%H:%M:%S)] iter $iter: $done_now stripped, $already already-stripped, $failed failed (saved ${savings_mb} MB)"
}

# Delete checkpoints from one iteration. Returns bytes reclaimed on stdout.
prune_iter_dir() {  # $1 iter_number
    local iter="$1"
    local iter_dir="traj_segs/$(printf '%06d' "$iter")"
    [ -d "$iter_dir" ] || { echo 0; return 0; }

    # Share the stripping lock so we never race the post_iter hook on an iter.
    local lock="$iter_dir/.stripping.lock"
    if [ "$DRY_RUN" -eq 0 ]; then
        if ! ( set -o noclobber; echo $$ > "$lock" ) 2>/dev/null; then
            local holder; holder=$(cat "$lock" 2>/dev/null || echo "?")
            echo "[prune $(date +%H:%M:%S)] iter $iter: locked by pid $holder, skipping" >&2
            echo 0; return 0
        fi
    fi

    local bytes=0 n=0
    for seg in "$iter_dir"/*/; do
        [ -d "$seg" ] || continue
        local ck="$seg/gamd_restart.checkpoint"
        [ -f "$ck" ] || continue
        local sz; sz=$(stat -c%s "$ck" 2>/dev/null || echo 0)
        bytes=$((bytes + sz)); n=$((n + 1))
        [ "$DRY_RUN" -eq 0 ] && rm -f "$ck"
    done
    if [ "$DRY_RUN" -eq 0 ]; then
        rm -f "$lock"
        [ "$n" -gt 0 ] && date +%FT%T > "$iter_dir/.checkpoints_pruned"
    fi
    echo "[prune $(date +%H:%M:%S)] iter $iter: $n checkpoints $( [ "$DRY_RUN" -eq 1 ] && echo "would be removed" || echo removed ) ($((bytes/1024/1024)) MB)" >&2
    echo "$bytes"
}

prune_checkpoints() {
    # Highest iteration directory present. During a live run this is the
    # in-flight iteration, which is the conservative choice for --behind.
    local latest=0 i
    for iter_dir in traj_segs/0*/; do
        [ -d "$iter_dir" ] || continue
        i=$((10#$(basename "$iter_dir")))
        [ "$i" -gt "$latest" ] && latest=$i
    done
    if [ "$latest" -eq 0 ]; then
        echo "[prune] no iterations found under traj_segs/; nothing to do" >&2
        return 0
    fi

    local cutoff=$((latest - BEHIND))
    echo "[prune] latest iteration present : $latest"
    echo "[prune] policy                   : keep every ${KEEP_EVERY}th, keep newest ${BEHIND}, keep iter 1"
    echo "[prune] prunable range           : iterations 2..$cutoff"
    [ "$DRY_RUN" -eq 1 ] && echo "[prune] DRY RUN -- nothing will be deleted"
    if [ "$cutoff" -lt 2 ]; then
        echo "[prune] nothing is old enough to prune (need iterations <= $cutoff)"
        return 0
    fi

    local total=0 pruned=0 kept=0 got
    for iter_dir in traj_segs/0*/; do
        [ -d "$iter_dir" ] || continue
        i=$((10#$(basename "$iter_dir")))
        if [ "$i" -eq 1 ] || [ "$i" -gt "$cutoff" ] || [ $((i % KEEP_EVERY)) -eq 0 ]; then
            kept=$((kept + 1)); continue
        fi
        got=$(prune_iter_dir "$i")
        total=$((total + got)); pruned=$((pruned + 1))
    done

    echo "[prune] ------------------------------------------------------------"
    echo "[prune] iterations pruned : $pruned"
    echo "[prune] iterations kept   : $kept  (anchors + newest $BEHIND + iter 1)"
    printf '[prune] space %s : %.1f GB\n' \
        "$( [ "$DRY_RUN" -eq 1 ] && echo reclaimable || echo reclaimed )" \
        "$(awk -v b="$total" 'BEGIN{print b/1073741824}')"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    iter)
        strip_iter_dir "$ITER"
        ;;
    prune)
        prune_checkpoints
        ;;
    finalize)
        # Walk every traj_segs/0*/ in order. Iterations are numeric; sort
        # numerically so 9 < 10 < 11 < ... (zero-padding handles this for us).
        for iter_dir in traj_segs/0*/; do
            [ -d "$iter_dir" ] || continue
            # 10# forces base-10 interpretation in case of leading zeros
            strip_iter_dir "$((10#$(basename "$iter_dir")))"
        done
        ;;
esac
