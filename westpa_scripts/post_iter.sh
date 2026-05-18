#!/bin/bash
# Fired by WESTPA after each iteration completes. Two jobs:
#   1. Tar up the per-segment log files into seg_logs/{ITER}.tar so the
#      seg_logs/ directory doesn't grow without bound.
#   2. (Optional, opt-in) Kick off background stripping of iter N-2's
#      trajectories via strip_iter.sh. The N-2 lag keeps the freeze-
#      fallback in runseg.sh safe — see strip_iter.sh header for details.

if [ -n "$SEG_DEBUG" ] ; then
    set -x
    env | sort
fi

cd "$WEST_SIM_ROOT" || exit 1

ITER=$(printf "%06d" $WEST_CURRENT_ITER)

# 1) Tar this iter's worker logs.
tar -cf seg_logs/$ITER.tar seg_logs/$ITER-*.log 2>/dev/null
rm  -f  seg_logs/$ITER-*.log

# 2) Optional trajectory stripping (opt-in via STRIP_AFTER_ITERS=1 in env.sh).
# We source env.sh defensively — WESTPA passes WEST_CURRENT_ITER but not our
# custom env vars.
# shellcheck disable=SC1091
[ -f "$WEST_SIM_ROOT/env.sh" ] && source "$WEST_SIM_ROOT/env.sh"
if [ "${STRIP_AFTER_ITERS:-0}" = "1" ]; then
    TO_STRIP=$(( ${WEST_CURRENT_ITER:-0} - 2 ))
    if [ "$TO_STRIP" -gt 0 ]; then
        # Fire-and-forget. All cpptraj output goes to a single strip.log
        # so the user has one place to grep for warnings/errors.
        nohup bash "$WEST_SIM_ROOT/westpa_scripts/strip_iter.sh" --iter "$TO_STRIP" \
            >> "$WEST_SIM_ROOT/strip.log" 2>&1 &
        disown
    fi
fi
exit 0
