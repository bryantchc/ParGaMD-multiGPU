#!/usr/bin/env bash
# stop_gamd.sh — stop a running GaMD equilibration without risking the checkpoint.
#
# WHY THIS EXISTS. The vendored gamd runtime installs NO signal handlers: no
# SIGTERM/SIGINT handler, no atexit, no try/finally around the run loop. A
# SIGTERM therefore kills the process wherever it happens to be. Since
# simulation.saveCheckpoint() writes gamd_restart.checkpoint IN PLACE and not
# atomically, a signal arriving mid-write truncates it and the entire
# equilibration is lost.
#
# The checkpoint is rewritten every restart_checkpoint_interval steps (default
# 50000, ~71 s at ~700 steps/s for a 266k-atom system). So: watch for a write
# to complete, kill immediately afterwards, and keep a verified backup either
# way.
#
# Usage:  ./stop_gamd.sh <run_dir>
set -euo pipefail
RUN="${1:?usage: stop_gamd.sh <run_dir>}"
CKPT="$RUN/equilibration/out/gamd_restart.checkpoint"
[ -f "$CKPT" ] || { echo "[stop] no checkpoint at $CKPT" >&2; exit 1; }

echo "[stop] backing up the current checkpoint first"
cp -f "$CKPT" "$CKPT.bak"

echo "[stop] waiting for a fresh checkpoint write (max 5 min)..."
BEFORE=$(stat -c %Y "$CKPT")
for _ in $(seq 1 300); do
    sleep 1
    NOW=$(stat -c %Y "$CKPT")
    if [ "$NOW" != "$BEFORE" ]; then
        sleep 2                      # let the write finish flushing
        echo "[stop] checkpoint just refreshed; stopping now"
        break
    fi
done

PIDS=$(ps -eo pid,args --no-headers | awk '/gamdRunner_init/ && !/awk/ {print $1}')
if [ -z "$PIDS" ]; then
    echo "[stop] no gamdRunner_init process found"
else
    for p in $PIDS; do kill -TERM "$p" 2>/dev/null && echo "[stop] TERM $p"; done
    sleep 5
    for p in $PIDS; do kill -KILL "$p" 2>/dev/null && echo "[stop] KILL $p"; done
fi
sleep 2
cp -f "$CKPT" "$CKPT.stopped"
echo "[stop] checkpoint saved as .stopped ; previous good copy is .bak"
echo "[stop] verify with:  ./verify_gamd_checkpoint.sh $RUN"
