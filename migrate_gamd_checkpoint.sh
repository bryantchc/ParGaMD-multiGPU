#!/usr/bin/env bash
# migrate_gamd_checkpoint.sh <run_dir> <src_gpu> <dst_gpu>
#
# Move a STOPPED GaMD equilibration to a different GPU so it can be resumed
# there with `CUDA_VISIBLE_DEVICES=<dst_gpu> ./equilibrate.sh --resume`.
#
# Why a plain --resume on the new card does not work: OpenMM CUDA checkpoints
# restore positions across devices, but the GaMD CustomIntegrator's globals
# (stepCount, stage, Vmax/Vmin/Vavg/sigmaV, k0, thresholds -- the accumulated
# boost statistics) came back scrambled moving 5070 -> 5090, and the run NaNed
# 5000 steps in. Worse, the failing runner then OVERWROTE the checkpoint.
#
# What this does instead:
#   1. on <src_gpu>: load the checkpoint, export the State (positions,
#      velocities, box, time) as OpenMM XML plus every integrator global and
#      per-DOF variable -- all device-independent
#   2. on <dst_gpu>: build the same GaMD simulation, import them, save a
#      checkpoint NATIVE to <dst_gpu>, and re-read it there to verify that
#      every global matches
#   3. install it; the original is kept as gamd_restart.checkpoint.src-gpu<N>
#
# Stop the run first (stop_gamd.sh). Uses the .stopped copy if present.
set -euo pipefail
RUN="$(readlink -f "${1:?usage: migrate_gamd_checkpoint.sh <run_dir> <src_gpu> <dst_gpu>}")"
SRC="${2:?src gpu}"; DST="${3:?dst gpu}"
EQ="$RUN/equilibration"; OUT="$EQ/out"
if ps -eo args --no-headers | awk '$1 ~ /(^|\/)python[0-9.]*$/ && /gamdRunner_init/' | grep -q "$RUN"; then
    echo "[migrate] a gamdRunner_init for $RUN is still running; stop it first" >&2; exit 1
fi
CK="$OUT/gamd_restart.checkpoint"; [ -f "$CK.stopped" ] && CK="$CK.stopped"
X="$EQ/ckpt_xfer_gpu${SRC}_to_gpu${DST}"; mkdir -p "$X"
source "$RUN/env.sh" >/dev/null
cd "$EQ"
echo "[migrate] export on GPU $SRC from $(basename "$CK")"
RT="$RUN/runtime" CUDA_VISIBLE_DEVICES=$SRC python "$RUN/runtime/migrate_gamd_checkpoint.py" export "$CK" "$X" 2>&1 | grep -E "^exported|Error|Trace"
echo "[migrate] import on GPU $DST"
RT="$RUN/runtime" CUDA_VISIBLE_DEVICES=$DST python "$RUN/runtime/migrate_gamd_checkpoint.py" import "$X" "$X/gamd_restart.checkpoint" 2>&1 | tee "$X/import.log" | grep -E "^imported|Error|Trace"
grep -q "mismatches: \[\]" "$X/import.log" || { echo "[migrate] verification FAILED; nothing installed" >&2; exit 1; }
cp -f "$OUT/gamd_restart.checkpoint" "$OUT/gamd_restart.checkpoint.src-gpu$SRC"
cp -f "$X/gamd_restart.checkpoint" "$OUT/gamd_restart.checkpoint"
echo "[migrate] installed. Resume with:"
echo "  cd $RUN && CUDA_VISIBLE_DEVICES=$DST ./equilibrate.sh --resume"
