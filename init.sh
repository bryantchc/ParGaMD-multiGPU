#!/bin/bash


set -x
# Set up simulation environment


source ~/.bashrc

# 1. Hook Conda into this shell script
eval "$(conda shell.bash hook)"
conda init
conda  activate pargamd

source env.sh


# Clean up from previous/ failed runs
rm -rf traj_segs seg_logs istates west.h5 
mkdir   seg_logs traj_segs istates

# Refresh the basis-state coords symlink to match the current SYSTEM_NAME.
# Without this, swapping SYSTEM_NAME but forgetting to update bstate0/ leaves
# stale coords there — get_pcoord.sh then loads them against the new topology
# and dies with an atom-count mismatch.
ln -sfv "../../common_files/${SYSTEM_NAME}.rst7" "$WEST_SIM_ROOT/bstates/bstate0/output_restart.rst7"

# Set pointer to bstate and tstate
BSTATE_ARGS="--bstate-file $WEST_SIM_ROOT/bstates/bstates.txt"
#TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstate.file"

# Run w_init
w_init \
  $BSTATE_ARGS \
  $TSTATE_ARGS \
  --segs-per-state 5 \
  --work-manager=threads "$@"
