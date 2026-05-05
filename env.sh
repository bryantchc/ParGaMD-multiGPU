#!/bin/bash
# Source ~/.bashrc for any user customizations; in non-interactive shells
# this is a no-op for conda, so we explicitly initialize the conda shell
# hook below before activating. Safe to source from any context.
source ~/.bashrc 2>/dev/null || true
if ! command -v conda >/dev/null || ! type conda 2>/dev/null | grep -q function; then
    # conda hook hasn't been loaded — do it explicitly
    for h in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
        [ -f "$h" ] && source "$h" && break
    done
fi
conda activate pargamd

if [[ -z "${WEST_SIM_ROOT:-}" ]]; then
    export WEST_SIM_ROOT="$PWD"
fi
export SIM_NAME=$(basename "$WEST_SIM_ROOT")
echo "simulation $SIM_NAME root is $WEST_SIM_ROOT"

export USE_LOCAL_SCRATCH=1
export WM_ZMQ_MASTER_HEARTBEAT=100
export WM_ZMQ_WORKER_HEARTBEAT=100
export WM_ZMQ_TIMEOUT_FACTOR=300

# System-specific files. To swap to a different protein, drop
# <name>.{parm7,rst7,pdb} into common_files/ and change this name.
# runseg.sh and get_pcoord.sh both honor this; input.xml uses generic
# in-segment-dir names (topology.parm7, coordinates.rst7) which are
# symlinked at runtime, so input.xml itself is system-agnostic.
export SYSTEM_NAME="${SYSTEM_NAME:-chignolin}"