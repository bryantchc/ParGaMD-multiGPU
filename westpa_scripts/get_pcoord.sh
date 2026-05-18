#!/bin/bash
##############################################################################
# MDAnalysis-based pcoord driver for WESTPA. Computes:
#   - RMSD of CA atoms (mass-weighted, best-fit alignment)
#   - Radius of gyration of CA atoms (mass-weighted)
#
# Called by WESTPA for both basis states (single-frame .rst7) and any
# externally-supplied structures. Per-segment pcoords are produced
# directly by runseg.sh, so this script primarily serves bstates.
#
# Inputs (in $WEST_STRUCT_DATA_REF):
#   - output_restart.dcd  -> trajectory (preferred if present)
#   - output_restart.rst7 -> single-frame restart (basis state fallback)
#
# Topology and reference structure are pulled from $PARGAMD_SYSTEM_DIR.
##############################################################################

# Source env.sh defensively so manual invocations (e.g. pre-flight pcoord
# checks before w_init) get SYSTEM_NAME and the conda env without the user
# having to source it themselves. Under WESTPA the env is already inherited
# from run_local.sh, but sourcing again is idempotent.
if [ -n "$WEST_SIM_ROOT" ] && [ -f "$WEST_SIM_ROOT/env.sh" ]; then
    source "$WEST_SIM_ROOT/env.sh"
fi

# 1. Optionally enable debugging
if [ -n "$SEG_DEBUG" ]; then
    set -x
    env | sort
fi

# 2. Move to the segment directory
cd "$WEST_STRUCT_DATA_REF" || {
    echo "Error: Could not cd to \$WEST_CURRENT_SEG_DATA_REF=$WEST_STRUCT_DATA_REF" >&2
    exit 1
}

# 3. Create temp files for RMSD and Rg data
RMSD_FILE=$(mktemp --tmpdir rmsd_XXXX.xvg)
RG_FILE=$(mktemp --tmpdir rg_XXXX.xvg)

# 4. Create a small Python script to run MDAnalysis
cat << EOF > mdanalysis_rmsd_rg.py
#!/usr/bin/env python

import os
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms

# Hard error if SYSTEM_NAME or PARGAMD_SYSTEM_DIR are unset rather than
# silently defaulting — missing env used to mask system swaps and produce
# misleading "atom count mismatch" failures against the wrong topology.
sys_name   = os.environ["SYSTEM_NAME"]
system_dir = os.environ["PARGAMD_SYSTEM_DIR"]
topology = os.path.join(system_dir, f"{sys_name}.parm7")
ref_pdb  = os.path.join(system_dir, f"{sys_name}.pdb")

# Prefer the segment trajectory; fall back to the basis-state restart.
# MDAnalysis cannot infer the AMBER restart format from the .rst7 extension,
# so we pass format="INPCRD" explicitly when loading the .rst7.
if os.path.exists("output_restart.dcd"):
    u = mda.Universe(topology, "output_restart.dcd")
elif os.path.exists("output_restart.rst7"):
    u = mda.Universe(topology, "output_restart.rst7", format="INPCRD")
else:
    raise FileNotFoundError(
        "Neither output_restart.dcd nor output_restart.rst7 found in "
        + os.getcwd()
    )

ref = mda.Universe(ref_pdb)

# Select CA atoms
ref_ca    = ref.select_atoms("name CA")
mobile_ca = u.select_atoms("name CA")

# Prepare lists for RMSD and Rg
rmsd_values = []
rg_values   = []

# Loop through frames in the trajectory
for ts in u.trajectory:
    # -- Mass-weighted best-fit RMSD of CA to the reference CA --
    this_rmsd = rms.rmsd(
        mobile_ca.positions,
        ref_ca.positions,
        center=True,
        superposition=True,
        weights=mobile_ca.masses
    )

    # -- Mass-weighted Rg calculation for CA atoms --
    coords = mobile_ca.positions
    masses = mobile_ca.masses
    com    = np.average(coords, axis=0, weights=masses)
    sq_dist = np.sum((coords - com)**2, axis=1)
    this_rg = np.sqrt(np.average(sq_dist, weights=masses))

    rmsd_values.append(this_rmsd)
    rg_values.append(this_rg)

# Write out data in a simple two-column format (frame vs. value):
with open("${RMSD_FILE}", "w") as f_rms:
    f_rms.write("# frame RMSD(Angstrom)\n")
    for i, val in enumerate(rmsd_values):
        # Frame index i+1 for readability
        f_rms.write(f"{i+1} {val}\n")

with open("${RG_FILE}", "w") as f_rg:
    f_rg.write("# frame Rg(Angstrom)\n")
    for i, val in enumerate(rg_values):
        f_rg.write(f"{i+1} {val}\n")
EOF

chmod +x mdanalysis_rmsd_rg.py

# 5. Run the Python script
if ! ./mdanalysis_rmsd_rg.py; then
    echo "Error: MDAnalysis script execution failed!" >&2
    rm -f "$RMSD_FILE" "$RG_FILE" mdanalysis_rmsd_rg.py
    exit 1
fi

# 6. Extract RMSD and Rg columns, then write them to $WEST_PCOORD_RETURN
paste <(awk 'NR>1 {print $2}' "$RMSD_FILE") \
      <(awk 'NR>1 {print $2}' "$RG_FILE") \
      > "$WEST_PCOORD_RETURN"

# 7. (Optional) Show a quick preview if in debug mode
if [ -n "$SEG_DEBUG" ]; then
    echo "Preview of \$WEST_PCOORD_RETURN:"
    head -v "$WEST_PCOORD_RETURN"
    echo "Number of lines in \$WEST_PCOORD_RETURN: \$(wc -l < "$WEST_PCOORD_RETURN")"
fi

# 8. Clean up
rm -f "$RMSD_FILE" "$RG_FILE" mdanalysis_rmsd_rg.py
exit 0
