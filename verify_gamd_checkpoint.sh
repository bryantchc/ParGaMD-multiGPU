#!/usr/bin/env bash
# verify_gamd_checkpoint.sh — confirm a GaMD checkpoint actually loads.
# A truncated checkpoint is not detectable by size alone; the only real test is
# to build the system and load it. Run this BEFORE rebooting, so a corrupt
# checkpoint is caught while the .bak copy is still fresh.
#
# LIMITATION: this loads positions with a plain Verlet integrator, so it cannot
# see the GaMD integrator's global variables (stepCount, stage, Vmax, k0, ...).
# A checkpoint written on one GPU and loaded on ANOTHER passes this check while
# every integrator global comes back scrambled (seen 5070 -> 5090: k0 = 5.8,
# stage 7.6) and the resume NaNs within 5000 steps. To change GPU, use
# migrate_gamd_checkpoint.sh, never a plain --resume on the new card.
set -euo pipefail
RUN="${1:?usage: verify_gamd_checkpoint.sh <run_dir>}"
cd "$RUN"; source env.sh
python - "$RUN" <<'PY'
import sys, warnings; warnings.filterwarnings('ignore')
import openmm as mm
from openmm import app, unit
run=sys.argv[1]
prm=app.AmberPrmtopFile(f'{run}/equilibration/topology.parm7')
s=prm.createSystem(nonbondedMethod=app.PME,nonbondedCutoff=0.9*unit.nanometer,constraints=app.HBonds)
sim=app.Simulation(prm.topology,s,mm.VerletIntegrator(0.001*unit.picoseconds),
                   mm.Platform.getPlatformByName('CUDA'),{'Precision':'mixed'})
for tag in ('','.stopped','.bak'):
    f=f'{run}/equilibration/out/gamd_restart.checkpoint{tag}'
    try:
        sim.loadCheckpoint(f)
        st=sim.context.getState(getEnergy=True)
        e=st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        try: step=int(sim.integrator.getGlobalVariableByName('stepCount'))
        except Exception: step=-1
        print('  OK   %-14s E=%.4e kJ/mol  stepCount=%d'%(tag or '(live)',e,step))
    except Exception as ex:
        print('  BAD  %-14s %s'%(tag or '(live)',type(ex).__name__))
PY
