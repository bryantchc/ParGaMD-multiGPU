"""Move a GaMD CustomIntegrator checkpoint between CUDA devices.
OpenMM CUDA checkpoints restore positions across cards but scrambled every
integrator global here (5070 -> 5090). So: export the State + all integrator
globals and per-DOF variables in device-independent form, re-import on the
target card, and write a checkpoint native to it.
  export <ckpt> <dir>     (run on the SOURCE card)
  import <dir> <ckpt_out> (run on the TARGET card)
"""
import sys, os, json, pickle
import openmm as mm
sys.path.insert(0, os.environ['RT'])
from gamd import gamdSimulation, parser
cfg = parser.ParserFactory().parse_file('input.xml', 'xml')
s = gamdSimulation.GamdSimulationFactory().createGamdSimulation(cfg, 'CUDA', '0').simulation
I = s.integrator
mode = sys.argv[1]
if mode == 'export':
    s.loadCheckpoint(sys.argv[2]); out = sys.argv[3]
    st = s.context.getState(getPositions=True, getVelocities=True, getParameters=True, getEnergy=True)
    open(f'{out}/state.xml', 'w').write(mm.XmlSerializer.serialize(st))
    g = {I.getGlobalVariableName(i): I.getGlobalVariable(i) for i in range(I.getNumGlobalVariables())}
    json.dump(g, open(f'{out}/globals.json', 'w'), indent=1)
    pd = {I.getPerDofVariableName(i): [tuple(v) for v in I.getPerDofVariable(i)] for i in range(I.getNumPerDofVariables())}
    pickle.dump(pd, open(f'{out}/perdof.pkl', 'wb'))
    print('exported: %d globals, %d per-DOF vars, stepCount %d, stage %g, E %.6e kJ/mol'
          % (len(g), len(pd), g['stepCount'], g['stage'], st.getPotentialEnergy()._value))
else:
    src = sys.argv[2]; out = sys.argv[3]
    st = mm.XmlSerializer.deserialize(open(f'{src}/state.xml').read())
    s.context.setState(st)
    g = json.load(open(f'{src}/globals.json'))
    for k, v in g.items(): I.setGlobalVariableByName(k, v)
    for k, vals in pickle.load(open(f'{src}/perdof.pkl', 'rb')).items(): I.setPerDofVariableByName(k, [mm.Vec3(*x) for x in vals])
    s.saveCheckpoint(out)
    # read back from the file we just wrote, on this card
    s2 = gamdSimulation.GamdSimulationFactory().createGamdSimulation(cfg, 'CUDA', '0').simulation
    s2.loadCheckpoint(out); I2 = s2.integrator
    bad = [k for k, v in g.items() if abs(I2.getGlobalVariableByName(k) - v) > 1e-9 * max(1.0, abs(v))]
    e = s2.context.getState(getEnergy=True).getPotentialEnergy()._value
    print('imported + re-read: %d/%d globals match, stepCount %d, stage %g, E %.6e kJ/mol, mismatches: %s'
          % (len(g) - len(bad), len(g), I2.getGlobalVariableByName('stepCount'), I2.getGlobalVariableByName('stage'), e, bad[:5]))
