# Local-workstation setup for ParGaMD

This document covers running this fork on a single Linux box with one or
more NVIDIA GPUs. For the original cluster-targeted documentation (theory,
reweighting, etc.), see `README.md`. The cluster launchers it references
have been moved to `_tacc_original/`.

## What changed vs. upstream

| Upstream (TACC) | This fork (local) |
|---|---|
| `run_WE_multi_MPS.sh` (SLURM, ZMQ master + node clients) | `run_local.sh` (single host, processes work manager) |
| Module loads (`module load cuda/12.6 …`) | Conda env `pargamd` only |
| `nvidia-cuda-mps-control -d` (one global daemon, always on) | MPS opt-in (`USE_MPS=1`); off by default — see "MPS gotcha on Blackwell" below |
| Hardcoded `/scratch/10597/anugrahat/...` paths in `runseg.sh`, `get_pcoord.sh`, `input.xml` | All `$WEST_SIM_ROOT`-relative |
| TACC seed `gamd_restart.checkpoint` (Hopper-built) | Locally-built Blackwell-compatible checkpoint at `common_files/equilibration/out/` (see "Phase 1 GaMD equilibration" below) |

## Prerequisites

- Linux + NVIDIA driver supporting your GPU (Blackwell/sm_120 needs the
  open driver branch; tested with the RTX 5090 on Ubuntu).
- Conda env `pargamd` with: Python 3.11, OpenMM ≥ 8.4 (CUDA build),
  WESTPA 2022.x, MDAnalysis, AmberTools, h5py.
- `nvidia-cuda-mps-control` on `PATH` (ships with the CUDA toolkit).

Verify before first run:

```bash
nvidia-smi -L
conda activate pargamd
python -c "import openmm; print(openmm.__version__)"
which w_run nvidia-cuda-mps-control
```

## Running a simulation

```bash
cd /path/to/ParGaMD_MPS_TACC
./run_local.sh
```

That's it. The launcher:

1. Activates conda env `pargamd`.
2. (If `USE_MPS=1`) starts an MPS daemon per visible GPU with isolated
   pipe and log dirs (`/tmp/nvidia-mps-$USER-<gpuid>` / `…-log-…`) and
   sets each daemon's `set_default_active_thread_percentage` to
   `100 / workers_on_that_gpu`. Default is `USE_MPS=0` — see "MPS
   gotcha on Blackwell" below.
3. Calls `init.sh` if and only if `west.h5` is missing — restarts are
   safe, no data loss.
4. Runs `w_run --work-manager=processes --n-workers=$TOTAL_WORKERS`.
5. Logs GPU utilization to `gpu_util.log` every 10 s.
6. On normal exit, Ctrl-C, or error: trap stops MPS (if started) and
   the logger. No orphan daemons.

## Knobs

All overridable from the env without editing the script:

| Var | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | GPUs to use, comma-separated (e.g. `0,1`). |
| `WORKERS_GPU0` | `12` | Concurrent workers on GPU 0 (RTX 5090, 32 GB). |
| `WORKERS_GPU1` | `6` | Concurrent workers on GPU 1 (RTX 5070, 12 GB). |
| `USE_MPS` | `auto` | `auto`: use sudo MPS if its daemon is running, else no MPS. `0`: force no MPS. `2`: force sudo MPS, fail if missing. `1`: user-mode MPS (broken on Blackwell — see below). |
| `SYSTEM_NAME` | `chignolin` | Picks `common_files/${SYSTEM_NAME}.{parm7,rst7,pdb}` for the propagator. See "Switching to a different system" below. |
| `CONDA_ENV` | `pargamd` | Conda environment name. |

### Scaling worker count

Each chignolin walker uses ~300–500 MB GPU memory. Tune by:

1. `nvidia-smi -l 5` while a run is happening.
2. If memory sits well below the card's capacity, raise `WORKERS_GPU0`.
3. If GPU utilization is pinned at 100% with no headroom for another
   walker to make forward progress, lower it.

For larger systems (>20k atoms), expect 1–2 GB/walker and scale down
accordingly.

### Iteration count

`max_total_iterations` lives in `west.cfg`. Edit it there. There is no
`w_run` CLI flag to override it.

### Adding the second GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_local.sh
```

**Caveat**: with `--work-manager=processes`, all workers inherit the
parent's environment, so individual workers cannot be pinned to
different GPUs. The launcher pins everything to the first visible GPU.
Multi-GPU therefore needs the ZMQ-per-worker pattern (one w_run
client per worker, each with its own `CUDA_VISIBLE_DEVICES` and
`CUDA_MPS_PIPE_DIRECTORY`).

The reference for that pattern is `_tacc_original/run_WE_multi_MPS.sh`
— most of the SLURM scaffolding can be cut out, leaving the for-loop
that spawns N ZMQ clients per GPU. Before doing that work, confirm
that single-GPU on the 5090 is bottlenecked, not just memory-bound.

## Restarting a stopped run

`run_local.sh` is idempotent: if `west.h5` exists, it skips `init.sh`
and `w_run` resumes from the last completed iteration.

```bash
# was killed mid-run; resume:
./run_local.sh
```

To start over, delete or move `west.h5`, `traj_segs/`, `seg_logs/`,
`istates/` first. The simplest:

```bash
mv west.h5 west.h5.$(date +%s).bak
./run_local.sh   # init.sh fires, fresh start
```

## Where things live

```
ParGaMD_MPS_TACC/
├── run_local.sh                        # the only launcher you need
├── env.sh                              # exports SYSTEM_NAME (default chignolin) + WESTPA env
├── init.sh                             # one-shot w_init for fresh starts
├── west.cfg                            # WESTPA config — bins, pcoord, iteration cap
├── bstates/                            # basis states + pre-computed pcoords
├── common_files/
│   ├── ${SYSTEM_NAME}.{parm7,rst7,pdb} # system bundle; swap by changing SYSTEM_NAME
│   ├── input.xml                       # GaMD config; references generic topology.parm7 / coordinates.rst7
│   ├── gamd_restart.checkpoint         # symlink to equilibration/out/gamd_restart.checkpoint
│   ├── gamd-restart.dat                # symlink to equilibration/out/gamd-restart.dat
│   ├── gamd/                           # gamd-openmm package (untouched)
│   ├── gamdRunner, gamdRunner_init     # runner scripts (untouched)
│   └── equilibration/                  # Phase 1 GaMD equilibration; produces the seed checkpoint
│       ├── input.xml                   # 31-ns equilibration protocol; uses generic names
│       ├── topology.parm7              # symlink to ../${SYSTEM_NAME}.parm7
│       ├── coordinates.rst7            # symlink to ../${SYSTEM_NAME}.rst7
│       └── out/                        # gamd_restart.checkpoint, gamd-restart.dat live here
├── westpa_scripts/                     # runseg.sh, get_pcoord.sh, gen_istate.sh, post_iter.sh, tar_segs.sh
├── traj_segs/, seg_logs/, istates/     # runtime output (recreated by init.sh)
├── west.h5                             # WESTPA HDF5 trajectory database
├── gpu_util.log                        # nvidia-smi log written each run
├── _tacc_original/                     # original SLURM/ZMQ launchers + the upstream TACC seed checkpoint, archived
└── .pre_refactor_backup/               # broken WE state from refactor smoke tests; safe to delete
```

## MPS modes

`run_local.sh` supports four MPS configurations via `USE_MPS=`:

| Mode | What happens | Throughput | When to use |
|---|---|---|---|
| `USE_MPS=auto` (default) | Probes `/tmp/nvidia-mps` for a live sudo daemon. If alive → uses it (= mode 2). If absent → no MPS (= mode 0). | depends | Always. Zero friction. |
| `USE_MPS=0` | Force no MPS. Workers serialize on the GPU via the CUDA driver. ~50% efficient at 4 workers. | ~309 ns/day aggregate | Debugging, or when you want the safe path explicitly. |
| `USE_MPS=1` | Launcher starts a per-GPU **user-mode** MPS daemon. **Known broken** — see below. | (fails) | Don't use until upstream fix. |
| `USE_MPS=2` | Force connection to an externally-managed **sudo-mode** daemon. Fails loud if missing. ~75% efficient at 4 workers. | **~1607 ns/day aggregate** | When you want to be sure MPS is active and not silently fall back. |

### Standard high-throughput run

```bash
sudo nvidia-cuda-mps-control -d   # once per session, type your password
./run_local.sh                    # uses MPS automatically (auto detects)
```

When you're done with the MPS daemon (you can leave it running indefinitely if you want):

```bash
echo quit | sudo nvidia-cuda-mps-control
```

Same lifecycle as your REST2 workflow — one system-wide root-mode MPS
daemon at `/tmp/nvidia-mps`, started once. ParGaMD and REST2 can share
it concurrently.

### The user-mode-MPS bug (USE_MPS=1)

When the launcher itself starts a per-GPU user-mode MPS daemon (no
sudo), 4 concurrent gamdRunner processes loading checkpoints all fail
with:

```
OverflowError: in method 'Context_setStepCount', argument 2 of type 'long long'
```

This is reproducible and it's *specifically* this combination: user-mode
MPS + multiple concurrent OpenMM contexts + GaMD's `CustomIntegrator` +
checkpoint loading on Blackwell (sm_120, OpenMM 8.4). Sudo-mode MPS at
the system pipe dir does not have this issue. Single-worker user-mode
MPS does not have this issue. Without MPS, no issue. So it's a narrow
upstream bug worth filing once we have a minimal repro.

## Throughput context

For reference: vanilla OpenMM Langevin on this same chignolin system
on the 5090, single precision, dt=2fs, NPT runs at **1267 ns/day**.
The current GaMD config (mixed precision, dt=2fs, NPT, upper-dual
boost) runs at **587 ns/day single-worker** — a 40% GaMD overhead
versus vanilla MD, which is in the published range (Miao 2015 reports
20-50% on small systems).

If you want more per-walker throughput:

1. **Switch GaMD to single precision** — buys ~30% (587 → ~760 ns/day).
   Hardcoded `CudaPrecision: mixed` in `common_files/gamd/runners.py:471`,
   `runners_init.py:471`, `gamdSimulation.py:226`. Slight accuracy
   tradeoff for NVE; for NPT/NVT with thermostat it's typically fine.
2. **Hydrogen-mass repartitioning + dt=4fs** — buys ~40%. Requires
   re-`parmed`'ing the parm7 (`hydrogenMass=4*unit.amu`) and
   re-equilibrating the GaMD bias.
3. **Fix MPS** — restores ~2× aggregate throughput at multi-worker.

## NaN robustness (segment failure rate)

ParGaMD walkers each pick a fresh random seed via OpenMM's
`setRandomNumberSeed(0)` auto-seed (we set `<random-seed>0</random-seed>`
in `common_files/input.xml`). Occasionally that seed lands the Langevin
integrator on an energy cliff and within a few steps a particle
coordinate goes NaN. WESTPA's default policy is strict — any segment
failure terminates the run; with 24 walkers and ~5-10% per-segment
failure rate, the chance of a clean iteration is only ~0.13.

`runseg.sh` ships with **three layered safety nets** that handle this
without your intervention:

1. **Stricter validity check than `-s`**: gamdRunner can write a
   header-only DCD (~96 bytes) and a frame or two before the NaN
   trips — non-empty but truncated. We read the DCD's NSET field
   directly from the header (`struct.unpack` on bytes 8-11) and
   require ≥ 100 frames before declaring success. Pure stdlib,
   ~10 ms per check, system-agnostic.
2. **Per-segment retry with new seed (×3)**: on a partial DCD,
   restore the parent checkpoint and `sed` a fresh `<random-seed>`
   into the local `input.xml`, then re-run gamdRunner. Different
   seeds rarely all NaN from the same parent state.
3. **Freeze-fallback**: if all 3 retries hit NaN, the parent state
   is deterministically NaN-prone. Rather than killing the WE run,
   `runseg.sh` copies the parent's DCD + checkpoint forward — the
   walker "stays in place" this iteration and gets resampled next.
   Logged as `FROZEN iter=N seg=M ...` in the seg log; grep for
   `FROZEN` in `seg_logs/` after a long run to count and audit.

In a 76-iteration production run on chignolin (24 walkers × 100 ps ×
76 iters = ~182 ns aggregate), **0 retries fired and 0 walkers
froze** — the auto-seed alone was sufficient. The retry/freeze
machinery is insurance, not the load-bearing fix.

Additional knobs you may want to consider for harder systems (the
above is correctness; these are quality):

- **Longer Phase 1 GaMD-eq**: bias parameters that are more converged
  push walkers less aggressively at iteration boundaries. The
  shipped Phase 1 protocol is 25 ns of GaMD-eq; doubling that to 50
  ns is reasonable for tougher systems.
- **Smaller timestep**: drop `dt` from 2 fs to 1.5 fs in
  `common_files/input.xml`. Trades ~25% throughput for stability.
- **Loosen `bin_target_counts`** in `west.cfg`: more walkers per
  bin = lower variance and fewer iterations needed to converge a
  given statistic. Pure science decision.
- **Per-segment minimization** is *not* currently active in WE
  segments: `<run-minimization>True</run-minimization>` only fires
  in gamdRunner's *initial* path, not in extension runs. Enabling it
  for extensions would require patching `common_files/gamd/runners.py`
  to call `simulation.minimizeEnergy()` after the new context is
  built — out of scope for this refactor (don't modify the gamd/
  package), but a cheap tweak if you decide you need it.

## Troubleshooting

**`w_run: error: unrecognized arguments`** — `w_run` has no flag for
iteration count. Edit `max_total_iterations` in `west.cfg`.

**`OSError: Can't synchronously read data (filter returned failure
during read)` on west.h5** — usually a malformed pcoord dataset from a
previous failed run. Move `west.h5` aside and re-run; `init.sh` will
fire and rebuild it.

**Workers crashing immediately on iteration 1** — check
`seg_logs/000001-*.log` for the GaMD output. Common causes: missing
`gamd_restart.checkpoint` (verify `common_files/gamd_restart.checkpoint`
points to a real file via `ls -L`), or topology/coordinate mismatch
between the parm7 and rst7 you put under
`common_files/${SYSTEM_NAME}.{parm7,rst7}`.

**Sudo MPS daemon doesn't start** — `sudo nvidia-cuda-mps-control -d`
hangs or errors. Usually because a stale daemon's pipe dir is
present. Check `ls /tmp/nvidia-mps`; if there's a `.pid` file and the
PID isn't running, `sudo rm -rf /tmp/nvidia-mps` and retry.

**Stale user-mode MPS dirs** — left over from running with
`USE_MPS=1` (broken). Cleanup:
```bash
for d in /tmp/nvidia-mps-$USER-*; do
    CUDA_MPS_PIPE_DIRECTORY=$d sh -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null
done
rm -rf /tmp/nvidia-mps-$USER-* /tmp/nvidia-log-$USER-*
```

**`OverflowError: ... 'long long'` in `Context_setStepCount`** — only
seen with `USE_MPS=1` on Blackwell with concurrent OpenMM contexts.
Use `USE_MPS=auto` (default) or `USE_MPS=2`. See "MPS modes".

**`Particle coordinate is NaN` in seg log** — physical NaN. The retry
+ freeze-fallback in `runseg.sh` should make this self-healing; if
you see seg logs reporting `FROZEN`, the safety net fired. See "NaN
robustness".

**`OSError: opened empty file. No frames are saved` (MDAnalysis)** —
historical: occurred when `runseg.sh`'s validity check was just `-s`
on the DCD (passes for header-only stubs). Fixed by reading the DCD's
NSET field via `struct` and requiring ≥ 100 frames. Shouldn't recur.

**GPU memory growing without bound across iterations** — usually a
WESTPA segment cleanup issue, not the GaMD propagator. Check that
`runseg.sh` is producing both `output_restart.dcd` and
`gamd_restart.checkpoint` in each segment dir (see `traj_segs/000NNN/000NNN/`).

## Phase 1 GaMD equilibration

ParGaMD WE walkers branch from a *seed* `gamd_restart.checkpoint` in
iteration 1. That seed encodes the atom positions, velocities, and
converged GaMD bias parameters (E threshold, k, V̄, σ_V) at the end of
a prior equilibration. **Checkpoints are GPU-architecture-specific** —
a checkpoint produced on, say, a TACC H100 (Hopper, sm_90) cannot be
loaded on Blackwell (sm_120); `loadCheckpoint()` fails inside OpenMM
allocating the integrator's per-particle RNG array.

The Phase 1 equilibration that produced this fork's chignolin seed
lives at `common_files/equilibration/`:

```
common_files/equilibration/
├── input.xml             # 15.5M steps = 31 ns: cMD-prep + cMD + GaMD-eq + GaMD-prod
├── topology.parm7    -> ../${SYSTEM_NAME}.parm7   (currently → ../chignolin.parm7)
├── coordinates.rst7  -> ../${SYSTEM_NAME}.rst7    (currently → ../chignolin.rst7)
├── equilibration.log
└── out/
    ├── gamd_restart.checkpoint   # the seed used by runseg.sh
    └── gamd-restart.dat          # GaMD running statistics
```

`common_files/gamd_restart.checkpoint` and `common_files/gamd-restart.dat`
are symlinks pointing into `equilibration/out/`. To re-equilibrate
(e.g., after a major OpenMM/CUDA upgrade, on a different GPU
architecture, or for a new system), repoint the topology/coordinates
symlinks if needed and rerun:

```bash
cd common_files/equilibration
ln -sfn ../${SYSTEM_NAME}.parm7 topology.parm7      # only if SYSTEM_NAME changed
ln -sfn ../${SYSTEM_NAME}.rst7  coordinates.rst7
conda activate pargamd
CUDA_VISIBLE_DEVICES=0 python ../gamdRunner_init -p CUDA xml input.xml
```

Step counts in `input.xml` are based on Miao et al. 2015 JCTC:
500k cMD-prep / 1M cMD / 500k GaMD-eq-prep / 12.5M GaMD-eq /
1M GaMD-prod, with averaging-window 500k. ~60 min wallclock on a 5090.

## Switching to a different system — full recipe

The runtime is parameterized by `$SYSTEM_NAME` (set in `env.sh`,
default `chignolin`). `runseg.sh` and `get_pcoord.sh` honor it;
`input.xml` is system-agnostic via in-segment generic names
(`topology.parm7` / `coordinates.rst7` symlinked at segment-run time).

End-to-end recipe to run a brand new protein. The whole sequence is
**~70-90 minutes wallclock** on a 5090 (most of it the Phase 1
equilibration). Replace `<name>` with whatever you want to call your
system — match it consistently across all steps.

### Quick path: `setup.sh`

Steps 2–6 below are bundled into a single command:

```bash
# After step 1 (drop common_files/<name>.{parm7,rst7,pdb} into place):
./setup.sh <name>
```

`setup.sh` writes `SYSTEM_NAME` into `env.sh`, runs Phase 1 GaMD
equilibration (~60 min), wires the seed checkpoint symlinks,
refreshes the basis-state symlink, and recomputes
`bstates/bstates.txt` pcoord values. It refuses to clobber an
existing `west.h5` or an existing seed checkpoint built from a
different system; pass `--force` to override.

You're then on the hook for step 7 (bin tuning, science decision)
and step 8 (wiping stale WE state, destructive). The manual
breakdown below is still useful when you want to do these steps
piecemeal or debug a failure.

### 1. Stage the system files (~minutes)

Drop your AMBER topology, coordinates, and a reference PDB for the
RMSD pcoord into `common_files/`:

```
common_files/<name>.parm7
common_files/<name>.rst7
common_files/<name>.pdb
```

The `.pdb` should be the structure you want to measure RMSD *against*
(e.g., the folded reference, or a known docked pose). Often it's just
the same atoms as `<name>.rst7` rendered as PDB.

### 2. Set SYSTEM_NAME in env.sh

```bash
sed -i "s/SYSTEM_NAME:-chignolin/SYSTEM_NAME:-<name>/" env.sh
grep SYSTEM_NAME env.sh    # verify
```

(Or override per-run with `SYSTEM_NAME=<name> ./run_local.sh` without
editing the file.)

### 3. Phase 1 GaMD equilibration (~60 min on 5090)

This produces the seed checkpoint that all WE walkers branch from in
iteration 1. Checkpoints are GPU-architecture-specific, so you must
build it on the same hardware you'll run production on.

```bash
cd common_files/equilibration

# repoint the generic symlinks at your system
ln -sfn ../<name>.parm7 topology.parm7
ln -sfn ../<name>.rst7  coordinates.rst7

# (optional) tune step counts in input.xml. Defaults are
# 500k cMD-prep + 1M cMD + 500k GaMD-eq-prep + 12.5M GaMD-eq +
# 1M GaMD-prod = 15.5M steps = 31 ns. Reasonable for most systems.

conda activate pargamd
CUDA_VISIBLE_DEVICES=0 python ../gamdRunner_init -p CUDA xml input.xml
cd ../..
```

Verify it produced a checkpoint:

```bash
ls -lh common_files/equilibration/out/gamd_restart.checkpoint    # should be a few MB
```

### 4. Wire up the seed checkpoint (~seconds)

`runseg.sh` looks for `common_files/gamd_restart.checkpoint` and
`common_files/gamd-restart.dat`. Repoint these symlinks at the new
equilibration's output:

```bash
cd common_files
ln -sfn equilibration/out/gamd_restart.checkpoint gamd_restart.checkpoint
ln -sfn equilibration/out/gamd-restart.dat        gamd-restart.dat
ls -L gamd_restart.checkpoint gamd-restart.dat   # verify both resolve
cd ..
```

### 5. Set up the basis state (~minutes)

WE walkers in iteration 1 branch from a *basis state* — the initial
structure stored in `bstates/bstate0/`. The basis-state coords are a
relative symlink (`bstate0/output_restart.rst7` →
`../../common_files/${SYSTEM_NAME}.rst7`) that `init.sh` refreshes on
every WE init based on the current `SYSTEM_NAME`. So as long as your
new `common_files/<name>.rst7` is in place and `SYSTEM_NAME` is set,
no manual copy is needed.

To verify (or if you want to use a different starting structure than
`common_files/<name>.rst7`):

```bash
ls -l bstates/bstate0/output_restart.rst7   # should be a symlink
# Or override with a different file:
ln -sfv ../../common_files/<other>.rst7 bstates/bstate0/output_restart.rst7
```

### 6. Recompute basis-state pcoord values

`bstates/bstates.txt` records the (RMSD, Rg) of the basis state so
WESTPA knows which bin to seed it into. Recompute for the new system:

```bash
conda activate pargamd
source env.sh                                # sets SYSTEM_NAME
export WEST_SIM_ROOT="$PWD"
export WEST_STRUCT_DATA_REF="$PWD/bstates/bstate0"
export WEST_PCOORD_RETURN=$(mktemp)
bash westpa_scripts/get_pcoord.sh
echo "Computed pcoord (RMSD Rg, in Angstroms):"
cat "$WEST_PCOORD_RETURN"
```

`get_pcoord.sh` also sources `env.sh` itself defensively, so a missed
manual `source` isn't fatal — but `SYSTEM_NAME` must be set somewhere.
A missing `SYSTEM_NAME` is now a hard error rather than a silent
fall-back to `chignolin` (the old default would mask atom-count
mismatches against whatever rst7 was in `bstate0/`).

Take that one row of values and put it in `bstates/bstates.txt`,
keeping the format `<state_id> <weight> <auxref> <rmsd> <rg>`:

```bash
# Example (replace with your actual values from above):
echo "0 1 bstate0 0.000479 4.6564" > bstates/bstates.txt
```

### 7. Tune `west.cfg` bin boundaries

Out of the box, `west.cfg` has a single bin (`-inf, inf` × `-inf,
inf`) — fine for smoke testing but *will not give meaningful WE
sampling on a real system*. Set bin edges along the RMSD and Rg axes
that bracket your system's expected pcoord range. Pure WESTPA
science decision; see the upstream docs.

### 8. Wipe stale WE state from the chignolin baseline

The repo ships with 76 iterations of chignolin data in `west.h5`.
For a fresh system, move that aside so `init.sh` rebuilds from the
new bstate:

```bash
mkdir -p .pre_refactor_backup/chignolin_baseline
mv west.h5 traj_segs seg_logs istates \
   .pre_refactor_backup/chignolin_baseline/ 2>/dev/null
```

### 9. Quick smoke test (~3-5 min)

Before committing to a long production run, sanity-check with one
iteration:

```bash
# temporarily cap the iteration count
sed -i 's/max_total_iterations: .*/max_total_iterations: 1/' west.cfg

sudo nvidia-cuda-mps-control -d   # if not already running
WORKERS_GPU0=4 ./run_local.sh

# If iter 1 completes, set the real cap and run production:
sed -i 's/max_total_iterations: 1/max_total_iterations: 200/' west.cfg
```

Verify the smoke test:

```bash
python -c "
import h5py
with h5py.File('west.h5','r') as f:
    s = f['summary'][:]
    print(f'completed iters: {sum(1 for r in s if r[\"walltime\"]>0)}')
    pc = f['iterations/iter_00000001/pcoord'][:]
    print(f'iter 1 pcoord shape: {pc.shape} (should be (24, 100, 2))')
    print(f'  RMSD range: {pc[...,0].min():.3f} - {pc[...,0].max():.3f}')
    print(f'  Rg   range: {pc[...,1].min():.3f} - {pc[...,1].max():.3f}')
"
```

Expect 24/24 segments to succeed for a well-equilibrated system. If
you see segments listed as `FROZEN` in `seg_logs/000001-*.log` (the
runseg.sh safety net firing), the parent state may have NaN-prone
configurations — consider tweaking the equilibration (longer GaMD-eq,
smaller dt; see "NaN robustness").

### 10. Production run

```bash
./run_local.sh        # uses USE_MPS=auto and the running sudo daemon
```

Monitor with `tail -f production_run.log` or `nvidia-smi`. Expect
~135 s/iter on the 5090 with 4 workers via sudo MPS for chignolin-
sized systems; bigger systems scale roughly with atom count.

### Default pcoord (RMSD, Rg of CA atoms)

`get_pcoord.sh` and the embedded MDAnalysis in `runseg.sh` compute
mass-weighted best-fit RMSD of CA atoms vs. the reference PDB and
CA-only radius of gyration. Works for most folded proteins. For
systems where these aren't the right CVs (e.g., ligand-binding
distance, RMSD of a specific domain, RoG of a flexible loop), you'd
edit the MDAnalysis blocks in those two files. That's a science-level
edit, not plumbing.
