# Session handoff — ParGaMD_MPS_TACC local-workstation refactor

This file is a self-contained briefing to onboard a fresh AI session (or
future-self) into the state of this fork. Read this first; everything
else (repo layout, knobs, troubleshooting) is in
[WORKSTATION.md](WORKSTATION.md). The pre-work audit is in
[REFACTOR_NOTES.md](REFACTOR_NOTES.md).

## TL;DR

Upstream ParGaMD (TACC SLURM/ZMQ) has been adapted to run on a single
Linux box. The pipeline is **production-ready for chignolin** — a
76-iteration run completed end-to-end with zero errors, zero retries,
and zero walker freezes. Repo is also genericized for arbitrary
systems via `$SYSTEM_NAME`. A handful of items remain open; see
"Outstanding" below.

## Hardware / environment

| | |
|---|---|
| CPU | AMD Threadripper 9960X (24 cores) |
| RAM | 128 GB |
| GPU 0 | NVIDIA RTX 5090, 32 GB, Blackwell (sm_120) |
| GPU 1 | NVIDIA RTX 5070, 12 GB, Blackwell (sm_120) — desktop GPU, currently unused for compute |
| OS | Ubuntu, NVIDIA driver supporting sm_120 |
| Conda env | `pargamd` — Python 3.11, OpenMM 8.4 (CUDA), WESTPA 2022.x, MDAnalysis, AmberTools, h5py |
| Repo path | `/home/bryantchc/MolecularDynamics/openmm/ParGaMD_MPS_TACC` |

## What's working

- **`run_local.sh`** (~70 LOC): single-host launcher. `USE_MPS=auto`
  default — auto-detects sudo MPS daemon at `/tmp/nvidia-mps`, uses it
  if alive (1607 ns/day aggregate at 4 workers), falls back to no-MPS
  silently if absent (309 ns/day). Trap-based cleanup. Restart-safe
  (only fires `init.sh` when `west.h5` is missing).

- **Phase 1 GaMD equilibration** at `common_files/equilibration/`
  produced a Blackwell-compatible seed checkpoint. The upstream TACC
  seed is at `_tacc_original/seed_out27_tacc_blackwell_incompatible/`
  and fails on sm_120 with a misleading
  `CUDA_ERROR_OUT_OF_MEMORY` from `loadCheckpoint`.

- **`runseg.sh`** ships three layered safety nets against NaN failures:
  (1) DCD frame-count validity check via `struct` (system-agnostic),
  (2) 3× retry-with-new-seed, (3) freeze-fallback (copy parent state
  forward; logs `FROZEN`). The actual fix that made the run robust
  was **`<random-seed>0</random-seed>`** in `common_files/input.xml`
  (auto-seed each Context). Production: 0 retries, 0 freezes / 76 iter.

- **`$SYSTEM_NAME` genericization**: env var in `env.sh` (default
  `chignolin`). `runseg.sh` symlinks `common_files/${SYSTEM_NAME}.parm7`
  / `.rst7` to generic per-segment names `topology.parm7` /
  `coordinates.rst7`. `input.xml` references only the generic names —
  so it's system-agnostic. Verified end-to-end with a `SYSTEM_NAME=testfold`
  smoke run pointing at chignolin via symlinks.

- **`bstates/` genericized**: dead `bstates/chignolin.{parm7,rst7}`
  (unreferenced top-level duplicates) removed.
  `bstates/bstate0/output_restart.rst7` is now a relative symlink to
  `common_files/${SYSTEM_NAME}.rst7`, refreshed by `init.sh` on every
  run. `get_pcoord.sh` and `runseg.sh` source `env.sh` defensively and
  hard-error if `SYSTEM_NAME` is unset (old silent `chignolin`
  fallback masked atom-count mismatches when setting up new systems —
  see Cas12a/lanmPace setup attempt).

- **Cleanup done**: 21 TACC/profiling/.bak/d2utmp files deleted;
  `_tacc_original/` (reference launchers + unusable TACC checkpoint)
  and `.pre_refactor_backup/` (broken WE state from refactor smoke
  tests) created with READMEs.

## Standard run (chignolin, sudo MPS)

```bash
sudo nvidia-cuda-mps-control -d           # once per session
./run_local.sh                            # auto-detects, uses MPS
echo quit | sudo nvidia-cuda-mps-control  # when done (optional)
```

Throughput on chignolin: 587 ns/day single-worker, **1607 ns/day
aggregate at 4 workers via sudo MPS** (~75% efficient scaling).

## Starting a fresh run on a NEW system

**Quick path:** drop `common_files/<name>.{parm7,rst7,pdb}` into
place, then `./setup.sh <name>` bundles steps 2–6 (env.sh edit,
Phase 1 equilibration, seed wiring, pcoord pre-compute). You then
do step 7 (bin tuning) and step 8 (wipe stale state) manually,
then `./run_local.sh`. Full step-by-step in
**[WORKSTATION.md "Switching to a different system — full recipe"](WORKSTATION.md#switching-to-a-different-system--full-recipe)**.
Summary of the 10 steps:

1. Drop `<name>.{parm7,rst7,pdb}` into `common_files/`
2. Set `SYSTEM_NAME` in `env.sh`
3. **Phase 1 equilibration** (~60 min): repoint
   `common_files/equilibration/{topology.parm7,coordinates.rst7}`
   symlinks at the new system, run `gamdRunner_init`
4. Wire up the **seed checkpoint symlinks** in `common_files/`
   (`gamd_restart.checkpoint`, `gamd-restart.dat`) → into
   `equilibration/out/`
5. ~~Replace `bstates/bstate0/output_restart.rst7`~~ — handled by
   `init.sh` automatically via a relative symlink to
   `common_files/${SYSTEM_NAME}.rst7`. No manual step.
6. **Recompute basis-state pcoord** (RMSD, Rg) and update
   `bstates/bstates.txt`. `source env.sh` first so `SYSTEM_NAME`
   is set; `get_pcoord.sh` now hard-errors when it isn't (the old
   silent `chignolin` fallback masked system-swap mistakes).
7. Tune `west.cfg` bin boundaries to the new pcoord range (science
   decision)
8. Wipe stale WE state (`mv west.h5 traj_segs seg_logs istates`
   into `.pre_refactor_backup/`)
9. Smoke test (`max_total_iterations: 1`, run, verify)
10. Set production iteration cap, run

Total wallclock: ~70-90 min (Phase 1 equilibration dominates).

## Outstanding / open

1. **Dual-GPU is broken-by-design** in current `run_local.sh`.
   `--work-manager=processes` can't pin individual workers to
   different GPUs; the `WORKERS_GPU1` knob in the WORKSTATION.md
   Knobs table inflates worker count without actually using GPU 1.
   **Decision pending**:
   - **(A) explicit single-GPU only** — error if
     `CUDA_VISIBLE_DEVICES` has >1 device, drop `WORKERS_GPU1` from
     the table. Honest, smaller surface.
   - **(B) keep variable names for future-ZMQ migration** — add
     runtime guard, document `WORKERS_GPU1` as reserved. No
     code-renaming when ZMQ is later added.
   - User leaning toward A; awaiting confirmation.

2. **ZMQ-per-worker launcher (deferred)** — the only way to get
   true dual-GPU (5090 + 5070). Reference pattern in
   `_tacc_original/run_WE_multi_MPS.sh` minus the SLURM scaffolding
   (~80-100 LOC). Worth doing when scaling to systems where the 5090
   alone is the bottleneck (Cas-scale ~30k+ atom systems). Adding
   the 5070 to a Cas9 run would gain ~30-50% aggregate (not 100% —
   5070 is ~2-3× weaker than the 5090).

3. **Per-segment minimization** — `<run-minimization>True</run-minimization>`
   is silently ignored on extension runs because gamdRunner's
   extension code path (`common_files/gamd/runners.py:~440-510`)
   creates a fresh Simulation but never calls `simulation.minimizeEnergy()`.
   Patching that one line into the gamd source would enable it for
   WE segments. **Out of scope for this refactor** (don't modify
   `common_files/gamd/`), but a cheap tweak worth knowing if NaN
   rate is high on a tougher system.

4. **MPS user-mode bug, not yet filed upstream**. Repro:
   `USE_MPS=1` (per-GPU user-mode daemon, started by the launcher) +
   ≥2 concurrent gamdRunner processes + GaMD CustomIntegrator +
   `loadCheckpoint` on Blackwell ⇒ `OverflowError: in method
   'Context_setStepCount', argument 2 of type 'long long'`. Sudo-mode
   MPS at `/tmp/nvidia-mps` is unaffected. File a minimal repro
   upstream when convenient — narrow combination, would be useful
   for OpenMM maintainers.

5. **README.md upstream body** — two stale upstream-cluster references
   in the Installation/Running sections (`sbatch
   common_files/gamd_prerun.sh`, `run run_WE.sh`) that point at files
   we've moved/deleted. Top-of-file pointer to WORKSTATION.md
   redirects readers correctly, so functionally OK, but technically
   broken commands. Easy patch (1-line inline disclaimers) on
   request — left in place per the original "keep upstream content
   intact" instruction.

## Hard rules from original brief (still in effect)

- **Don't modify** `common_files/gamd/`, `common_files/gamdRunner*`,
  WESTPA internals, or `west.cfg` scientific parameters (bins,
  pcoord_ndim, bin_target_counts). The refactor is plumbing, not
  science.
- **Pause before destructive actions**: deletes, overwrites of
  `west.h5`, long simulations.

## Current `west.h5` state (as of last session)

- 76 iterations completed (= ~182 ns aggregate sampling on chignolin)
- `max_total_iterations: 76` in `west.cfg`
- To extend: bump that, run `./run_local.sh`
- To start fresh: see "Starting a fresh run on a NEW system" above

## Key files (quick orientation)

| File | What's in it |
|---|---|
| `setup.sh` | One-shot system setup. Bundles WORKSTATION steps 2–6 (env.sh edit, Phase 1 equilibration, seed symlinks, pcoord pre-compute). Refuses to clobber existing west.h5 / wrong-system seed without `--force`. |
| `run_local.sh` | The launcher. ~70 LOC, processes work manager, USE_MPS=auto, trap cleanup. Warns (doesn't block) on placeholder `[-inf, inf]` west.cfg bins. |
| `env.sh` | Exports `SYSTEM_NAME` (default `chignolin`) and WESTPA env vars |
| `init.sh` | One-shot `w_init`; called by `run_local.sh` only when `west.h5` is absent |
| `west.cfg` | WESTPA config — bins, pcoord shape, iteration cap. **Science; don't touch except bin tuning.** |
| `common_files/input.xml` | GaMD config. Generic file references (`topology.parm7` / `coordinates.rst7`); `random-seed=0`; `run-minimization=True` (no-op on extensions, see #3 above) |
| `common_files/${SYSTEM_NAME}.{parm7,rst7,pdb}` | System bundle |
| `common_files/gamd_restart.checkpoint` | Symlink to `equilibration/out/...` — the seed walkers branch from |
| `common_files/equilibration/` | Phase 1 equilibration setup; produces the seed checkpoint |
| `westpa_scripts/runseg.sh` | Per-segment propagator. Has the retry/freeze safety nets and the frame-count validity check. |
| `westpa_scripts/get_pcoord.sh` | Pcoord computation; honors `SYSTEM_NAME` via `os.environ` |
| `WORKSTATION.md` | Local-execution doc — knobs, MPS modes, troubleshooting, full system-swap recipe |
| `REFACTOR_NOTES.md` | Pre-work audit + completion summary header |
| `_tacc_original/` | Original SLURM launchers + Blackwell-incompatible TACC seed checkpoint |
| `.pre_refactor_backup/` | Three named snapshots from refactor smoke tests; safe to delete |
| `HANDOFF.md` | This file |
