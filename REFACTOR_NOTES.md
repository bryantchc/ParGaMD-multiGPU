# ParGaMD_MPS_TACC → Local Workstation Refactor — historical audit

> **Status**: this is the **pre-work planning doc** from the start of the
> refactor (2026-05-01). It captures the initial audit and proposed plan.
> Everything below has either shipped or been deliberately deferred — the
> *current* state of the repo and how to operate it lives in
> [`WORKSTATION.md`](WORKSTATION.md). Kept here for the audit trail.
>
> **What actually happened**, condensed:
>
> - All path-bug fixes shipped (`get_pcoord.sh` `format="INPCRD"` fallback,
>   `input.xml` relative paths, `bstates.txt` pcoords verified).
> - Local Phase 1 GaMD equilibration produced a Blackwell-compatible
>   seed checkpoint (~31 ns; the upstream TACC checkpoint at
>   `_tacc_original/seed_out27_tacc_blackwell_incompatible/` is unusable
>   on sm_120 — `loadCheckpoint()` fails with a misleading
>   `CUDA_ERROR_OUT_OF_MEMORY`).
> - `run_local.sh` shipped: ~70 LOC, processes work manager, trap-based
>   cleanup, `USE_MPS=auto` default that auto-detects a sudo-mode MPS
>   daemon at `/tmp/nvidia-mps`. User-mode MPS bug discovered and
>   documented (`OverflowError` in `Context_setStepCount` on Blackwell
>   under concurrent contexts); workaround is sudo-mode MPS.
> - `runseg.sh` shipped retry-with-new-seed and freeze-fallback safety
>   nets, plus a system-agnostic frame-count DCD validity check via
>   `struct`. `<random-seed>0</random-seed>` (auto-seed) in `input.xml`
>   is the real fix; the safety nets are insurance.
> - 76-iteration production run completed (~182 ns aggregate, 0 retries
>   fired, 0 walkers froze).
> - Genericization: `SYSTEM_NAME` env var in `env.sh`; `runseg.sh` and
>   `get_pcoord.sh` honor it; `input.xml` is now system-agnostic via
>   in-segment-dir generic names (`topology.parm7`, `coordinates.rst7`).
> - All proposed cleanup (15 files + 4 stray `d2utmp*` Python files +
>   `out{23..26}/`) deleted; `_tacc_original/` and `.pre_refactor_backup/`
>   created as documented archives.
> - **Deferred**: ZMQ-per-worker launcher for true dual-GPU (5090 + 5070).
>   Current launcher pins all workers to GPU 0. Revisit when scaling
>   to systems large enough that the 5090 alone is the bottleneck.

Audit performed on 2026-05-01 from `/home/bryantchc/MolecularDynamics/openmm/ParGaMD_MPS_TACC`.
Goal: keep the WESTPA + GaMD scientific machinery untouched; replace the TACC/SLURM/ZMQ orchestration layer.

This document was read-only at the time of writing. Files have since been modified per the plan below.

---

## TL;DR

- Seed GaMD checkpoint exists (`common_files/out27/gamd_restart.checkpoint`, 2.5 MB, symlinked at `common_files/gamd_restart.checkpoint`). **No Phase 1 equilibration needed.**
- A few items in your original task list are already partially fixed in the working tree (`bstates.txt` has pcoords; `runseg.sh` already uses `$WEST_SIM_ROOT`; `get_pcoord.sh` no longer has hardcoded scratch paths). Real outstanding bugs are narrower than expected.
- Plan: KEEP the entire `common_files/gamd/` package, `westpa_scripts/`, `west.cfg`, `bstates/`. REWRITE the launcher and patch a few small path bugs. ARCHIVE two reference TACC scripts. DELETE 14 files of TACC/profiling/duplicate cruft.

---

## File inventory

Categories: **K**=keep, **K***=keep with patch, **R**=rewrite, **A**=archive to `_tacc_original/`, **D**=delete.

### Top-level orchestration

| File | Action | Notes |
|---|---|---|
| `env.sh` | K | Already clean (no module load, activates `pargamd`, sets `WEST_SIM_ROOT`). |
| `init.sh` | K | Already clean (no module load, uses `--work-manager=threads`). |
| `run_WE_local.sh` | R | Your in-progress launcher. Will replace with `run_local.sh` (~60 lines, processes work manager, trap-based cleanup). |
| `run_WE_multi_MPS.sh` | A | Canonical TACC launcher with MPS + ZMQ. Move to `_tacc_original/` for reference. |
| `run_WE.sh` | A | Older PSC SLURM launcher. Move to `_tacc_original/` for reference. |
| `run_WE_no_MPS.sh` | D | Profiling A/B variant. SLURM-only. |
| `run_WE_profile.sh` | D | nsys-wrapped variant. SLURM-only. |
| `run_WE_multi_profile_No_MPS.sh` | D | nsys + no-MPS variant. SLURM-only. |
| `node.sh` | D | Per-node srun helper. SLURM-only. |
| `node_NO_MPS.sh` | D | Same minus MPS. SLURM-only. |
| `wrapper.sh` | D | nsys profiler wrapper. References a script that doesn't exist (`run_WE_multi_no_mps.sh`). |
| `nodefilelist.txt` | D | Single-line `sapphire-2`; populated by `scontrol`. |
| `my_profile_no_mps.nsys-rep` | D | Profiling artifact (200 KB). |
| `my_profile_no_mps.sqlite` | D | Profiling artifact (650 KB). |
| `env.sh.bak` | D | Old TACC version with module loads. |
| `init.sh.bak` | D | Old TACC version with module loads. |

### `westpa_scripts/`

| File | Action | Notes |
|---|---|---|
| `runseg.sh` | K | Already uses `$WEST_SIM_ROOT/common_files/gamdRunner` and `…/gamd_restart.checkpoint`. No path patches needed. |
| `get_pcoord.sh` | K* | Paths already `$WEST_SIM_ROOT`-relative via `os.environ`. **Real bug**: line 50 reads `output_restart.dcd`, but at basis-state time the bstate dir contains `output_restart.rst7` only. Need a branch that reads the `.rst7` with `format="INPCRD"` when the `.dcd` is absent. |
| `gen_istate.sh` | K | Trivial bstate→istate symlink. Fine. |
| `post_iter.sh` | K | Tar segment logs per iteration. Fine. |
| `tar_segs.sh` | K | Same idea for `traj_segs/`. Fine. |
| `runseg.sh.bak` | D | Old version with `/scratch/10597/anugrahat/...` paths. |
| `runseg_og.sh` | D | Original Amber/pmemd codepath; superseded by GaMD/OpenMM. |
| `d2utmpHVnbVg` | D | Stray committed temp file (looks like an early `get_pcoord.sh` draft with hardcoded `/home/anugraha/...` paths). |

### `common_files/`

Everything in `common_files/gamd/` (the GaMD package, integrators, Runner) is **KEEP, untouched**. Same for the GaMD logger / debug helpers / `_version.py` / `__init__.py`.

| File | Action | Notes |
|---|---|---|
| `gamd/` (whole tree) | K | GaMD-OpenMM package. Do not touch per your instruction. |
| `gamdRunner`, `gamdRunner_init` | K | Core executables (the only diff is `from gamd.runners import Runner` vs `runners_init`). |
| `chignolin.parm7`, `chignolin.rst7`, `chignolin.pdb` | K | System files. |
| `gamd_restart.checkpoint` (symlink → `out27/...`) | K | Seed checkpoint. **Verified present.** |
| `out27/` | K | Holds the seed checkpoint (target of the symlink) and prior GaMD outputs. |
| `gamd-restart.dat`, `temperature.dat`, `upper_dual_temp.xml` | K | Referenced by `runseg.sh` and the GaMD machinery. |
| `input.xml` | K* | **Patch**: lines 42–43 currently absolute (`/home/bryantchc/.../common_files/...`). `runseg.sh` already symlinks `chignolin.parm7` and `chignolin.rst7` into the segment dir, so these should be `chignolin.parm7` / `chignolin.rst7` (relative). |
| `input_init.xml` | K | Phase 1 equilibration template. Not needed now (seed checkpoint exists), but cheap to keep. |
| `input.xml.bak` | D | Old TACC `/scratch/10597/...` paths. |
| `out23/`, `out24/`, `out25/`, `out26/` | **? — please confirm** | Look like older GaMD runs. `out26/gamd_restart.checkpoint` is 0 bytes. Default plan: **keep** (small, harmless). Tell me if you want them gone. |
| `gamd_prerun.sh` | D | TACC SLURM batch wrapper for one-off `gamdRunner` invocations. Not used by the WE pipeline. |
| `gamd/langevin/d2utmpPnF9y3`, `…WbEW5N`, `…YZ6MBZ`, `…zqwKBm` | D | Stray committed Python temp files (4 of them). They are not imported anywhere. |

### `bstates/`

| File | Action | Notes |
|---|---|---|
| `bstates.txt` | K* | Currently `0 1 bstate0 0.0005 4.6564`. **Verify**: RMSD≈0 (bstate IS the reference, so this is correct) and Rg≈4.66 Å (sensible for folded chignolin). I'd like to recompute deterministically with `mda.Universe(parm7, rst7, format="INPCRD")` to confirm. |
| `bstate0/output_restart.rst7` | K | Starting structure. |
| `chignolin.parm7`, `chignolin.rst7` | K | Duplicates of the `common_files/` versions; kept here because `west.cfg`'s `data_refs.basis_state` resolves to `bstates/{auxref}` and the `runseg.sh` symlinks pull from `common_files/`. Not worth deduplicating. |
| `d2utmph7Hdt7` | D | Stray committed temp file (an old version of `bstates.txt` without pcoords). |

### Other

| File | Action | Notes |
|---|---|---|
| `west.cfg` | K (untouched) | Per your instruction. |
| `tstate.file` | K | One-line `folded 0.49`. Currently unused (commented out in `init.sh`) but harmless. |
| `west.h5`, `west-master.log` | (runtime) | Will be cleared on the next `init.sh` run; the new launcher will only call `init.sh` if `west.h5` is missing, so existing runs are restartable. |
| `traj_segs/`, `seg_logs/`, `istates/` | (runtime) | Empty now. Recreated by `init.sh`. |
| `reweigh/reweigh.py`, `reweigh/reweigh_CE.py`, `reweight-2d.sh`, `run_data.sh` | K | Post-run analysis scripts. Out of scope for this refactor; may need their own path-cleanup pass later. |
| `README.md` | K* | Keep upstream content (especially the reweighting section). Add a `WORKSTATION.md` for the local-execution story instead of rewriting in place. |

---

## Outstanding bugs to fix (Task 2)

These are the only code changes I'd make in Task 2 — small and well-scoped:

1. **`westpa_scripts/get_pcoord.sh`** — basis-state branch.
   When `output_restart.dcd` does not exist in `$WEST_STRUCT_DATA_REF` (i.e., we're computing pcoord for a basis state, not a finished segment), fall back to reading the `.rst7` with `mda.Universe(parm7, rst7, format="INPCRD")` and compute a single (RMSD, Rg) pair. Same MDAnalysis math as the segment branch.

2. **`common_files/input.xml`** — make topology/coordinates relative.
   Lines 42–43: `/home/bryantchc/.../common_files/chignolin.parm7` → `chignolin.parm7`, same for `.rst7`. `runseg.sh` already symlinks both into the segment dir before invoking gamdRunner.

3. **`bstates/bstates.txt`** — verify pcoord values.
   Recompute (RMSD, Rg) once with MDAnalysis using `format="INPCRD"`; overwrite if different from the existing `0.0005 4.6564`.

Nothing else is blocking iteration 1 once the launcher is rewritten.

---

## New launcher (Task 4) — sketch

`run_local.sh`, ~60 lines, structured as:

```text
1. Strict mode (set -euo pipefail), trap EXIT for cleanup.
2. cd to script dir; export WEST_SIM_ROOT=$PWD.
3. Source ~/.bashrc; conda activate pargamd.
4. Config block at top: CUDA_VISIBLE_DEVICES (default "0"),
   WORKERS_5090, WORKERS_5070, MPS_THREAD_PCT.
5. Per visible GPU: start nvidia-cuda-mps-control -d with
   isolated CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY.
6. Set MPS active_thread_percentage proportional to walker count.
7. If west.h5 is missing → ./init.sh. Otherwise restart in place.
8. source env.sh.
9. Compute total_workers from per-GPU counts; w_run with
   --work-manager=processes --n-workers=$total_workers.
   (No ZMQ master/client split needed on a single host.)
10. Background nvidia-smi utilization logger to gpu_util.log.
11. wait. On exit (success, error, or Ctrl-C): trap stops the
    logger and sends "quit" to each MPS daemon.
```

The trade-off vs your current `run_WE_local.sh`: the processes work manager doesn't let us pin individual workers to specific MPS pipe dirs as cleanly as the ZMQ-per-worker subshell pattern. **For a single-GPU run (CUDA_VISIBLE_DEVICES=0) this is fine** — all workers share the one MPS daemon. For dual-GPU we'd need a small wrapper script per worker that sets `CUDA_MPS_PIPE_DIRECTORY` based on a slot index, OR fall back to the ZMQ pattern. I'll start with single-GPU + processes; we can graduate to dual-GPU later.

If you'd rather I keep the per-worker ZMQ-client pattern from your current script (which already does correct per-GPU pipe isolation), say so and I'll instead clean up that file in place. Either is fine; the processes path is just simpler.

---

## Archive layout

```
_tacc_original/
  run_WE.sh
  run_WE_multi_MPS.sh
  README.md          # one-liner: "Original TACC launchers, kept for reference."
```

---

## Confirmation needed before I touch anything

1. **Approve the delete list above** (15 files + 4 stray `d2utmp*` Python temp files inside `gamd/langevin/`). All recoverable from git history.
2. **`common_files/out{23..26}/`** — keep, or delete? Default = keep.
3. **Launcher direction**: processes work manager (simpler, single-GPU first) vs. cleaning up your existing ZMQ-per-worker script. Default = processes.
4. **`bstates.txt` pcoords**: OK if I recompute and overwrite if different? Default = yes.

Reply with a green light (and any of the four answers above) and I'll proceed with Task 2 (the path-bug fixes) before touching the launcher or doing any deletions.
