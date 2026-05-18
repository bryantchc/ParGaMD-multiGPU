# ParGaMD — Parallelizable Gaussian Accelerated Molecular Dynamics

A hybrid enhanced-sampling method combining **GaMD** (Gaussian Accelerated
Molecular Dynamics — adds a harmonic boost potential to flatten energy
barriers) with **WESTPA** (Weighted Ensemble — many walkers exploring in
parallel along user-defined collective variables). Runs on a single Linux
workstation with one or more NVIDIA GPUs, no SLURM required. Recovers
unbiased free energies via established reweighting protocols.

Reference: Sonti, Thyagatur, Wan, et al., *Accelerating free energy
exploration using parallelizable Gaussian accelerated molecular dynamics
(ParGaMD)*, ChemRxiv (2025).
DOI: [10.26434/chemrxiv-2025-rr5v9](https://doi.org/10.26434/chemrxiv-2025-rr5v9).

---

## Quick start

You have a system as `<name>.{parm7,rst7,pdb}` and want to run ParGaMD on it.

```bash
# 0. One-time: install dependencies (see Installation below)

# 1. Park the system files
mkdir -p ~/pargamd_systems/<name>
cp /path/to/<name>.{parm7,rst7,pdb} ~/pargamd_systems/<name>/

# 2. Scaffold a fresh run directory (instant)
./new_run.sh ~/runs/<name>-prod \
    --system <name> \
    --system-from ~/pargamd_systems/<name> \
    --skip-equil

# 3. Edit equilibration GaMD params for your system (temperature, dt, sigma0,
# total step counts, etc.)
$EDITOR ~/runs/<name>-prod/equilibration/input.xml

# 4. Equilibrate (~60 min on a 5090; also computes basis-state pcoord)
cd ~/runs/<name>-prod && ./equilibrate.sh

# 5. Edit per-segment production params (dt, extension-steps, etc.)
$EDITOR input.xml

# 6. Edit WE bin grid for your system's pcoord range
$EDITOR west.cfg

# 7. (Optional) Pick the best worker split for your hardware (~20 min)
./bench/run_bench.sh

# 8. (Optional, but recommended for solvated systems > ~50 k atoms)
# Enable background trajectory stripping to keep disk in check
sed -i 's|STRIP_AFTER_ITERS:-0|STRIP_AFTER_ITERS:-1|' env.sh

# 9. Production
USE_MPS=1 CUDA_VISIBLE_DEVICES=0,1 WORKERS_GPU0=12 WORKERS_GPU1=4 ./run_local.sh

# 10. When the run is truly done, mop up the final held-back iterations
# (skip if you didn't enable stripping in step 8)
./westpa_scripts/strip_iter.sh --finalize
```

Each `~/runs/<name>-*/` is a fully self-contained simulation directory.
Multiple runs in parallel just need different directories.

---

## Repository layout

The repo is **code only**. Simulation runs live elsewhere (in
`~/runs/<name>/`) and are scaffolded by `new_run.sh`.

```
pargamd/                          # this repo
├── runtime/                      # the gamd-openmm package + gamdRunner scripts
├── westpa_scripts/               # runseg.sh, get_pcoord.sh — invoked by WESTPA
├── bench/run_bench.sh            # worker-distribution throughput sweep
├── templates/                    # per-system templates copied into each run dir
│   ├── env.sh.template
│   ├── west.cfg.template
│   ├── input.xml.template                  # production GaMD per-segment
│   ├── equilibration_input.xml.template    # Phase-1 GaMD equilibration
│   ├── cv_0.py.template                    # CV definitions, one per pcoord dim
│   ├── cv_1.py.template                    #   default: CA-RMSD + CA Rg
│   └── bstates.txt.template
├── new_run.sh                    # scaffold + first-time equilibrate
├── equilibrate.sh                # (re-)equilibrate a scaffolded run dir
├── run_local.sh                  # ZMQ launcher (master + N pinned workers)
├── init.sh                       # WESTPA w_init wrapper (idempotent)
├── analyze/                      # post-run analysis (PMF reweighting)
├── reweigh/                      # upstream reweighting tools (1D, 2D, CE)
├── README.md                     # this file
└── BLACKWELL_NOTES.md            # hardware quirks + multi-GPU tuning rationale
```

A scaffolded run directory looks like this:

```
~/runs/<name>-prod/
├── env.sh                        # SYSTEM_NAME, paths, ZMQ tuning (editable)
├── west.cfg                      # WE bin grid (edit for your system!)
├── input.xml                     # production GaMD (edit per system)
├── equilibration/input.xml       # equilibration GaMD (edit per system)
├── cv_0.py, cv_1.py, ...         # collective-variable definitions (edit per system)
├── system/<name>.{parm7,rst7,pdb}     # copy from --system-from
├── bstates/bstates.txt           # written by equilibrate.sh
├── equilibration/out/            # gamd_restart.checkpoint, gamd-restart.dat
├── gamd_restart.checkpoint → equilibration/out/...   # symlink, auto-wired
├── runtime → /path/to/repo/runtime                   # code symlinks
├── westpa_scripts → /path/to/repo/westpa_scripts
├── bench → /path/to/repo/bench
├── run_local.sh, equilibrate.sh, init.sh             # script symlinks
├── [west.h5, traj_segs/, seg_logs/, gpu_util.log]    # generated at runtime
└── [strip.log, system/<name>.stripped.parm7]         # if STRIP_AFTER_ITERS=1
```

---

## Installation

### Prerequisites
- Python ≥ 3.9, Anaconda / Miniconda
- NVIDIA GPU with CUDA driver (see [BLACKWELL_NOTES.md](BLACKWELL_NOTES.md)
  for consumer-Blackwell-specific notes)

### Conda env
```bash
conda create -n pargamd python=3.11
conda activate pargamd
conda install -c conda-forge openmm mdanalysis mdtraj ambertools
pip install westpa numpy matplotlib h5py
```

### Repo
```bash
git clone <this-repo-url> ~/code/pargamd
cd ~/code/pargamd
```

---

## Detailed usage

### Setting up a new system

You have a topology `villin.parm7`, restart `villin.rst7`, and reference
PDB `villin.pdb`. All three filenames must share the basename you'll pass
as `--system`.

**Step 1: place the system files**

Two options:
- **Library style** (recommended if you'll have multiple systems):
  `~/pargamd_systems/villin/villin.{parm7,rst7,pdb}` — then
  `--system-from ~/pargamd_systems/villin` later.
- **Ad hoc**: any directory, then `--system-from /that/dir`.

`new_run.sh` *copies* (not symlinks) these into `<run_dir>/system/`, so
the run is self-contained.

**Step 2: scaffold the run directory**

```bash
./new_run.sh ~/runs/villin-test01 \
    --system villin \
    --system-from ~/pargamd_systems/villin \
    --skip-equil
```

`--skip-equil` lets you edit equilibration parameters before running it.
Drop the flag if defaults are fine; equilibration will run automatically
at the end of `new_run.sh`.

Available `new_run.sh` flags:
| flag | purpose |
|---|---|
| `--system <name>`        | required — sets SYSTEM_NAME in env.sh |
| `--system-from <path>`   | source dir for the three system files; default `../pargamd_systems/<name>` |
| `--skip-equil`           | scaffold only; you run `./equilibrate.sh` later |
| `--seed-from <dir>`      | import an existing equilibration's outputs (skip the 60 min) |

**Step 3: edit `equilibration/input.xml`**

Per-system tunables in equilibration:

| element | default (chignolin) | tune when |
|---|---|---|
| `<temperature>`               | 300 K          | match your target ensemble |
| `<dt>`                        | 0.002 ps       | shorter for stiff systems / no SHAKE |
| `<sigma0p>`, `<sigma0d>`      | 6.0 kcal/mol   | smaller for stiffer systems (= less aggressive boost) |
| `<number-of-steps>` block     | 200k/800k/200k/800k (cMD-prep / cMD / GaMD-prep / GaMD) | scale up for larger systems |
| `<friction>`                  | 1.0 ps⁻¹       | rarely changed |
| `<barostat>` element          | NPT or NVT     | match your equilibration ensemble |

You don't need to touch path elements (`topology.parm7`,
`coordinates.rst7`) — `equilibrate.sh` symlinks those against
`$SYSTEM_NAME` at runtime.

**Step 4: equilibrate**

```bash
cd ~/runs/villin-test01
./equilibrate.sh                  # ~60 min on a 5090; also writes bstates.txt
```

On success: `equilibration/out/gamd_restart.checkpoint` is produced and
symlinked into the run dir, and `bstates/bstates.txt` is populated with
the basis-state pcoord. The script is **idempotent**: re-running it with
an existing seed just refreshes `bstates.txt` (cheap). Pass `--force` to
actually re-equilibrate.

If equilibration fails (NaN, instability), check
`equilibration/equilibration.log`, tune `equilibration/input.xml`
(typically smaller `dt`, smaller `sigma0`, longer cMD-prep), and re-run
with `./equilibrate.sh --force`.

**Step 5: edit `input.xml` (production)**

Per-segment GaMD parameters that affect throughput and per-segment time
resolution:

| element | default | notes |
|---|---|---|
| `<temperature>`         | 300 K               | match equilibration |
| `<dt>`                  | 0.002 ps            | match equilibration |
| `<sigma0p>`, `<sigma0d>`| 6.0 kcal/mol        | match equilibration |
| `<extension-steps>`     | 50000 (= 100 ps)    | per-WE-segment length |
| `<reporting-rate>`      | 500 (= 1 ps)        | pcoord frames per segment |
| `<random-seed>`         | any int             | `runseg.sh` rotates this on retry |

**Step 6a: edit `cv_*.py` (collective variables)**

The default scaffold ships two CVs — `cv_0.py` (mass-weighted CA-RMSD vs
the reference PDB) and `cv_1.py` (mass-weighted CA Rg). For other systems
you'll likely want different CVs: distance between active-site residues
for an enzyme, contact-map metric for a complex, dihedral for a
conformational switch, etc.

Each `cv_N.py` defines:
```python
def compute(u, ref):
    # u is an MDAnalysis Universe positioned at the current frame.
    # ref is the reference PDB as a Universe, or None.
    # Return one float.
    ...
```

Add `cv_2.py`, `cv_3.py`, ... for more pcoord dimensions (lex-sorted by
filename, mapping to pcoord dim 0, 1, ...). Use `cv_00.py` zero-padding
if you ever need ≥10 CVs.

The dispatcher (`westpa_scripts/_pcoord_dispatch.py`) loads MDAnalysis
and the Universe once per walker, then calls each module's `compute()`
per frame — keeps Python startup + import + parse out of the inner loop
even with many CVs.

**Step 6b: edit `west.cfg` (WE bin grid)**

The default boundaries are `[0, 8 Å]` with 0.2 Å spacing — chignolin-tuned.
For most other systems you'll need to widen. Use the basis-state pcoord
in `bstates/bstates.txt` (printed by `equilibrate.sh`) as a starting
point for the bin range you want to cover.

Edit the two `boundaries:` lists (one per pcoord dim):

```yaml
boundaries:
  - ['-inf', 0.0, 0.5, 1.0, ..., 25.0, 'inf']    # pcoord dim 0 (RMSD)
  - ['-inf', 5.0, 5.5, 6.0, ..., 15.0, 'inf']    # pcoord dim 1 (Rg)
```

Rules of thumb:
- Cover the range you expect the system to explore (folded ↔ unfolded etc.)
- Bin spacing ≈ thermal RMSD fluctuation (~0.2–1 Å)
- Total bins < ~5000 for tractable WESTPA bookkeeping
- `bin_target_counts: 4` is fine for most systems

`max_total_iterations` and `max_run_wallclock` are also in `west.cfg`;
edit to taste.

**Step 7: (optional) benchmark worker distribution**

```bash
./bench/run_bench.sh                  # full 11-config sweep, ~20 min
# Or quick check:
BENCH_CONFIGS="8,0 12,0 12,4" BENCH_SEGS_PER_WORKER=2 ./bench/run_bench.sh
```

Output is a markdown table; pick the config with the highest **fed_ns_day**
whose `g0_s` and `g1_s` are within ~20% of each other (well-balanced).
For chignolin on our 5090+5070 box this was `12,4`. For other systems
it may differ.

See [BLACKWELL_NOTES.md](BLACKWELL_NOTES.md) for the methodology behind
the two throughput metrics and why MPS=1 is the default.

**Step 8: (optional) enable trajectory stripping**

For systems > ~50 k atoms (Cas9 / Cas12a / LanM / membrane proteins),
solvent dominates trajectory bytes — turn on background stripping now to
keep disk usage in check during the run. Edit `env.sh` (or sed):

```bash
sed -i 's|STRIP_AFTER_ITERS:-0|STRIP_AFTER_ITERS:-1|' env.sh
```

You can also change `STRIP_MASK` in `env.sh` if your system has unusual
solvent or counterion residue names; the default `:WAT,Na+,Cl-,K+`
covers standard TIP3P + monovalents and preserves catalytic divalents
(Mg²⁺, Zn²⁺, Ca²⁺). Skip this step for chignolin-scale systems where
disk isn't tight — there's no behavioral downside, but the cpptraj
overhead is wasted on tiny systems. See [Disk management](#disk-management-trajectory-stripping-opt-in)
below for the full mechanism.

**Step 9: production**

```bash
USE_MPS=1 CUDA_VISIBLE_DEVICES=0,1 WORKERS_GPU0=12 WORKERS_GPU1=4 ./run_local.sh
```

`run_local.sh` will:
- Start one user-mode MPS daemon per visible GPU (auto cleaned up on exit)
- Launch a ZMQ master + N pinned workers (one per (gpu, slot) pair)
- Call `./init.sh` to create `west.h5` from `bstates/bstates.txt` if
  none exists
- Stream GPU utilization to `gpu_util.log`
- Catch Ctrl-C and drain WESTPA cleanly so iterations end on a boundary

**Step 10: when the run is truly done, mop up the held-back iters**

Only relevant if you enabled stripping in step 8. During the live run,
stripping always lags by 2 iters so the freeze-fallback path in
`runseg.sh` stays safe. When you're sure you won't run more iterations,
strip the held-back tail:

```bash
./westpa_scripts/strip_iter.sh --finalize     # idempotent; safe to re-run
```

Watch progress / errors in `strip.log`. The full mechanism (lock files,
atomic mv, restart safety) is described in
[Disk management](#disk-management-trajectory-stripping-opt-in) below.

### Restarting a stopped or crashed run

Just re-invoke `run_local.sh`. If `west.h5` exists it resumes from the
last completed iteration. Walker logs are appended (the old `west_master.log`
is overwritten — copy it aside if you need it).

If stripping was on, the N-2 lag means iters N-1 and N stayed full-atom
during the original run — so the freeze-fallback in iter N+1 still has
the parent DCD it needs to copy forward. Already-stripped iters
(.stripped markers) are skipped on subsequent strip invocations. **Do
NOT run `--finalize` if you intend to resume** — that strips the
held-back iters that the next iteration's freeze-fallback might need.

### Switching to a different system

Don't edit an existing run dir — make a new one:

```bash
./new_run.sh ~/runs/<other_system>-prod --system <other_system> ...
```

Each system gets its own clean run dir.

---

## Reweighting (recovering unbiased free energies)

GaMD adds a harmonic boost ΔV(**r**) = ½ k (E − V(**r**))² when V(**r**)
falls below E. Recovery of the unbiased free-energy surface requires
reweighting that accounts for both the GaMD boost and the WE walker weights.

Two methods, picked by the shape of the ΔV distribution in each bin:

| ΔV distribution                       | Method | Script |
|---------------------------------------|--------|--------|
| Broad / non-Gaussian (e.g. chignolin, PPARα) | Maclaurin series | [`reweigh/reweigh.py`](reweigh/reweigh.py) |
| Near-Gaussian (e.g. α-synuclein-PAL)  | Cumulant expansion | [`reweigh/reweigh_CE.py`](reweigh/reweigh_CE.py) |

### Maclaurin (recommended default)

Approximates the Boltzmann factor directly:

$$\langle e^{\beta \Delta V} \rangle \approx \sum_{k=0}^{n} \frac{\beta^k \langle \Delta V^k \rangle}{k!}$$

Recommended expansion order n=10. Each frame's reweight factor combines
the GaMD boost and the WE walker weight; per-bin sums give the canonical
probability, from which the PMF follows.

```bash
python reweigh/reweigh.py \
    --input merged_data.dat --order 10 --T 300 \
    --discX 0.5 --discY 0.5 --Xdim 0 10 --Ydim 0 12 --Emax 10.0
```

Input format (whitespace-delimited): `CV1  CV2  DeltaV(kcal/mol)  WE_weight`.
Output: `pmf-<input>.xvg` with `X_center  Y_center  PMF(kcal/mol)`.

### Cumulant expansion

When ΔV is near-Gaussian, the C2 / C3 cumulant expansion is more
efficient and well-behaved. See [`reweigh/reweigh_CE.py`](reweigh/reweigh_CE.py)
and the original ParGaMD paper for the math.

### Chignolin-specific examples

The repo also ships a working chignolin example built on top of these
generic reweighters:

- [`analyze/pmf_mc10.py`](analyze/pmf_mc10.py) — extracts (RMSD, Rg, ΔV,
  WE_weight) from `west.h5`, applies MC-10 reweighting, plots the
  2D PMF. Use as a starting point for your own system's PMF script.
- [`reweight_compare.py`](reweight_compare.py) — side-by-side comparison
  of WE-only, C2, and MC-10 PMFs for sanity checking.

---

## Disk management: trajectory stripping (opt-in)

For large systems (~200k+ atoms with explicit solvent), raw trajectories
fill disk fast — a 250-iter Cas9-class run with ~16 walkers/iter is
~100 GB raw, of which ~90% is water. `westpa_scripts/strip_iter.sh`
strips solvent via cpptraj for ~10–60× disk reduction; opt in by setting
`STRIP_AFTER_ITERS=1` in `env.sh`.

**How it works during a live run.** When stripping is on, the WESTPA
`post_iteration` hook fires `strip_iter.sh --iter N-2` in the background
after every iter completes. The 2-iter lag is deliberate: the
freeze-fallback in `runseg.sh` reads iter N-1's full-atom DCD when iter N
has all retries fail, so iters N-1 and N must stay full-atom to keep
that path safe across crashes and resumes. Stripping never blocks the
WESTPA master — it's fire-and-forget into `strip.log`.

**When you're done with a run** (no more iters planned, ready for
analysis), strip the held-back final iters too:

```bash
cd ~/runs/<name>-prod
./westpa_scripts/strip_iter.sh --finalize     # strips every not-yet-stripped iter
```

**Configuration** (in `env.sh`):
- `STRIP_AFTER_ITERS=1` — turn on background stripping (default `0`)
- `STRIP_MASK=":WAT,Na+,Cl-,K+"` — cpptraj atom-mask to REMOVE. Default
  strips water + standard counterions. Catalytic divalents (Mg²⁺, Zn²⁺,
  Ca²⁺) are preserved — important for Cas systems with Mg²⁺-dependent
  cleavage chemistry.

**Outputs.** Stripped DCDs replace the originals (with a `.stripped`
marker file alongside, so re-runs are idempotent). A stripped topology
appears in two places:
- `system/<name>.stripped.parm7` (canonical, next to the full topology)
- `traj_segs/<name>.stripped.parm7` (convenience copy for analysis scripts)

`cv_*.py` and analysis code work transparently against the stripped
data — just load `<name>.stripped.parm7` instead of `<name>.parm7`. CV
definitions that select on `name CA` or residue identity don't change.

## Health monitoring (PSU + thermals)

`run_local.sh` polls GPU telemetry every 10 s into `gpu_util.log` with
nine columns including `temperature.gpu`, `power.draw`, and
`clocks_event_reasons.hw_power_brake_slowdown` /
`hw_thermal_slowdown`.

The **`hw_power_brake_slowdown` flag is the single best PSU-trouble
signal**: it fires when the external power source asserts the brake
line back to the GPU — i.e. the PSU told the GPU "back off" because it
couldn't deliver requested power. Healthy operation = "Not Active"
throughout the entire run. Even one "Active" event in a long run means
your PSU got caught struggling.

### Live watcher

Run this alongside production to surface brake / thermal / over-temp
events in real time:

```bash
./westpa_scripts/monitor_psu.sh                 # tail forever, alert to stdout + psu_alerts.log
./westpa_scripts/monitor_psu.sh --halt-on-brake # also SIGINT run_local.sh on first brake
```

The watcher has zero polling overhead — it just tails the existing
`gpu_util.log`. `WATCH_TEMP_C=85 ./monitor_psu.sh` to raise the
"too hot" threshold (default 83 °C, well below the GPU's hardware trip).

### Motherboard telemetry (`board_health.log`)

If `lm_sensors` is installed and the right Super I/O / hwmon drivers
are loaded, `run_local.sh` also writes a `board_health.log` next to
`gpu_util.log` with timestamped `sensors -A` dumps every 10 s. Captures
fan RPMs, motherboard zone temps, CPU temp, NVMe temp, and (depending
on the driver) some voltage rails.

On Gigabyte boards the absolute voltage scaling reported by the open
`it87` driver is often wrong (vendor uses undeclared dividers), so
treat in0..in6 as **uncalibrated trend signals**: a 10% droop under
load is still meaningful even if the absolute value isn't. Fan RPMs
and temps are accurate.

For the TRX50 AERO D specifically, load `it87` with `force_id=0x8695`.
See [BLACKWELL_NOTES.md](BLACKWELL_NOTES.md#motherboard-sensors-on-gigabyte-trx50)
for the persistence recipe.

### Post-mortem use

After a crash, even if the journal stops cold:
```bash
tail -5 gpu_util.log                            # what were temp / power right before?
grep Active gpu_util.log                        # any brake/thermal events at all?
grep -A4 'fan\|temp' board_health.log | tail -30  # board temps / fan state near crash
```
A clean run has no "Active" entries anywhere in gpu_util.log.

## Hardware notes

This fork is developed on a TRX50 + RTX 5090 + RTX 5070 workstation
(consumer Blackwell, sm_120). Three hardware-specific issues came up
during development and are documented in
[BLACKWELL_NOTES.md](BLACKWELL_NOTES.md):

1. **ReBAR must be enabled** on both GPUs (BIOS-level — most consumer
   boards default it off).
2. **Integrator scalar readback after `loadCheckpoint` is corrupted**
   on Blackwell — `runtime/gamd/runners.py` works around this by always
   deriving `currentStep` from `state.getTime()`.
3. **MPS routing**: under the shared sudo MPS daemon, multi-GPU work
   distribution is ambiguous. `run_local.sh`'s `USE_MPS=1` mode uses
   per-GPU user-mode daemons instead and is the recommended default
   for multi-GPU runs.

If you're on data-center hardware (A100, H100, GH200) these may not
apply, but the workarounds are non-invasive and harmless.

---

## Citation

```bibtex
@article{sonti2025pargamd,
  title={Accelerating free energy exploration using parallelizable
         Gaussian accelerated molecular dynamics (ParGaMD)},
  author={Sonti, Siddharth and Thyagatur, Anugraha and Wan, Hung-Yu
          and Hamelynck, Maxen and Faller, Roland and Ahn, Surl-Hee},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-rr5v9}
}
```

---

## License

MIT.

## Contact

Open an issue on the repository.
