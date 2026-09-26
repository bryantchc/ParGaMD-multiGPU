# Annotated `west.cfg` examples

Worked references for binning schemes and the other knobs in `west.cfg`. None
of these are used by `new_run.sh` — it copies `templates/west.cfg.template`.
Copy one over a scaffolded run's `west.cfg` to use it.

`west.cfg` is read when the sim manager **starts**. Changing it mid-run does
nothing; stop `w_run`, edit, and restart — WESTPA resumes from the last
completed iteration and you lose only the in-flight one.

## The examples

| file | scheme | reach for it when |
|---|---|---|
| `01-rectilinear-1d.cfg` | fixed bins, one CV | default starting point; you can read the occupancies and understand them |
| `02-rectilinear-2d-uniform.cfg` | fixed bins, two CVs | two coordinates are **coupled** and driving either alone parks the system somewhere useless |
| `03-rectilinear-2d-nonuniform-targets.cfg` | 2D + per-bin walker counts | some region deserves more walkers than another — a frontier fighting a restoring force, or an oversampled bulk |
| `04-mab-adaptive.cfg` | adaptive bins | you genuinely don't know the CV's range, **and** the wrong direction is bounded |

## Choosing a scheme

**Rectilinear unless you have a reason.** Fixed edges mean a bin index always
denotes the same region, which is what makes occupancy maps, per-bin target
counts, and post-hoc analysis all straightforward.

**2D when coordinates are coupled.** If pushing coordinate A alone leaves B in
a useless state, bin both — that is the only way WE can hold the diagonal. It
costs you: bins multiply, and wall-clock per iteration scales with walker count
once you're worker-limited. A 15×17 grid at 4 walkers/bin is up to 1020 walkers.

**MAB only when the wrong direction is bounded.** MAB spans ensemble min..max
per dimension. An unbounded wrong direction — a contact distance where the
ligand can dissociate to infinity, an RMSD with nothing capping it — inflates
the span every iteration and drains walkers into the tail. This cost this
project a run. A rectilinear grid with a deliberate overflow bin caps the
damage at one bin instead of rescaling the whole mapper.

## Calibrate bin width against τ

Measure the median per-segment step in your CV and make bins roughly that wide.
Much narrower and walkers cross several bins per τ anyway, so the splits are
wasted; much wider and WE cannot see progress at all. Both axes in `02` were
set this way (docking ~0.24 Å/τ → 0.25 Å bins; clearance 0.31 Å/τ → 0.40 Å).

## Parameter reference

| key | what it does | the part that bites |
|---|---|---|
| `pcoord_ndim` | number of CV dimensions | must equal the number of `boundaries:` lists **and** the `cv_*.py` count the dispatcher finds. Mismatch dies at iteration 1 with an opaque `digitize` index error |
| `pcoord_len` | frames recorded per segment | must be ≥ 2 or most analysis tools break |
| `pcoord_dtype` | storage type | `!!python/name:numpy.float32` is YAML executing code — needs `yaml.unsafe_load` to read the file back |
| `bins.boundaries` | bin edges per dimension | the `'inf'` overflow is **not** decoration: without it an out-of-range walker is unassignable and the run dies. With it, everything past that edge is invisible to WE |
| `bin_target_counts` | walkers per **occupied** bin | scalar, or a list of length `nbins`. Total cost is this × occupied bins, which **grows as the run explores** — it is not a fixed budget |
| `max_total_iterations` | stop after N iterations | |
| `max_run_wallclock` | stop after this long | `HH:MM:SS`; the run stops cleanly and can be resumed |
| `gen_istates` | WESTPA builds initial states itself | `false` when you supply prepared states (e.g. `make_bstates.py`) |
| `datasets.scaleoffset` | lossy float compression | `4` keeps ~4 decimal digits. Fine for Ångström pcoords, **wrong** for small dimensionless CVs — silently quantises them |
| `data_refs.*` | where segments/bstates live on disk | `traj_segs` is usually a symlink into a scratch pool; keep the link, don't replace it |

## Non-uniform `bin_target_counts`

Supported in 2022.15: `_rc.py` accepts an iterable and asserts
`len == mapper.nbins`. Targets are indexed by absolute bin number, and
`we_driver.py` skips empty bins before reading the target — so the list covers
occupied and unoccupied cells alike and **nothing needs recomputing as bins
fill**.

Generate it, never hand-edit it:

```bash
python westpa_scripts/make_target_counts.py \
    --west-cfg west.cfg --policy corner-weighted --emit yaml --assert-order
```

`--assert-order` re-verifies the C-order flattening
(`index = dim0_bin * n_dim1 + dim1_bin`) against WESTPA's own mapper before
emitting. Any boundary change invalidates the whole list.

**Only safe for a static mapper with no target states.** Three hazards:

- **Adaptive mappers** (MAB, adaptive Voronoi) — boundaries move, so bin `i`
  means a different region each iteration and the list mis-targets silently.
- **`RecursiveBinMapper`** — flattening is not the simple C-order product.
- **Target states** — two places in WESTPA assume uniform counts:
  `we_driver.py:296` overwrites your value with 0 for a target-state bin, and
  `sim_manager.py:389` computes
  `total_replicas = sum(...) - bin_target_counts[-1] * len(target_states)`,
  using the *last* bin's count for every target-state bin. Both are inert when
  there are no target states, which is why this is safe today — but adding a
  target state later activates both at once.

Both of those are unchanged on upstream's active branch as of September 2026,
and the related issue (westpa#83) has been open since 2018, so don't expect a
fix.
