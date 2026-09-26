# Alternate template examples

These are **not** used by `new_run.sh`. They are kept as worked references for
systems whose GaMD parameters differ from the defaults in `templates/`.

## chignolin/

The original upstream templates, tuned for chignolin (~2 k atoms):

| | chignolin | current default (Cas12a-scale) |
|---|---|---|
| `dt` | 0.002 ps | 0.004 ps |
| `boost-type` | `upper-dual` | `lower-dual-nonbonded-dihedral` |
| `averaging-window-interval` | 50 000 | 500 000 |
| cMD (equilibration) | 800 k steps | 25 M steps |
| GaMD-eq (equilibration) | 800 k steps | 15 M steps |

The defaults moved to the Cas12a-scale values because every real run in this
repo is a solvated nucleoprotein complex of a few hundred thousand atoms, and
the chignolin numbers are wrong for those in ways that are easy to miss:

* `upper-dual` boosts far more aggressively than `lower-dual-nonbonded-dihedral`,
  which is what a system this size needs.
* a 2 fs timestep halves tau for a fixed `extension-steps`, which silently
  invalidates any bin spacing calibrated against a 4 fs run.
* a 50 k averaging window refreshes the boost statistics ten times more often,
  making them noisy on a large system.

A misconfigured run of this kind is not obviously broken -- it propagates, it
writes plausible pcoords, and you only notice when the sampling is poor or the
boost statistics are inconsistent with the checkpoints you seeded from.

To use these instead, point `new_run.sh` at them or copy them over the run's
`input.xml` / `equilibration/input.xml` after scaffolding.

## west.cfg/

Annotated `west.cfg` variants: 1D and 2D rectilinear grids, per-bin walker
counts, and MAB. Includes a parameter reference for the rest of the file
(`pcoord_len`, `scaleoffset`, `gen_istates`, the overflow bins) and notes on
which schemes compose with which. See [west.cfg/README.md](west.cfg/README.md).
