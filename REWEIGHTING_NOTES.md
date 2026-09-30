# Reweighting notes: GaMD boost × WE weights

Practical guidance for turning ParGaMD output into free energies, written
after reweighting on the Cas12a runs kept misbehaving. The short version:

> **σ₀ is rarely the problem. The WE weights are.** Reweight with MC-10 or
> CE1, report the effective sample size per bin, and treat CE2 as a
> cross-check, not the answer.

Sources: Wang, Arantes, Bhattarai, Miao et al., *GaMD: Principles and
applications* (WIREs review; eqs. 7–9, 16); Sonti et al., *Accelerating free
energy exploration using ParGaMD* (JCTC 2026 / ChemRxiv), §2.2–2.4 and §3.

---

## 1. What σ₀ actually controls

GaMD sets the boost force constant as

    k0 = min(1, k0'),   k0' = (σ₀ / σ_V) · (Vmax − Vmin) / (Vmax − Vavg)

so that the standard deviation of the boost obeys σ_ΔV ≤ σ₀ (review eq. 7).

- **Units: kcal/mol, per boost term.** For a dual boost (`sigma0` primary /
  secondary, or Amber `sigma0P` / `sigma0D`) each term is capped separately.
  If the two terms are roughly uncorrelated, the *total* σ_ΔV adds in
  quadrature, so σ₀ = 3/3 gives a total of about 4 kcal/mol.
- **Published ceiling: "e.g., 10 kBT"** (≈ 6 kcal/mol at 300 K). The default
  is 6.0 in NAMD, Amber and upstream ParGaMD; 3–6 is the usual range. σ₀ is
  the dial for how hard to boost.
- **What decides whether CE2 is valid is the *shape* of ΔV, not its width.**
  For an exactly Gaussian ΔV, the 2nd-order cumulant is exact at any σ. The
  review's diagnostic is the anharmonicity γ (eq. 16); γ ≈ 0 means
  Gaussian. `pyreweight.py -job amd_dV` reports it per bin.
- σ₀ is frozen at equilibration. `gamd-restart.dat` carries k0, and
  production (stage 5) does not recompute it, so changing σ₀ means re-running
  the GaMD-equilibration stages.
- A k0 stuck at 1.0 means σ₀ isn't binding: the boost is at its maximum and
  σ_ΔV is whatever the system gives. That is worth checking before production.

Reference point (Ultra_RL2.tmd_noMg, 266k atoms, σ₀ = 3/3, dual
nonbonded+dihedral):

| term      | σ_ΔV (kcal/mol) | σ_ΔV / kT | skew | γ     |
|-----------|-----------------|-----------|------|-------|
| nonbonded | 2.5             | 4.75      | 0.71 | 0.067 |
| dihedral  | 3.3             | 5.23      | 0.58 | 0.047 |
| total     | 4.1             | 6.99      | 0.35 | 0.024 |

That is within the published range and near-Gaussian: a usable
equilibration.

## 2. Why reweighting still blows up: WE weights shrink the sample

In ParGaMD every frame carries two factors: the GaMD Boltzmann factor
e^{βΔV} and the WE walker weight w. WE weights are **not** uniform. Splitting
and merging, compounded over many iterations, spreads them across many
orders of magnitude.

Measured on `Ultra_RL2.rna` (124 iterations, 82,912 segments):

- ln w spans **92 units** (sd 23).
- On a 20×20 grid over the first two pcoord dims, the median bin holds 164
  segments but an **effective sample size**

      N_eff = (Σ w)² / Σ w²

  of only **6** (4 % of the raw count). 65 % of bins have N_eff < 10.

A weighted per-bin moment of ΔV is only as good as N_eff, not the raw count.
The CE2 correction is ½β²σ²_ΔV, and the standard error of a variance estimate
is about σ²·√(2/(N_eff − 1)). So at σ_ΔV ≈ 7 kT:

    C2 ≈ ½ · 7² ≈ 24 kT,   error ≈ 24 · √(2/5) ≈ ±15 kT per bin.

That is enough noise to turn a PMF into confetti, and it is **independent of
how carefully σ₀ was chosen**. Halving σ₀ cuts C2 (and its noise) about 4×.
Raising N_eff, or not using the σ² term at all, does more.

### Two ways to fold in WE weights (they are not the same)

- **Probability weight on the moments** (what `pyreweight.py -we` and
  `reweigh/reweigh_CE.py` do):
  ⟨ΔV^k⟩_j = Σ w_i ΔV_i^k / Σ w_i over frames in bin j, and the histogram is
  Σ w_i. This is the sensible choice.
- **Effective boost** (ParGaMD paper eq. 16): ΔV_eff = ln(w)/β + ΔV, with
  cumulants taken over ΔV_eff. The ln w spread (sd ~23 kT above) then enters
  the variance directly and is far from Gaussian, so **do not use CE2 this
  way on long runs.**

Formally both give the same exponential average; they differ only in what the
Gaussian assumption is applied to.

### What the ParGaMD authors actually used

Their paper describes CE2 but reports no CE2 results: **chignolin was
reweighted with MC-10, α-synuclein–PAL with CE1.** Follow their practice.

## 3. Recommendations

1. **Estimator: MC-10 first, CE1 second, CE2 only as a cross-check.**
   - MC-10 (`-job amdweight_MC -order 10`): stable for broad ΔV, but can bias
     barriers by 2–3 kT (review).
   - CE1 (`pmf-c1` from `-job amdweight_CE`): drops the noisy σ² term. It is
     biased low by ½β²σ² per bin, but the bias is smooth rather than noisy
     when σ_ΔV is similar across bins, so shapes and minima survive.
   - CE2 (`pmf-c12`): trust it only in bins with large N_eff *and* small γ.
     If CE2 and MC-10 disagree by more than ~1 kcal/mol in a bin, believe
     neither there.
2. **Report and mask by N_eff, not raw counts.** Mask bins with N_eff below
   ~10–20. The upstream `reweigh_CE.py` `--cutoff` is a *weight-sum*
   threshold, which is not the same thing.
3. **Drop the pre-steady-state iterations.** WE weights are most uneven
   early, while probability is still flowing into newly opened bins. Check
   that the PMF is stable when the first N iterations are excluded.
   `pyreweight.py` currently has only `--max-iter` (an upper cut), so
   excluding early iterations needs a `--min-iter` flag, which has not been
   added yet. Comparing `--max-iter` at several values at least shows
   whether the PMF has stopped moving.
4. **Coarsen before you refine.** Reweight in 1D, or on a coarse 2D grid,
   first. Every halving of the bin width on a 2D grid roughly quarters N_eff
   per bin.
5. **Bootstrap by segment lineage, not by frame.** Frames within a segment
   share one weight and are time-correlated; resampling frames overstates
   precision.
6. **Pick σ₀ on γ and N_eff, not on a σ_ΔV/kT rule.** 3 kcal/mol per term
   is a good starting point for large (~250k-atom) systems. Go lower (~2)
   only if MC-10 and CE2 still disagree in well-sampled (high-N_eff) bins.
7. **Check k0 after equilibration.** If k0 = 1.0 and ⟨ΔV⟩ was still climbing
   when equilibration ended, it isn't converged. Extend or re-equilibrate
   before trusting any reweighting.

## 4. Tools

| Tool | Where | Notes |
|---|---|---|
| `pyreweight.py` | per run dir (not tracked in this repo) | `-we` reads `west.h5` + per-segment `gamd.log`; jobs `amdweight_MC`, `amdweight_CE` (writes c1 / c12 / c123), `amd_dV` (per-bin σ, γ) |
| `pgd fes` | [`pgd/cmds/fes.py`](pgd/cmds/fes.py) | wrapper around the run's `pyreweight.py`; `--job amdweight_MC` / `amdweight_CE` |
| `reweigh/reweigh.py` | this repo | upstream MC reweighter (flat-file input) |
| `reweigh/reweigh_CE.py` | this repo | upstream CE1/2/3; WE weight as probability weight; cutoff is on Σw |
| `reweight_compare.py` | this repo | WE-only vs C2 vs MC-10 side-by-side |

Quick N_eff check on any run (read-only, safe while `w_run` is live):

```python
import h5py, numpy as np
f = h5py.File("west.h5", "r")   # HDF5_USE_FILE_LOCKING=FALSE if w_run is running
w = np.concatenate([f["iterations"][k]["seg_index"][:]["weight"] for k in f["iterations"]])
p = np.concatenate([f["iterations"][k]["pcoord"][:, -1, :2] for k in f["iterations"]])
lo, hi = np.percentile(p, 1, 0), np.percentile(p, 99, 0)
b = np.clip(((p - lo) / (hi - lo) * 20).astype(int), 0, 19); key = b[:, 0] * 20 + b[:, 1]
neff = [(w[key == k].sum() ** 2) / (w[key == k] ** 2).sum() for k in np.unique(key)]
print("median N_eff per bin:", np.median(neff))
```
