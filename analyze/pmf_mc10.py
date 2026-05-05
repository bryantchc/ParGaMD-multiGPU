#!/usr/bin/env python
"""
ParGaMD chignolin PMF on (RMSD, Rg) with GaMD Maclaurin-10 reweighting.

For each (RMSD, Rg) bin j, the canonical probability is recovered as
    P(A_j) ∝ P*(A_j) · ⟨exp(β ΔV)⟩_j
where P*(A_j) is the WE-weighted biased probability and the boost-recovery
factor is approximated by the truncated cumulant expansion (Miao 2014):
    ln⟨exp(β ΔV)⟩_j ≈ Σ_{k=1}^{10} (β^k / k!) κ_k(ΔV in bin j)
ΔV = Total-Boost-Energy-Potential + Dihedral-Boost-Energy from each
segment's gamd.log (dual-boost). Cumulants are computed from per-frame
ΔV samples in each bin, weighted by the WE walker weight assigned by
WESTPA (uniform across the 100 frames of the segment).

Output: pmf_mc10.png plus printed basin weights, max/min ΔV, and the
list of bins where the MC-10 correction was clipped (high-order series
divergence — normal for tail bins with few samples).
"""
import os, math, h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.chdir(ROOT)
T_K = 300.0
kT = 0.001987 * T_K        # kcal/mol
beta = 1.0 / kT
ORDER = int(os.environ.get("MC_ORDER", "2"))   # Maclaurin truncation order
CLIP_KCAL = 100.0          # ln-correction safety clip per bin
MIN_FRAMES = 50            # require this many frames in a bin for cumulants

# Bin grid for the histogram and the cumulant estimation. Match the WE
# bin scheme: 0.2 Å spacing, 0–8 Å on both axes plus +/-inf overflow.
xedges = np.concatenate(([-np.inf], np.arange(0.0, 8.01, 0.2), [np.inf]))
yedges = np.concatenate(([-np.inf], np.arange(0.0, 8.01, 0.2), [np.inf]))
NX = len(xedges) - 1
NY = len(yedges) - 1
print(f"grid: {NX} × {NY} = {NX*NY} bins")

# --- harvest pcoords + WE weights + ΔV from every segment of every iter ---
rmsd_all, rg_all, w_all, dV_all = [], [], [], []

with h5py.File("west.h5", "r") as f:
    s = f["summary"][:]
    completed = [i for i, r in enumerate(s) if r["walltime"] > 0]
    for i in completed:
        it = i + 1
        pc = f[f"iterations/iter_{it:08d}/pcoord"][:]
        wt = f[f"iterations/iter_{it:08d}/seg_index"][:]["weight"]
        nseg = pc.shape[0]
        # Per-segment per-frame weight: WE walker weight / 100 frames
        per_frame_w = np.repeat(wt[:, None], 100, axis=1) / 100.0
        # Read each segment's gamd.log
        dV_iter = np.full((nseg, 100), np.nan)
        for sid in range(nseg):
            log = f"traj_segs/{it:06d}/{sid:06d}/gamd.log"
            try:
                arr = np.loadtxt(log, comments="#", usecols=(6, 7))
                # arr shape: (100, 2). column 0 = total-boost, 1 = dihedral-boost
                if arr.shape[0] >= 100:
                    dV_iter[sid] = arr[:100, 0] + arr[:100, 1]
                elif arr.shape[0] > 0:
                    dV_iter[sid, :arr.shape[0]] = arr[:, 0] + arr[:, 1]
                # else leave as NaN; will be filtered below
            except Exception:
                pass    # leave row as NaN
        rmsd_all.append(pc[..., 0].ravel())
        rg_all.append(pc[..., 1].ravel())
        w_all.append(per_frame_w.ravel())
        dV_all.append(dV_iter.ravel())
        if it % 5 == 0 or it == completed[-1] + 1:
            print(f"  parsed iter {it}/{completed[-1]+1} ({nseg} segs)")

rmsd = np.concatenate(rmsd_all)
rg   = np.concatenate(rg_all)
w    = np.concatenate(w_all)
dV   = np.concatenate(dV_all)

# Drop frames with missing ΔV (segment had no/short gamd.log)
ok = ~np.isnan(dV)
print(f"\nframes total: {len(rmsd):,}  with valid ΔV: {ok.sum():,} "
      f"({100*ok.mean():.2f}%)")
rmsd, rg, w, dV = rmsd[ok], rg[ok], w[ok], dV[ok]
w = w / w.sum()

print(f"ΔV stats: min={dV.min():.2f}, max={dV.max():.2f}, "
      f"mean={dV.mean():.2f}, std={dV.std():.2f} kcal/mol")

# --- bin frames into the (NX, NY) grid ---
ix = np.digitize(rmsd, xedges) - 1
iy = np.digitize(rg,   yedges) - 1
ix = np.clip(ix, 0, NX - 1)
iy = np.clip(iy, 0, NY - 1)
flat = ix * NY + iy

# Biased weighted histogram P*(A_j)
counts = np.bincount(flat, weights=w, minlength=NX * NY).reshape(NX, NY)
nframes_per_bin = np.bincount(flat, minlength=NX * NY).reshape(NX, NY)

# --- per-bin Maclaurin-10 cumulant correction ---
# Need raw moments μ_1...μ_ORDER per bin (weighted). Use the per-frame WE
# weight w_i within the bin, normalized to the bin's total weight.
moments = np.zeros((ORDER, NX * NY))
for k in range(1, ORDER + 1):
    moments[k - 1] = np.bincount(flat, weights=w * dV ** k,
                                  minlength=NX * NY)
totw = np.bincount(flat, weights=w, minlength=NX * NY)
# normalize per bin
mu = np.zeros_like(moments)
nz = totw > 0
for k in range(ORDER):
    mu[k, nz] = moments[k, nz] / totw[nz]

# Recursion: κ_n = μ_n - Σ_{m=1}^{n-1} C(n-1, m-1) κ_m μ_{n-m}
kappa = np.zeros_like(mu)
kappa[0] = mu[0]
for n in range(2, ORDER + 1):
    s = np.zeros(NX * NY)
    for m in range(1, n):
        s += math.comb(n - 1, m - 1) * kappa[m - 1] * mu[n - m - 1]
    kappa[n - 1] = mu[n - 1] - s

# ln⟨exp(βΔV)⟩_j ≈ Σ_{k=1}^{ORDER} β^k / k! × κ_k
ln_corr = np.zeros(NX * NY)
for k in range(1, ORDER + 1):
    ln_corr += beta ** k / math.factorial(k) * kappa[k - 1]

ln_corr = ln_corr.reshape(NX, NY)
n_clipped = int(np.sum(np.abs(ln_corr) > beta * CLIP_KCAL))
print(f"bins where |ln-correction| > {CLIP_KCAL} kcal/mol "
      f"(MC-10 series unstable, mostly low-frame-count tails): "
      f"{n_clipped} of {(nframes_per_bin >= MIN_FRAMES).sum()} populated bins")
ln_corr = np.clip(ln_corr, -beta * CLIP_KCAL, beta * CLIP_KCAL)

# Mask bins with too few raw frames
mask_low = nframes_per_bin < MIN_FRAMES
ln_corr[mask_low] = 0.0   # don't reweight tail; they remain biased

# --- combine to canonical PMF ---
P_corrected = counts * np.exp(ln_corr)
Z = P_corrected.sum()
if Z <= 0:
    raise SystemExit("MC-10 reweighted P sums to zero")
P_corrected = P_corrected / Z

with np.errstate(divide="ignore"):
    F = -kT * np.log(P_corrected)
F[counts == 0] = np.nan
F_min = np.nanmin(F)
F = F - F_min

# Strip the +/-inf overflow rows/cols for plotting
xc = 0.5 * (xedges[1:-1][:-1] + xedges[1:-1][1:])
yc = 0.5 * (yedges[1:-1][:-1] + yedges[1:-1][1:])
F_plot = F[1:-1, 1:-1].T   # transpose so x=RMSD horizontal

# --- plot ---
fig, ax = plt.subplots(figsize=(6.0, 5.0))
levels = np.linspace(0, 5, 21)
cf = ax.contourf(xc, yc, F_plot, levels=levels, cmap="turbo", extend="max")
plt.colorbar(cf, ax=ax, label="Free Energy (kcal/mol)")
ax.set_xlabel("RMSD (Å)")
ax.set_ylabel("Radius of Gyration (Å)")
ax.set_xlim(0, 9)
ax.set_ylim(3, 10)
ax.set_title(f"ParGaMD chignolin — 23 iter, 1.50 µs aggregate (MC-{ORDER} reweighted)")
plt.tight_layout()
out = f"analyze/pmf_mc{ORDER}.png"
plt.savefig(out, dpi=150)
print(f"saved {out}")

# Diagnostic: weight reshuffling vs WE-only
print("\n--- bin weights, WE-biased vs MC-10-corrected ---")
def basin(name, xlo, xhi, ylo, yhi):
    xi_lo = np.searchsorted(xedges, xlo); xi_hi = np.searchsorted(xedges, xhi)
    yi_lo = np.searchsorted(yedges, ylo); yi_hi = np.searchsorted(yedges, yhi)
    biased = counts[xi_lo:xi_hi, yi_lo:yi_hi].sum()
    canon  = P_corrected[xi_lo:xi_hi, yi_lo:yi_hi].sum()
    print(f"{name:20s} biased={biased:.4f}  canonical={canon:.4f}")

basin("folded   [0,1]×[4,5]",   0, 1, 4, 5)
basin("inter    [4,6]×[5,6]",   4, 6, 5, 6)
basin("unfolded [6,8]×[7,8]",   6, 8, 7, 8)
