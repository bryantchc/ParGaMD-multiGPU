#!/usr/bin/env python3
"""
MC-10 (Maclaurin series order 10) reweighting of the WE-weighted PMF on
(RMSD, Rg) for ParGaMD chignolin.

For each 2D bin j, computes the canonical reweight factor
    <e^(βΔV)>_j  ≈  e^(β <ΔV>_j) * Σ_{k=0..10} β^k m_k(j) / k!
where m_k(j) = <(ΔV - <ΔV>_j)^k>_j (weighted by WE walker weight).
Splitting off the bin-mean keeps the truncated polynomial series
numerically stable (the residual fluctuations have mean 0 and small std
even when β<ΔV> is itself ~100).

Inputs:
    west.h5 — WESTPA HDF5 file with pcoords + walker weights
    traj_segs/<iter>/<seg>/gamd.log — per-segment GaMD boost log
        columns 7 (Total-Boost-Energy-Potential) + 8 (Dihedral-Boost)
        in kcal/mol; one line per output frame.

Outputs:
    pmf_we_only.png — biased (WE-only) PMF (re-rendered for comparison)
    pmf_mc10.png    — MC-10 reweighted canonical PMF
    pmf_compare.png — both PMFs side-by-side
"""

import os, sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

T = 300.0                      # K (matches input.xml)
kB = 0.001987204               # kcal/mol/K
kT = kB * T
beta = 1.0 / kT
K_ORDER = 10                   # Maclaurin truncation order

ROOT = Path(__file__).resolve().parent

def read_dV_for_segment(it, sid, n_frames):
    """Return ΔV_total + ΔV_dihedral per frame from gamd.log; pad/truncate to n_frames."""
    p = ROOT / 'traj_segs' / f'{it:06d}' / f'{sid:06d}' / 'gamd.log'
    try:
        arr = np.loadtxt(p, comments='#', usecols=(6, 7))
        if arr.ndim == 1:
            arr = arr[None, :]
        dV = arr[:, 0] + arr[:, 1]
    except Exception:
        return np.zeros(n_frames)
    if len(dV) >= n_frames:
        return dV[:n_frames]
    pad = dV[-1] if len(dV) else 0.0
    return np.concatenate([dV, np.full(n_frames - len(dV), pad)])


def main():
    print(f'kT = {kT:.4f} kcal/mol  (T = {T} K)')
    print(f'β  = {beta:.4f} (kcal/mol)^-1')
    print(f'MC truncation order K = {K_ORDER}')

    # ------------------------------------------------------------------
    # Pass 1: read pcoords, weights, and ΔV per frame for every segment
    # ------------------------------------------------------------------
    rmsd_list, rg_list, w_list, dV_list = [], [], [], []
    with h5py.File(ROOT / 'west.h5', 'r') as f:
        s = f['summary'][:]
        completed = [i + 1 for i, r in enumerate(s) if r['walltime'] > 0]
        print(f'iterations completed: {len(completed)} (1..{completed[-1]})')

        for it in completed:
            pc = f[f'iterations/iter_{it:08d}/pcoord'][:]
            wt = f[f'iterations/iter_{it:08d}/seg_index'][:]['weight']
            nseg, nframe, _ = pc.shape
            for sid in range(nseg):
                dV_seg = read_dV_for_segment(it, sid, nframe)
                rmsd_list.append(pc[sid, :, 0])
                rg_list.append(pc[sid, :, 1])
                w_list.append(np.full(nframe, wt[sid] / nframe))
                dV_list.append(dV_seg)
            if it % 5 == 0 or it == completed[-1]:
                print(f'  iter {it:3d}: {nseg:5d} segs read')

    rmsd = np.concatenate(rmsd_list)
    rg   = np.concatenate(rg_list)
    w    = np.concatenate(w_list)
    dV   = np.concatenate(dV_list)
    w   /= w.sum()
    print(f'\ntotal frames: {len(rmsd):,}')
    print(f'ΔV  (kcal/mol): mean={dV.mean():.2f} std={dV.std():.2f} '
          f'min={dV.min():.2f} max={dV.max():.2f}')
    print(f'βΔV          : mean={beta*dV.mean():.2f} std={beta*dV.std():.2f} '
          f'min={beta*dV.min():.2f} max={beta*dV.max():.2f}')

    # ------------------------------------------------------------------
    # Bin assignments
    # ------------------------------------------------------------------
    x_lo, x_hi, x_step = 0.0, 9.0, 0.1
    y_lo, y_hi, y_step = 3.0, 10.0, 0.1
    xedges = np.arange(x_lo, x_hi + x_step / 2, x_step)
    yedges = np.arange(y_lo, y_hi + y_step / 2, y_step)
    nx, ny = len(xedges) - 1, len(yedges) - 1
    print(f'grid: {nx} × {ny} bins  (RMSD {x_step}Å, Rg {y_step}Å)')

    ix = np.digitize(rmsd, xedges) - 1
    iy = np.digitize(rg,   yedges) - 1
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = ix * ny + iy

    def bincount2d(weights):
        b = np.bincount(flat[mask], weights=weights[mask], minlength=nx * ny)
        return b.reshape(nx, ny)

    # Biased probability (WE-only) and per-bin <ΔV>
    W_in   = bincount2d(w)
    WV_in  = bincount2d(w * dV)
    nonempty = W_in > 0
    V_mean = np.zeros_like(W_in)
    V_mean[nonempty] = WV_in[nonempty] / W_in[nonempty]

    # ------------------------------------------------------------------
    # Bin-relative moments m_k(j) = <(ΔV - <ΔV>_j)^k>_j  for k = 0..K
    # ------------------------------------------------------------------
    V_mean_per_frame = np.zeros_like(dV)
    V_mean_per_frame[mask] = V_mean[ix[mask], iy[mask]]
    d = dV - V_mean_per_frame

    moments = [np.ones((nx, ny))]              # k = 0  →  1
    moments.append(np.zeros((nx, ny)))         # k = 1  →  0 by construction
    power = d * d                              # start with d^2
    for k in range(2, K_ORDER + 1):
        m_k = bincount2d(w * power)
        m_k_norm = np.zeros_like(m_k)
        m_k_norm[nonempty] = m_k[nonempty] / W_in[nonempty]
        moments.append(m_k_norm)
        power = power * d                      # advance to d^(k+1) for next iter

    # MC-K series: Σ β^k m_k / k!
    fact = [1.0]
    for k in range(1, K_ORDER + 1):
        fact.append(fact[-1] * k)

    reweight_fluct = np.zeros((nx, ny))
    for k in range(K_ORDER + 1):
        reweight_fluct += (beta ** k) * moments[k] / fact[k]
    # Per Miao et al., enforce non-negative reweight factor (truncation can give
    # spuriously negative values in the long-tail bins where moments are noisy).
    reweight_fluct = np.clip(reweight_fluct, 1e-30, None)

    # log <e^(βΔV)>_j  =  β <ΔV>_j  +  log(MC-K of fluctuations)
    log_reweight = beta * V_mean + np.log(reweight_fluct)
    log_reweight[~nonempty] = -np.inf

    log_P_biased = np.log(W_in + 1e-300)
    log_P_biased[~nonempty] = -np.inf

    # log P_canon = log P_biased + log <e^(βΔV)>_j  (then renormalize)
    log_P_canon = log_P_biased + log_reweight
    log_P_canon[~nonempty] = -np.inf
    log_P_canon -= log_P_canon[nonempty].max()
    P_canon = np.exp(log_P_canon)
    P_canon[~nonempty] = 0.0
    P_canon /= P_canon.sum()

    # ------------------------------------------------------------------
    # PMFs
    # ------------------------------------------------------------------
    def to_pmf(P):
        m = P > 0
        F = np.full_like(P, np.nan, dtype=float)
        F[m] = -kT * np.log(P[m])
        F[m] -= F[m].min()
        return F

    P_biased_norm = W_in / W_in.sum()
    pmf_biased = to_pmf(P_biased_norm)
    pmf_mc10   = to_pmf(P_canon)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    levels = np.linspace(0, 5, 21)

    def render(ax, F, title):
        cf = ax.contourf(xc, yc, F.T, levels=levels, cmap='turbo', extend='max')
        ax.contour(xc, yc, F.T, levels=np.arange(0, 5, 1),
                   colors='k', linewidths=0.4, alpha=0.5)
        ax.set_xlabel('RMSD (Å)')
        ax.set_ylabel('Radius of Gyration (Å)')
        ax.set_xlim(0, 9); ax.set_ylim(3, 10)
        ax.set_title(title)
        return cf

    # Standalone MC-10 PMF
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = render(ax, pmf_mc10,
                'ParGaMD chignolin — 23 iter, 1.50 μs\nMC-10 reweighted PMF')
    plt.colorbar(cf, ax=ax, label='Free Energy (kcal/mol)')
    plt.tight_layout()
    plt.savefig(ROOT / 'pmf_mc10.png', dpi=150)
    plt.close(fig)

    # Side-by-side comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cf0 = render(axs[0], pmf_biased, 'WE-only (biased)')
    cf1 = render(axs[1], pmf_mc10,   f'MC-{K_ORDER} reweighted (canonical)')
    fig.colorbar(cf1, ax=axs, label='Free Energy (kcal/mol)',
                 orientation='vertical', fraction=0.04, pad=0.04)
    fig.suptitle('ParGaMD chignolin — 23 iter, 1.50 μs aggregate', y=1.02)
    plt.savefig(ROOT / 'pmf_compare.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # Basin populations
    # ------------------------------------------------------------------
    def basin_weight(P, x_range, y_range):
        i0 = max(0, int(np.floor((x_range[0] - x_lo) / x_step)))
        i1 = min(nx, int(np.ceil((x_range[1] - x_lo) / x_step)))
        j0 = max(0, int(np.floor((y_range[0] - y_lo) / y_step)))
        j1 = min(ny, int(np.ceil((y_range[1] - y_lo) / y_step)))
        return P[i0:i1, j0:j1].sum()

    print('\n--- basin populations (paper definitions) ---')
    print(f'{"basin":<14}{"WE-only":>12}{"MC-10":>12}')
    for name, xr, yr in [('folded   [0-1, 4-5]', (0, 1), (4, 5)),
                         ('inter    [4-6, 5-6]', (4, 6), (5, 6)),
                         ('unfolded [6-8, 7-8]', (6, 8), (7, 8))]:
        wb = basin_weight(P_biased_norm, xr, yr)
        wc = basin_weight(P_canon,        xr, yr)
        print(f'{name:<22}{wb:>12.5f}{wc:>12.5f}')

    print('\nsaved: pmf_mc10.png, pmf_compare.png')


if __name__ == '__main__':
    main()
