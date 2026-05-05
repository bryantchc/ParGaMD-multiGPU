#!/usr/bin/env python3
"""
Compare three GaMD reweighting strategies for the ParGaMD chignolin run:

    A. WE-only (no GaMD reweight) — baseline, what we already plotted.
    B. Cumulant-2 (C2) — Miao 2014 original; uses only <ΔV>_j and σ²_V,j.
       PMF correction:  -kT log<e^(βΔV)>_j  ≈  -<ΔV>_j - β σ²_V,j / 2
       Stable but approximate; the standard "default" in PyReweighting.
    C. MC-10 on ΔV_dihedral only — dihedral boost is much smaller
       (typically 1-10 kcal/mol → βΔV ~2-15) so the polynomial
       truncation actually converges. Gives a *partial* reweight that
       captures conformational-sampling enhancement but not the
       solvent/total-energy boost.

Total-boost MC-10 is omitted because βΔV_total ~90 makes the truncation
divergent (already verified — collapses to a single bin).

Outputs:
    pmf_we_only.png       — biased baseline
    pmf_c2.png            — Cumulant-2 reweighted
    pmf_mc10_dihed.png    — MC-10 on dihedral boost only
    pmf_compare_all.png   — three-panel comparison
"""

from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

T = 300.0
kB = 0.001987204
kT = kB * T
beta = 1.0 / kT
K_ORDER = 10

ROOT = Path(__file__).resolve().parent


def read_dV(it, sid, n_frames):
    """Return (ΔV_total, ΔV_dihedral) per frame from gamd.log."""
    p = ROOT / 'traj_segs' / f'{it:06d}' / f'{sid:06d}' / 'gamd.log'
    try:
        arr = np.loadtxt(p, comments='#', usecols=(6, 7))
        if arr.ndim == 1:
            arr = arr[None, :]
        dVt = arr[:, 0]
        dVd = arr[:, 1]
    except Exception:
        return np.zeros(n_frames), np.zeros(n_frames)
    def fix(v):
        if len(v) >= n_frames:
            return v[:n_frames]
        pad = v[-1] if len(v) else 0.0
        return np.concatenate([v, np.full(n_frames - len(v), pad)])
    return fix(dVt), fix(dVd)


def gather():
    rmsd, rg, w, dVt, dVd = [], [], [], [], []
    with h5py.File(ROOT / 'west.h5', 'r') as f:
        s = f['summary'][:]
        completed = [i + 1 for i, r in enumerate(s) if r['walltime'] > 0]
        for it in completed:
            pc = f[f'iterations/iter_{it:08d}/pcoord'][:]
            wt = f[f'iterations/iter_{it:08d}/seg_index'][:]['weight']
            nseg, nframe, _ = pc.shape
            for sid in range(nseg):
                t, d = read_dV(it, sid, nframe)
                rmsd.append(pc[sid, :, 0]); rg.append(pc[sid, :, 1])
                w.append(np.full(nframe, wt[sid] / nframe))
                dVt.append(t); dVd.append(d)
            if it % 5 == 0 or it == completed[-1]:
                print(f'  iter {it:3d}: {nseg:5d} segs')
    out = {k: np.concatenate(v) for k, v in
           dict(rmsd=rmsd, rg=rg, w=w, dVt=dVt, dVd=dVd).items()}
    out['w'] /= out['w'].sum()
    return out, len(completed)


def grid_setup(rmsd, rg):
    x_lo, x_hi, x_step = 0.0, 9.0, 0.1
    y_lo, y_hi, y_step = 3.0, 10.0, 0.1
    xedges = np.arange(x_lo, x_hi + x_step / 2, x_step)
    yedges = np.arange(y_lo, y_hi + y_step / 2, y_step)
    nx, ny = len(xedges) - 1, len(yedges) - 1
    ix = np.digitize(rmsd, xedges) - 1
    iy = np.digitize(rg,   yedges) - 1
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = ix * ny + iy
    return dict(xedges=xedges, yedges=yedges, nx=nx, ny=ny,
                ix=ix, iy=iy, mask=mask, flat=flat,
                x_lo=x_lo, y_lo=y_lo, x_step=x_step, y_step=y_step)


def bincount2d(g, weights):
    b = np.bincount(g['flat'][g['mask']], weights=weights[g['mask']],
                    minlength=g['nx'] * g['ny'])
    return b.reshape(g['nx'], g['ny'])


def reweight_C2(g, w, dV):
    """Cumulant-2 reweighting:  log<e^(βΔV)>_j ≈ β<ΔV>_j + β² σ²_V,j / 2 ."""
    W   = bincount2d(g, w)
    WV  = bincount2d(g, w * dV)
    WV2 = bincount2d(g, w * dV * dV)
    nz = W > 0
    Vm = np.zeros_like(W); Vm[nz] = WV[nz] / W[nz]
    V2 = np.zeros_like(W); V2[nz] = WV2[nz] / W[nz]
    var = np.maximum(V2 - Vm * Vm, 0.0)
    log_rw = np.full_like(W, -np.inf)
    log_rw[nz] = beta * Vm[nz] + 0.5 * (beta ** 2) * var[nz]
    return W, log_rw, nz


def reweight_MCK(g, w, dV, K=10):
    """MC-K on bin-relative fluctuations: log<e^(βΔV)>_j = β<ΔV>_j + log Σ β^k m_k/k! ."""
    W  = bincount2d(g, w)
    WV = bincount2d(g, w * dV)
    nz = W > 0
    Vm = np.zeros_like(W); Vm[nz] = WV[nz] / W[nz]
    Vm_per = np.zeros_like(dV)
    Vm_per[g['mask']] = Vm[g['ix'][g['mask']], g['iy'][g['mask']]]
    d = dV - Vm_per

    moments = [np.ones_like(W), np.zeros_like(W)]   # k=0,1
    power = d * d
    for _ in range(2, K + 1):
        m = bincount2d(g, w * power)
        mn = np.zeros_like(m); mn[nz] = m[nz] / W[nz]
        moments.append(mn)
        power = power * d

    fact = [1.0]
    for k in range(1, K + 1):
        fact.append(fact[-1] * k)
    series = np.zeros_like(W)
    for k in range(K + 1):
        series += (beta ** k) * moments[k] / fact[k]
    series = np.clip(series, 1e-30, None)

    log_rw = np.full_like(W, -np.inf)
    log_rw[nz] = beta * Vm[nz] + np.log(series[nz])
    return W, log_rw, nz


def to_pmf(W, log_rw, nz):
    log_P = np.full_like(W, -np.inf)
    log_P[nz] = np.log(W[nz]) + log_rw[nz]
    log_P[nz] -= log_P[nz].max()
    P = np.zeros_like(W); P[nz] = np.exp(log_P[nz])
    P /= P.sum() if P.sum() > 0 else 1
    F = np.full_like(W, np.nan)
    pos = P > 0
    F[pos] = -kT * np.log(P[pos])
    F[pos] -= F[pos].min()
    return P, F


def render(ax, F, xc, yc, title, levels):
    cf = ax.contourf(xc, yc, F.T, levels=levels, cmap='turbo', extend='max')
    ax.contour(xc, yc, F.T, levels=np.arange(0, levels.max(), 1),
               colors='k', linewidths=0.4, alpha=0.5)
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Radius of Gyration (Å)')
    ax.set_xlim(0, 9); ax.set_ylim(3, 10)
    ax.set_title(title)
    return cf


def main():
    print(f'kT = {kT:.4f} kcal/mol; β = {beta:.4f}')
    data, niter = gather()
    rmsd, rg, w, dVt, dVd = data['rmsd'], data['rg'], data['w'], data['dVt'], data['dVd']
    print(f'\ntotal frames: {len(rmsd):,}')
    print(f'ΔV_total       : mean={dVt.mean():.2f} std={dVt.std():.2f}  '
          f'βΔV mean/std={beta*dVt.mean():.2f}/{beta*dVt.std():.2f}')
    print(f'ΔV_dihedral    : mean={dVd.mean():.2f} std={dVd.std():.2f}  '
          f'βΔV mean/std={beta*dVd.mean():.2f}/{beta*dVd.std():.2f}')

    g = grid_setup(rmsd, rg)
    xc = 0.5 * (g['xedges'][:-1] + g['xedges'][1:])
    yc = 0.5 * (g['yedges'][:-1] + g['yedges'][1:])
    levels = np.linspace(0, 5, 21)

    # Biased baseline
    W = bincount2d(g, w)
    nz = W > 0
    P_bias = W / W.sum()
    F_bias = np.full_like(W, np.nan); F_bias[nz] = -kT * np.log(P_bias[nz]); F_bias[nz] -= F_bias[nz].min()

    # C2 on full ΔV (total + dihedral)
    dV_full = dVt + dVd
    W2, lrw_c2, nz_c2 = reweight_C2(g, w, dV_full)
    P_c2, F_c2 = to_pmf(W2, lrw_c2, nz_c2)

    # MC-10 on dihedral only
    Wd, lrw_d, nz_d = reweight_MCK(g, w, dVd, K=K_ORDER)
    P_d, F_d = to_pmf(Wd, lrw_d, nz_d)

    # Plots
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    cf0 = render(axs[0], F_bias, xc, yc, 'WE-only (biased)', levels)
    cf1 = render(axs[1], F_c2,   xc, yc, 'Cumulant-2 (full ΔV)', levels)
    cf2 = render(axs[2], F_d,    xc, yc, f'MC-{K_ORDER} (dihedral ΔV only)', levels)
    fig.colorbar(cf2, ax=axs, label='Free Energy (kcal/mol)',
                 fraction=0.025, pad=0.02)
    fig.suptitle(f'ParGaMD chignolin — {niter} iter, '
                 f'{1499200 * 0.0001:.2f} μs aggregate', y=1.02)
    plt.savefig(ROOT / 'pmf_compare_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    for F, name in [(F_bias, 'pmf_we_only.png'),
                    (F_c2,   'pmf_c2.png'),
                    (F_d,    'pmf_mc10_dihed.png')]:
        fig, ax = plt.subplots(figsize=(6, 5))
        cf = render(ax, F, xc, yc, name.replace('.png',''), levels)
        plt.colorbar(cf, ax=ax, label='Free Energy (kcal/mol)')
        plt.tight_layout()
        plt.savefig(ROOT / name, dpi=150)
        plt.close(fig)

    # Basin populations
    def basin(P, xr, yr):
        i0 = max(0, int(np.floor((xr[0]-g['x_lo'])/g['x_step'])))
        i1 = min(g['nx'], int(np.ceil((xr[1]-g['x_lo'])/g['x_step'])))
        j0 = max(0, int(np.floor((yr[0]-g['y_lo'])/g['y_step'])))
        j1 = min(g['ny'], int(np.ceil((yr[1]-g['y_lo'])/g['y_step'])))
        return P[i0:i1, j0:j1].sum()

    print('\n--- basin populations (paper definitions) ---')
    print(f'{"basin":<22}{"biased":>10}{"C2":>10}{"MC10-d":>10}')
    for n, xr, yr in [('folded   [0-1, 4-5]', (0,1), (4,5)),
                      ('inter    [4-6, 5-6]', (4,6), (5,6)),
                      ('unfolded [6-8, 7-8]', (6,8), (7,8))]:
        print(f'{n:<22}{basin(P_bias,xr,yr):>10.4f}'
              f'{basin(P_c2,xr,yr):>10.4f}{basin(P_d,xr,yr):>10.4f}')

    # ΔΔG vs folded
    def DG(P, xr, yr):
        b = basin(P, xr, yr)
        f = basin(P, (0,1), (4,5))
        if b<=0 or f<=0: return float('inf')
        return -kT * np.log(b/f)

    print('\n--- ΔG (basin → folded) in kcal/mol ---')
    for n, xr, yr in [('intermediate', (4,6), (5,6)),
                      ('unfolded',     (6,8), (7,8))]:
        print(f'{n:<14}'
              f'biased={DG(P_bias,xr,yr):>6.2f}  '
              f'C2={DG(P_c2,xr,yr):>6.2f}  '
              f'MC10-d={DG(P_d,xr,yr):>6.2f}')


if __name__ == '__main__':
    main()
