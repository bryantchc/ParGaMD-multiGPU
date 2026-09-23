"""Search the explored PES for segments at a given point.

A full scan of west.h5's pcoord -- all 72,852 segments x 100 frames x 2 CVs --
takes about 0.3 s, so there is no index to build or invalidate. The optional
cached flat index exists only for the FES picker, which wants the arrays
anyway.

CV semantics for Ultra_RL2.rna: cv0 is the mean heavy-atom distance over the
REC1-clamp subset of unsatisfied native contacts, cv1 the same for the PI
clamp. Both are in Angstroms and both are driven DOWN, so they share units and
meaning -- which is why the default tolerance is a Euclidean disk rather than
a box.
"""

from __future__ import annotations

import numpy as np

from . import paths as P


def scan(h5, max_iter: int, iter_lo: int = 1):
    """Yield (n_iter, pcoord[nseg,nframe,ndim], weight[nseg]) per iteration."""
    for it in range(iter_lo, max_iter + 1):
        g = P.h5_iter_group(it)
        try:
            pc = h5[g + "/pcoord"][:]
            w = h5[g + "/seg_index"]["weight"][:]
        except KeyError:
            continue
        yield it, pc, w


def find_points(h5, max_iter, cv, tol, metric="disk", frames="last",
                iter_lo=1, min_weight=0.0):
    """Segments whose pcoord lands within `tol` of `cv`.

    frames='last' matches only the final frame -- the iteration endpoint, and
    the honest reading of "a populated point on the explored PES".
    frames='any' matches a segment if any of its 100 frames qualifies.

    Returns a list of dicts, unsorted.
    """
    cv = np.asarray(cv, dtype=np.float64)
    tol = np.atleast_1d(np.asarray(tol, dtype=np.float64))
    if tol.size == 1:
        tol = np.repeat(tol, cv.size)

    hits = []
    for it, pc, w in scan(h5, max_iter, iter_lo):
        sel = pc[:, -1:, :] if frames == "last" else pc
        d = sel - cv[None, None, :]
        if metric == "box":
            ok = (np.abs(d) <= tol[None, None, :]).all(axis=2)
            dist = np.abs(d / tol[None, None, :]).max(axis=2)
        else:
            dist = np.sqrt((d ** 2).sum(axis=2))
            ok = dist <= tol[0]
        if not ok.any():
            continue
        segs = np.unique(np.nonzero(ok)[0])
        for s in segs:
            if w[s] < min_weight:
                continue
            fr = np.nonzero(ok[s])[0]
            best = fr[np.argmin(dist[s, fr])]
            frame_idx = (pc.shape[1] - 1) if frames == "last" else int(best)
            hits.append({
                "iter": int(it),
                "seg": int(s),
                "frame": frame_idx,
                "n_matching_frames": int(fr.size),
                "dist": float(dist[s, best]),
                "weight": float(w[s]),
                "cv": [float(x) for x in pc[s, frame_idx, :]],
            })
    return hits


def nearest(h5, max_iter, cv, frames="last", iter_lo=1):
    """Closest populated point to `cv` -- for a helpful message on no hits."""
    cv = np.asarray(cv, dtype=np.float64)
    best = None
    for it, pc, w in scan(h5, max_iter, iter_lo):
        sel = pc[:, -1:, :] if frames == "last" else pc
        d = np.sqrt(((sel - cv[None, None, :]) ** 2).sum(axis=2))
        i = int(np.argmin(d))
        s, f = np.unravel_index(i, d.shape)
        if best is None or d[s, f] < best["dist"]:
            frame_idx = (pc.shape[1] - 1) if frames == "last" else int(f)
            best = {"iter": int(it), "seg": int(s), "frame": frame_idx,
                    "dist": float(d[s, f]), "weight": float(w[s]),
                    "cv": [float(x) for x in pc[s, frame_idx, :]]}
    return best


def cv_extent(h5, max_iter, iter_lo=1):
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for _, pc, _ in scan(h5, max_iter, iter_lo):
        flat = pc.reshape(-1, pc.shape[-1])
        lo = np.minimum(lo, flat.min(axis=0))
        hi = np.maximum(hi, flat.max(axis=0))
    return lo, hi
