"""Frame slicing: which frames of which segment go into the stitched output.

The seam
--------
The standard WESTPA/AMBER convention is that frame 0 of a child segment
reproduces frame 99 of its parent (the "initpoint"), so a stitcher must drop
it.  **This run does not do that.**  The OpenMM-written DCDs carry
ISTART=500, NSAVC=500 -- the first frame is written at step 500, i.e. 2 ps
*into* the segment.  There is no t=0 frame in the file, so there is nothing to
duplicate.  Measured over 3,788 non-root segments: zero exact pcoord matches
at the seam, and the seam displacement equals an ordinary 2 ps step.

So `detect_seam` MEASURES rather than assumes, and the default is 'auto'.
Dropping frame 0 on this run would delete 114 real frames per lineage and open
a 4 ps gap at every seam.

(files_dist_cv/stitch_lineages.py drops frame 0 by default; the existing
tica_chains/ were built that way.  Do not inherit that.)

The stride
----------
Stride is applied on the GLOBAL concatenated timeline, so the phase carries
across seams.  `--stride 3` over 100-frame segments gives true uniform 6 ps
spacing even though 3 does not divide 100.  Neither catdcd (whose -stride is
global but cannot be combined with per-file trimming) nor cpptraj (whose
trajin offset restarts per file) gets this right on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import paths as P

FRAME_PS = 2.0        # input.xml: dt=0.004 ps * 500-step DCD interval
SEG_FRAMES = 100      # west.cfg pcoord_len; every segment DCD has exactly 100
DCD_INTERVAL_STEPS = 500
DT_PS = 0.004

SEAM_KEEP = "keep"
SEAM_DROP = "drop"


@dataclass(frozen=True)
class Slice:
    """A contiguous strided run of frames taken from one segment.

    first/last are 0-based and INCLUSIVE, python-side.  Converters below hand
    out the 1-based form cpptraj wants and the 0-based form VMD wants.
    """
    n_iter: int
    seg_id: int
    first: int
    last: int
    step: int
    n_frames: int
    t0_ps: float          # global time of this slice's first frame

    def to_cpptraj_trajin(self) -> tuple[int, int, int]:
        """cpptraj: trajin <file> <start> <stop> <offset>, 1-based inclusive."""
        return self.first + 1, self.last + 1, self.step

    def to_vmd_addfile(self) -> tuple[int, int, int]:
        """VMD: mol addfile ... first <f> last <l> step <s>, 0-based."""
        return self.first, self.last, self.step

    def to_byte_range(self) -> tuple[int, int, int]:
        """(first, stop_exclusive, step) for the byte splicer."""
        return self.first, self.last + 1, self.step


def detect_seam(h5, chain, exact_tol: float = 1e-5) -> tuple[str, dict]:
    """Decide whether child frame 0 duplicates parent frame 99.

    Reads only pcoord, so it is nearly free.  Returns (verdict, evidence).
    Raises if the answer is ambiguous rather than guessing -- a half-duplicated
    lineage means something is wrong that a default cannot paper over.
    """
    diffs = []
    for parent, child in zip(chain, chain[1:]):
        p = h5[P.h5_iter_group(parent.n_iter) + "/pcoord"][parent.seg_id, -1, :]
        c = h5[P.h5_iter_group(child.n_iter) + "/pcoord"][child.seg_id, 0, :]
        diffs.append(float(np.abs(np.asarray(p) - np.asarray(c)).max()))

    if not diffs:
        return SEAM_KEEP, {"n_links": 0, "note": "single-segment lineage"}

    arr = np.asarray(diffs)
    frac_exact = float((arr < exact_tol).mean())
    ev = {
        "n_links": len(diffs),
        "frac_exact": frac_exact,
        "max_diff": float(arr.max()),
        "median_diff": float(np.median(arr)),
        "exact_tol": exact_tol,
    }
    if frac_exact > 0.9:
        return SEAM_DROP, ev
    if frac_exact < 0.1:
        return SEAM_KEEP, ev
    raise RuntimeError(
        "ambiguous seam: %.1f%% of %d links are exact duplicates. "
        "Inspect manually and pass --seam keep|drop explicitly."
        % (100 * frac_exact, len(diffs))
    )


def segment_slices(chain, stride: int = 1, seam: str = SEAM_KEEP,
                   seg_frames: int = SEG_FRAMES,
                   tip_last_frame: int | None = None) -> list[Slice]:
    """Turn a root->tip chain into the exact frame slices to concatenate.

    seam='drop'  -> skip local frame 0 of every non-root segment
    seam='keep'  -> take all frames (correct for this run)
    tip_last_frame -> truncate the final segment at this local frame
                      (inclusive), so the trajectory ends on a chosen structure

    Stride phase carries across segment boundaries: `taken` counts emitted
    frames and `g` counts available frames, so the next segment resumes the
    stride pattern where the previous one left off rather than restarting.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if seam not in (SEAM_KEEP, SEAM_DROP):
        raise ValueError("seam must be %r or %r" % (SEAM_KEEP, SEAM_DROP))

    out: list[Slice] = []
    g = 0        # global index of the next available frame
    taken = 0    # count of frames emitted so far

    for k, nd in enumerate(chain):
        base = 1 if (k > 0 and seam == SEAM_DROP) else 0
        end = seg_frames - 1
        if tip_last_frame is not None and k == len(chain) - 1:
            end = min(end, tip_last_frame)
        if end < base:
            g += max(0, end - base + 1)
            continue
        avail = end - base + 1

        # Index (within this segment's available frames) of the next frame the
        # global stride pattern wants.
        offset = (taken * stride) - g
        if offset >= avail:
            g += avail
            continue
        if offset < 0:
            offset = 0

        n = 1 + (avail - 1 - offset) // stride
        first = base + offset
        last = first + (n - 1) * stride
        out.append(Slice(
            n_iter=nd.n_iter, seg_id=nd.seg_id,
            first=first, last=last, step=stride,
            n_frames=n,
            # Original global time of this slice's first frame. Frame 0 of a
            # segment is at 2 ps, not 0, so the +1.
            t0_ps=(taken * stride + 1) * FRAME_PS,
        ))
        taken += n
        g += avail

    return out


def total_frames(slices) -> int:
    return sum(s.n_frames for s in slices)


def total_time_ps(slices) -> float:
    """Wall-clock span the emitted frames cover, in ps."""
    if not slices:
        return 0.0
    step = slices[0].step
    return total_frames(slices) * FRAME_PS * step
