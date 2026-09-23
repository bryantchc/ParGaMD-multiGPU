#!/usr/bin/env python3
"""Unit tests for the frame-slicing arithmetic.

This is the only non-obvious computation in pgd, and everything downstream
(VMD load, cpptraj export, the byte splicer, provenance) trusts it.  The
invariant under test: the emitted slices must reproduce exactly the frames a
naive global concatenate-then-stride would produce -- no duplicates, no gaps,
correct stride phase across segment boundaries.

Standalone by design: `pytest` is not installed in the `pargamd` env and this
suite must not require adding anything to the env a live simulation is using.

    python tests/test_frames.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from pgdlib import frames as F

_FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


class FakeNode:
    def __init__(self, n_iter, seg_id):
        self.n_iter, self.seg_id = n_iter, seg_id


def chain(n):
    return [FakeNode(i + 1, i * 10) for i in range(n)]


def expand(slices):
    """Flatten slices into a global (iter, seg, local_frame) list."""
    out = []
    for s in slices:
        idxs = list(range(s.first, s.last + 1, s.step))
        check(len(idxs) == s.n_frames,
              "n_frames %d != len(range) %d" % (s.n_frames, len(idxs)))
        check((s.last - s.first) % s.step == 0,
              "last-first not a multiple of step")
        out += [(s.n_iter, s.seg_id, f) for f in idxs]
    return out


def reference(chain_, stride, seam, seg_frames=F.SEG_FRAMES, tip_last=None):
    """Naive model: build the whole global frame list, then stride it."""
    glob = []
    for k, nd in enumerate(chain_):
        base = 1 if (k > 0 and seam == F.SEAM_DROP) else 0
        end = seg_frames - 1
        if tip_last is not None and k == len(chain_) - 1:
            end = min(end, tip_last)
        glob += [(nd.n_iter, nd.seg_id, f) for f in range(base, end + 1)]
    return glob[::stride]


# ---------------------------------------------------------------- tests ----
def test_matches_naive_global_stride():
    for nseg in (1, 2, 3, 7, 115):
        for stride in (1, 2, 3, 5, 7, 50, 100, 250):
            for seam in (F.SEAM_KEEP, F.SEAM_DROP):
                c = chain(nseg)
                got = expand(F.segment_slices(c, stride=stride, seam=seam))
                exp = reference(c, stride, seam)
                check(got == exp,
                      "nseg=%d stride=%d seam=%s: %d frames vs %d expected"
                      % (nseg, stride, seam, len(got), len(exp)))


def test_no_duplicates_or_gaps_stride_one():
    got = expand(F.segment_slices(chain(115), stride=1, seam=F.SEAM_KEEP))
    check(len(got) == 115 * 100, "expected 11500 frames, got %d" % len(got))
    check(len(set(got)) == len(got), "duplicate frames emitted")


def test_keep_vs_drop_frame_counts():
    c = chain(115)
    keep = F.total_frames(F.segment_slices(c, seam=F.SEAM_KEEP))
    drop = F.total_frames(F.segment_slices(c, seam=F.SEAM_DROP))
    check(keep == 11500, "keep should be 11500, got %d" % keep)
    check(drop == 100 + 114 * 99, "drop should be 11386, got %d" % drop)
    check(keep - drop == 114,
          "dropping the seam would discard 114 real frames, got %d" % (keep - drop))


def test_tip_truncation():
    for stride in (1, 3, 7):
        c = chain(5)
        sl = F.segment_slices(c, stride=stride, seam=F.SEAM_KEEP,
                              tip_last_frame=40)
        check(expand(sl) == reference(c, stride, F.SEAM_KEEP, tip_last=40),
              "tip truncation mismatch at stride=%d" % stride)
        check(sl[-1].last <= 40, "tip slice ran past the requested frame")


def test_stride_phase_carries_across_seam():
    """100 %% 3 != 0, so a per-file offset would restart the pattern and
    produce a short interval at every seam."""
    c = chain(3)
    sl = F.segment_slices(c, stride=3, seam=F.SEAM_KEEP)
    order = {(s.n_iter, s.seg_id): i for i, s in enumerate(sl)}
    glob = [order[(it, sg)] * 100 + f for it, sg, f in expand(sl)]
    check(glob == list(range(0, 300, 3)),
          "global stride pattern broken: %s..." % glob[:8])
    check(sl[1].first == 2,
          "second segment should resume at local frame 2, got %d" % sl[1].first)


def test_t0_ps_is_original_global_time():
    sl = F.segment_slices(chain(3), stride=1, seam=F.SEAM_KEEP)
    # Frame 0 of a segment sits at 2 ps, not 0 (ISTART=NSAVC=500).
    for i, want in enumerate((2.0, 202.0, 402.0)):
        check(approx(sl[i].t0_ps, want),
              "slice %d t0_ps=%.3f, expected %.3f" % (i, sl[i].t0_ps, want))


def test_converters_round_trip():
    for s in F.segment_slices(chain(2), stride=2, seam=F.SEAM_KEEP):
        start, stop, off = s.to_cpptraj_trajin()
        check((start, stop, off) == (s.first + 1, s.last + 1, s.step),
              "cpptraj converter wrong")
        check(s.to_vmd_addfile() == (s.first, s.last, s.step),
              "vmd converter wrong")
        a, b, st = s.to_byte_range()
        check(len(range(a, b, st)) == s.n_frames, "byte range wrong")


def test_rejects_bad_input():
    for kwargs in ({"stride": 0}, {"seam": "sometimes"}):
        try:
            F.segment_slices(chain(2), **kwargs)
        except ValueError:
            continue
        raise AssertionError("should have rejected %r" % kwargs)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print("  ok   %s" % t.__name__)
        except Exception:
            _FAILS.append(t.__name__)
            print("  FAIL %s" % t.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - len(_FAILS), len(tests)))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
