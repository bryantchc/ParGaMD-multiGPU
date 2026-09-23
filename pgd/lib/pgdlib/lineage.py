"""Walk a walker's ancestry backwards through west.h5.

Lineage lives *only* in west.h5.  Nothing on disk records a segment's parent.
For a segment (n_iter, seg_id):

    parent_id = west.h5['iterations/iter_%08d/seg_index'][seg_id]['parent_id']

    parent_id >= 0  ->  the parent is (n_iter - 1, parent_id)
    parent_id <  0  ->  this is a root; istate_id = -(parent_id + 1), which
                        resolves through /ibstates to a basis state.

wtg_n_parents / wtg_offset index into /wtgraph and list the *weight-flow*
parents of a merge.  Those are reported by `pgd info` and `pgd trace` but are
never followed: parent_id is the single dynamical parent, and it is the only
path that is continuous MD.

Why walking backwards from one tip is simpler than stitch_lineages.py: that
script partitions *every* segment into vertex-disjoint chains, so it needs
successor selection and tie-breaking.  Each segment has exactly one parent_id,
so walking backwards is unambiguous and needs none of that.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths as P

# seg_index['status'] values (westpa.core.segment.Segment)
SEG_STATUS_COMPLETE = 2
# seg_index['endpoint_type']
ENDPOINT_TYPE_NAMES = {0: "unset", 1: "continues", 2: "merged", 3: "recycled"}


class LineageBroken(RuntimeError):
    pass


@dataclass(frozen=True)
class Node:
    n_iter: int
    seg_id: int
    weight: float
    parent_id: int
    endpoint_type: int
    status: int
    merge_parents: tuple[int, ...] = ()
    flags: tuple[str, ...] = field(default=())

    @property
    def is_root(self) -> bool:
        return self.parent_id < 0

    @property
    def istate_id(self) -> int | None:
        return -(self.parent_id + 1) if self.parent_id < 0 else None


def _seg_index(h5, n_iter):
    """Whole-array read, cached per iteration.

    115 small reads instead of 115 * nseg element reads, and it hands back
    weight/status/endpoint_type in the same pass.
    """
    return h5[P.h5_iter_group(n_iter) + "/seg_index"][:]


class _IterCache:
    def __init__(self, h5):
        self.h5 = h5
        self._si = {}

    def seg_index(self, n_iter):
        if n_iter not in self._si:
            self._si[n_iter] = _seg_index(self.h5, n_iter)
        return self._si[n_iter]


def max_complete_iter(h5, rp: P.RunPaths | None = None,
                      require_disk: bool = False) -> int:
    """Highest iteration that is actually finished.

    Uses summary['walltime'] > 0 as the completed predicate -- the same rule
    pyreweight.load_we_data uses, so the stitcher and the FES always agree on
    the iteration set by construction.  west_current_iteration is NOT used: it
    counts the in-flight iteration (116 here), which has status=1 everywhere.

    `require_disk` additionally demands an on-disk segment directory. That is
    a separate question from completeness: a run whose trajectories have been
    archived still has a perfectly good west.h5, and find/info/fes work on it.
    Commands that actually need coordinates resolve per segment and report the
    specific missing file, which is a better error than a global cap.
    """
    summary = h5["summary"][:]
    completed = [i + 1 for i, row in enumerate(summary) if row["walltime"] > 0]
    if not completed:
        raise LineageBroken("no completed iterations in west.h5 summary")
    n = completed[-1]
    while n >= 1:
        si = _seg_index(h5, n)
        if (si["status"] != SEG_STATUS_COMPLETE).any():
            n -= 1
            continue
        if require_disk and rp is not None and not rp.iter_dir(n).is_dir():
            n -= 1
            continue
        return n
    raise LineageBroken(
        "no iteration is complete%s"
        % (" and present on disk" % () if require_disk else ""))


def trajectories_present(rp: P.RunPaths, n_iter: int) -> bool:
    return rp.iter_dir(n_iter).is_dir()


def merge_parents(h5, n_iter: int, seg_id: int, si=None) -> tuple[int, ...]:
    """Weight-flow parents of a merge. Descriptive only -- never followed."""
    if si is None:
        si = _seg_index(h5, n_iter)
    row = si[seg_id]
    n = int(row["wtg_n_parents"])
    if n <= 1:
        return ()
    off = int(row["wtg_offset"])
    wtg = h5[P.h5_iter_group(n_iter) + "/wtgraph"]
    return tuple(int(x) for x in wtg[off:off + n])


def node(h5, n_iter: int, seg_id: int, cache: _IterCache | None = None) -> Node:
    si = cache.seg_index(n_iter) if cache else _seg_index(h5, n_iter)
    if seg_id >= len(si):
        raise LineageBroken(
            "iteration %d has %d segments; seg %d does not exist"
            % (n_iter, len(si), seg_id)
        )
    row = si[seg_id]
    return Node(
        n_iter=n_iter,
        seg_id=seg_id,
        weight=float(row["weight"]),
        parent_id=int(row["parent_id"]),
        endpoint_type=int(row["endpoint_type"]),
        status=int(row["status"]),
        merge_parents=merge_parents(h5, n_iter, seg_id, si),
    )


def walk_lineage(h5, tip_iter: int, tip_seg: int, stop_iter: int = 1
                 ) -> list[Node]:
    """Return the ancestry of (tip_iter, tip_seg), ordered ROOT -> TIP.

    Follows seg_index['parent_id'] only.  Stops at a root (parent_id < 0) or
    when stop_iter is reached.
    """
    if stop_iter < 1:
        raise ValueError("stop_iter must be >= 1")
    cache = _IterCache(h5)
    chain: list[Node] = []
    it, seg = tip_iter, tip_seg
    while it >= stop_iter:
        nd = node(h5, it, seg, cache)
        chain.append(nd)
        if nd.is_root:
            break
        it, seg = it - 1, nd.parent_id
    chain.reverse()
    if not chain:
        raise LineageBroken("empty lineage for %d:%d" % (tip_iter, tip_seg))
    return chain


def basis_state_for(h5, rp: P.RunPaths, nd: Node):
    """Resolve a root node's istate back to its basis state file."""
    if not nd.is_root:
        return None
    istate_id = nd.istate_id
    try:
        grp = h5["ibstates/0"]
        ist = grp["istate_index"][istate_id]
        bstate_id = int(ist["basis_state_id"])
        bst = grp["bstate_index"][bstate_id]
        auxref = bst["auxref"]
        if isinstance(auxref, bytes):
            auxref = auxref.decode()
        label = bst["label"]
        if isinstance(label, bytes):
            label = label.decode()
    except (KeyError, IndexError, ValueError) as e:
        return {"istate_id": istate_id, "error": str(e)}
    return {
        "istate_id": istate_id,
        "basis_state_id": bstate_id,
        "auxref": auxref,
        "label": label,
        "path": str(rp.system_dir / ("%s.rst7" % rp.system_name)),
    }


def children_of(h5, n_iter: int, seg_id: int) -> list[int]:
    """Segments in n_iter+1 whose parent_id is seg_id."""
    try:
        si = _seg_index(h5, n_iter + 1)
    except KeyError:
        return []
    return [int(i) for i in np.flatnonzero(si["parent_id"] == seg_id)]


@functools.lru_cache(maxsize=256)
def frozen_segments(tar_path: str, n_iter: int) -> frozenset[int]:
    """Segments whose DCD was copied verbatim from the parent.

    runseg.sh's freeze-fallback copies the parent's trajectory forward when all
    NaN retries fail.  Those frames are duplicated, not new sampling.  We grep
    the tarred seg_logs directly -- no extraction.

    Zero occurrences in Ultra_RL2.rna; this exists for other runs.
    """
    p = Path(tar_path)
    if not p.is_file():
        return frozenset()
    try:
        out = subprocess.run(
            ["grep", "-a", "-o", "-E", r"FROZEN[^0-9]*seg[ =:]*[0-9]+", str(p)],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    segs = set()
    for line in out.splitlines():
        digits = "".join(ch for ch in line if ch.isdigit())
        if digits:
            segs.add(int(digits))
    return frozenset(segs)


def retried_segments(rp: P.RunPaths, n_iter: int) -> set[int]:
    """Segments with a gamdRunner_attempt2.out -- reseeded after a NaN.

    The DCD is genuine MD; this is reported, never acted on.
    """
    d = rp.iter_dir(n_iter)
    if not d.is_dir():
        return set()
    out = set()
    for p in d.glob("*/gamdRunner_attempt2.out"):
        try:
            out.add(int(p.parent.name))
        except ValueError:
            pass
    return out
