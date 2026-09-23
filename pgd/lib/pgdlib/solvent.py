"""Classify a segment's solvent state and resolve it to a usable DCD.

Iterations 58, 114 and 115 of Ultra_RL2.rna are still full-atom (234,549
atoms); every other iteration has been solvent-stripped to 24,790 atoms and
carries a zero-byte `.stripped` marker.  A lineage therefore spans two atom
counts and cannot be concatenated until it is normalized.

Normalizing means stripping COPIES of the full-atom segments into a cache.
The originals are never touched: the live simulation depends on them.

Note that VMD's failure mode here is silent -- loading a 234,549-atom DCD into
a 24,790-atom molecule prints "ERROR) Incorrect number of atoms" to stderr but
raises no Tcl error and loads nothing.  So mismatches must be caught here,
before VMD ever sees them.
"""

from __future__ import annotations

import functools
from enum import Enum
from pathlib import Path

from . import dcdio
from . import paths as P


class SolventState(str, Enum):
    STRIPPED = "stripped"
    FULL = "full"
    MISSING = "missing"
    UNKNOWN = "unknown"


class NeedsStripCopy(RuntimeError):
    """Raised when a lineage needs full-atom segments normalized first.

    Carries every offending segment, so the user gets one actionable message
    instead of three consecutive failures.
    """

    def __init__(self, pairs, tip=None):
        self.pairs = list(pairs)
        self.tip = tip
        spec = ",".join("%d:%d" % (i, s) for i, s in self.pairs)
        cmd = ("pgd strip-copy --for-lineage %d:%d" % tip) if tip else \
              ("pgd strip-copy --segments %s" % spec)
        super().__init__(
            "%d segment(s) on this lineage are still full-atom and have no "
            "stripped copy in the cache: %s\n"
            "The originals will not be modified. Run:\n    %s"
            % (len(self.pairs), spec, cmd))


@functools.lru_cache(maxsize=8)
def topo_natoms(parm7: str) -> int:
    """Atom count of a topology. Measured, never hardcoded."""
    import MDAnalysis as mda
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(mda.Universe(str(parm7)).atoms.n_atoms)


def classify(rp: P.RunPaths, n_iter: int, seg_id: int,
             strict: bool = False) -> SolventState:
    """Solvent state of a segment.

    Fast path is the `.stripped` marker -- one stat, correct for 69,556 of the
    72,852 segments here.  `strict` forces the authoritative check (reading the
    DCD header) at the cost of an open per segment.
    """
    dcd = rp.seg_dcd(n_iter, seg_id)
    if not dcd.is_file():
        return SolventState.MISSING
    if not strict and rp.seg_stripped_marker(n_iter, seg_id).is_file():
        return SolventState.STRIPPED
    try:
        n = dcdio.dcd_natoms(dcd)
    except dcdio.DcdLayoutError:
        return SolventState.UNKNOWN
    if n == topo_natoms(str(rp.stripped_topo)):
        return SolventState.STRIPPED
    if n == topo_natoms(str(rp.full_topo)):
        return SolventState.FULL
    raise RuntimeError(
        "iter %d seg %d: DCD has %d atoms, matching neither %s (%d) nor %s (%d)"
        % (n_iter, seg_id, n, rp.stripped_topo.name,
           topo_natoms(str(rp.stripped_topo)), rp.full_topo.name,
           topo_natoms(str(rp.full_topo))))


def topology_for(rp: P.RunPaths, state: SolventState) -> Path:
    if state is SolventState.STRIPPED:
        return rp.stripped_topo
    if state is SolventState.FULL:
        return rp.full_topo
    raise ValueError("no topology for solvent state %r" % state)


def cached_strip_is_valid(rp: P.RunPaths, n_iter: int, seg_id: int) -> bool:
    """Is there a usable stripped copy of this full-atom segment?

    Validity is checked against the source's size+mtime and the strip mask, so
    a re-run simulation or a changed mask invalidates the entry automatically.
    """
    import json
    out = rp.cache_seg_dcd(n_iter, seg_id)
    meta = out.parent / "meta.json"
    if not (out.is_file() and meta.is_file()):
        return False
    try:
        m = json.loads(meta.read_text())
        src = rp.seg_dcd(n_iter, seg_id)
        st = src.stat()
        return (m.get("src_size") == st.st_size
                and m.get("src_mtime_ns") == st.st_mtime_ns
                and m.get("strip_mask") == rp.strip_mask
                and m.get("out_natoms") == topo_natoms(str(rp.stripped_topo))
                and m.get("out_nframes") == dcdio.probe(out).nframes)
    except (OSError, ValueError, KeyError, dcdio.DcdLayoutError):
        return False


def resolve_chain(rp: P.RunPaths, chain, want: SolventState = SolventState.STRIPPED,
                  tip=None, strict: bool = False):
    """Resolve every segment of a lineage to a DCD with a uniform atom count.

    Returns (topology, [(path, state)]) parallel to `chain`.
    Raises NeedsStripCopy listing EVERY offending segment at once.
    """
    resolved = []
    missing_strip = []
    absent = []

    for nd in chain:
        st = classify(rp, nd.n_iter, nd.seg_id, strict=strict)
        if st is SolventState.MISSING:
            absent.append((nd.n_iter, nd.seg_id))
            resolved.append((None, st))
            continue

        if want is SolventState.STRIPPED:
            if st is SolventState.STRIPPED:
                resolved.append((rp.seg_dcd(nd.n_iter, nd.seg_id), st))
            else:
                if cached_strip_is_valid(rp, nd.n_iter, nd.seg_id):
                    resolved.append((rp.cache_seg_dcd(nd.n_iter, nd.seg_id),
                                     SolventState.STRIPPED))
                else:
                    missing_strip.append((nd.n_iter, nd.seg_id))
                    resolved.append((None, st))
        elif want is SolventState.FULL:
            if st is SolventState.FULL:
                resolved.append((rp.seg_dcd(nd.n_iter, nd.seg_id), st))
            else:
                raise RuntimeError(
                    "iter %d seg %d was stripped in place; the solvent atoms no "
                    "longer exist on disk. --solvent full is impossible for any "
                    "lineage that crosses a stripped iteration."
                    % (nd.n_iter, nd.seg_id))
        else:
            resolved.append((rp.seg_dcd(nd.n_iter, nd.seg_id), st))

    if absent:
        raise RuntimeError(
            "missing trajectory for %d segment(s): %s"
            % (len(absent), ", ".join("%d:%d" % x for x in absent)))
    if missing_strip:
        raise NeedsStripCopy(missing_strip, tip=tip)

    states = {s for _, s in resolved}
    if len(states) != 1:
        raise RuntimeError("mixed solvent states after resolution: %r" % states)
    return topology_for(rp, states.pop()), resolved
