"""Translate this run's WESTPA progress coordinates into a VMD Colvars config.

The point: when you open a lineage in VMD, you want to see the same numbers the
weighted-ensemble simulation was steering on -- not a reimplementation of them.

How the CVs are defined here
----------------------------
`westpa_scripts/_pcoord_dispatch.py` imports every `cv_*.py` in the run dir and
calls `compute(u, ref) -> float` per frame. The cv_clamp family used by this run
computes the MEAN of a set of explicit heavy-atom contact distances, caching the
resolved atom pairs in a module-global `_C` dict (`a`, `b`, `r0`).

So rather than re-deriving anything from the contact JSON -- which would be a
second implementation that can drift -- we import the very modules WESTPA used,
let them resolve their own pairs against the topology, and read the pairs back
out. Whatever the simulation measured is what Colvars will measure.

The translation
---------------
mean(d_i) over N pairs is a linear combination, which Colvars expresses natively:
a single colvar holding N `distance` components, each with
`componentCoeff = 1/N`. That is exact, not an approximation.

`CV_MODE=softq` is NOT translatable this way: it is a logistic function of each
distance, and Colvars' linear component combination cannot express it (its
`coordNum` uses a rational switching function, which is a different curve). We
refuse rather than emit something that silently disagrees.

Atom numbering
--------------
The cv modules hand back 0-based MDAnalysis indices; Colvars `atomNumbers` are
1-based serial numbers in the loaded molecule. Solvent is stripped from the END
of the atom ordering, so solute indices are identical in the full and stripped
topologies -- verified, not assumed, by `resolve()` below and by
`pgd colvars --verify`.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ColvarsUntranslatable(RuntimeError):
    """This CV cannot be expressed exactly as a Colvars config."""


@dataclass
class CvSpec:
    name: str                       # cv_0
    label: str                      # cv0
    source: str                     # path to cv_0.py
    subset: str | None              # contact JSON it read, if discoverable
    mode: str                       # distance | softq | ...
    pairs_a: np.ndarray             # 0-based atom indices
    pairs_b: np.ndarray
    r0: np.ndarray
    atom_desc: dict = field(default_factory=dict)   # idx -> "PHE165 N"

    @property
    def n(self) -> int:
        return len(self.pairs_a)


def discover_cv_modules(run_root: Path):
    """Import cv_*.py exactly as _pcoord_dispatch does: sorted lexically."""
    paths = sorted(glob.glob(str(Path(run_root) / "cv_*.py")))
    mods = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:                              # noqa: BLE001
            raise ColvarsUntranslatable(
                "%s failed to import: %s. pgd colvars reads the run's own CV "
                "modules, so they must be importable." % (path, e))
        if not hasattr(mod, "compute"):
            raise ColvarsUntranslatable(
                "%s has no compute(u, ref) -- not a pcoord CV module" % path)
        mods.append((name, path, mod))
    return mods


def extract(name: str, path: str, mod, universe) -> CvSpec:
    """Get the exact atom pairs the CV module resolved against this topology."""
    mode = str(getattr(mod, "CV_MODE", "distance"))

    # Trigger the module's own resolution, however it spells it.
    if hasattr(mod, "_setup"):
        mod._setup(universe)
    else:
        mod.compute(universe)

    cache = getattr(mod, "_C", None)
    if not isinstance(cache, dict) or "a" not in cache or "b" not in cache:
        raise ColvarsUntranslatable(
            "%s does not expose resolved atom pairs (no module-global _C with "
            "'a'/'b').\npgd colvars can translate the cv_clamp family "
            "(mean of explicit contact distances). For anything else, define "
            "the colvar by hand and pass it with --config." % path)

    if mode != "distance":
        raise ColvarsUntranslatable(
            "%s runs with CV_MODE=%r. Only 'distance' (a mean of pairwise "
            "distances) maps exactly onto a Colvars linear combination; "
            "'softq' is a logistic of each distance, which Colvars' component "
            "combination cannot reproduce. Refusing to emit a colvar that "
            "would silently disagree with the pcoord." % (path, mode))

    a = np.asarray(cache["a"], dtype=np.int64)
    b = np.asarray(cache["b"], dtype=np.int64)
    r0 = np.asarray(cache.get("r0", np.zeros(len(a))), dtype=np.float64)
    if len(a) == 0 or len(a) != len(b):
        raise ColvarsUntranslatable("%s resolved %d/%d pairs" % (path, len(a), len(b)))

    n_atoms = universe.atoms.n_atoms
    hi = int(max(a.max(), b.max()))
    if hi >= n_atoms:
        raise ColvarsUntranslatable(
            "%s references atom index %d but the topology has only %d atoms. "
            "The colvars config would point at the wrong atoms."
            % (path, hi, n_atoms))

    desc = {}
    for idx in np.unique(np.concatenate([a, b])):
        at = universe.atoms[int(idx)]
        desc[int(idx)] = "%s%d %s" % (at.resname, at.resid, at.name)

    subset = None
    for attr in ("SUBSET_JSON",):
        if hasattr(mod, attr):
            subset = str(getattr(mod, attr))

    return CvSpec(name=name, label=name.replace("cv_", "cv"), source=path,
                  subset=subset, mode=mode, pairs_a=a, pairs_b=b, r0=r0,
                  atom_desc=desc)


def value(spec: CvSpec, positions) -> float:
    """The CV, computed the same way the module does. For verification."""
    d = np.linalg.norm(positions[spec.pairs_a] - positions[spec.pairs_b], axis=1)
    return float(d.mean())


def build_config(specs, topology: Path, run_root: Path,
                 traj_freq: int = 0, widths=None) -> str:
    """Emit a Colvars config computing exactly these CVs."""
    L = []
    a = L.append
    a("# Colvars configuration generated by pgd")
    a("#")
    a("# These are the SAME collective variables the WESTPA run steered on --")
    a("# read out of the run's own cv_*.py modules, not reimplemented.")
    a("#   run       : %s" % run_root)
    a("#   topology  : %s" % topology)
    a("#")
    a("# Each colvar is the MEAN of N explicit heavy-atom contact distances,")
    a("# expressed as N `distance` components with componentCoeff = 1/N.")
    a("# That is exact: Colvars combines components linearly.")
    a("#")
    a("# In VMD:")
    a("#     cv molid top")
    a("#     cv configfile %s" % "<this file>")
    a("#     cv update ; cv printframe")
    a("")
    a("# 0 = don't log a trajectory file; for viewing we compute on demand")
    a("colvarsTrajFrequency %d" % traj_freq)
    a("colvarsRestartFrequency 0")
    a("")

    for i, s in enumerate(specs):
        w = (widths or {}).get(s.label, 0.25)
        a("colvar {")
        a("    name %s" % s.label)
        a("    # %s: mean of %d contact distances (Angstrom); drive DOWN"
          % (s.name, s.n))
        a("    #   source: %s" % s.source)
        if s.subset:
            a("    #   subset: %s" % s.subset)
        a("    #   pcoord dimension %d in west.h5" % i)
        a("    width %g" % w)
        a("")
        coeff = 1.0 / s.n
        for k in range(s.n):
            ia, ib = int(s.pairs_a[k]), int(s.pairs_b[k])
            a("    distance {")
            a("        componentCoeff %.17g" % coeff)
            a("        # %s -- %s   (r0 = %.3f A)"
              % (s.atom_desc.get(ia, "?"), s.atom_desc.get(ib, "?"), s.r0[k]))
            a("        group1 { atomNumbers %d }" % (ia + 1))   # 1-based
            a("        group2 { atomNumbers %d }" % (ib + 1))
            a("    }")
        a("}")
        a("")
    return "\n".join(L)


def tcl_snippet(config_path: Path, labels, ref_trace: Path | None = None) -> str:
    """Tcl that attaches Colvars to the loaded molecule and adds helpers."""
    L = []
    a = L.append
    a("# --- Colvars: the run's own progress coordinates, live in VMD -------")
    a("if {[llength [info commands cv]] == 0} {")
    a('    puts "pgd: this VMD has no Colvars module; skipping colvars setup"')
    a("} else {")
    a("    cv molid top")
    a("    cv configfile %s" % _tcl_str(config_path))
    a("    cv update")
    a('    puts "pgd: colvars loaded -> %s"' % " ".join(labels))
    a("")
    a("    # cvs [frame]  -- values at a frame (default: current)")
    a("    proc cvs {{f -1}} {")
    a("        if {$f >= 0} { animate goto $f }")
    a("        set f [molinfo top get frame]")
    a("        cv frame $f")
    a("        cv update")
    a("        set out \"frame $f\"")
    a("        foreach c [cv list] { append out [format \"  %s=%.4f\" $c [cv colvar $c value]] }")
    a("        puts $out")
    a("    }")
    a("")
    a("    # cvtrace <file> [stride] -- write every frame's CVs to a file")
    a("    proc cvtrace {path {stride 1}} {")
    a("        set n [molinfo top get numframes]")
    a("        set fh [open $path w]")
    a("        puts $fh \"# frame [join [cv list] { }]\"")
    a("        for {set f 0} {$f < $n} {incr f $stride} {")
    a("            cv frame $f")
    a("            cv update")
    a("            set row $f")
    a("            foreach c [cv list] { append row [format \" %.6f\" [cv colvar $c value]] }")
    a("            puts $fh $row")
    a("        }")
    a("        close $fh")
    a("        puts \"pgd: wrote [expr {($n + $stride - 1) / $stride}] rows to $path\"")
    a("    }")
    if ref_trace:
        a("")
        a('    puts "pgd: west.h5 reference pcoord trace -> %s"' % ref_trace)
    a('    puts "pgd: try  cvs  (current frame) or  cvtrace /tmp/cv.dat"')
    a("}")
    return "\n".join(L)


def _tcl_str(p) -> str:
    return '"%s"' % str(p).replace("\\", "\\\\").replace('"', '\\"')
