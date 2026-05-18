#!/usr/bin/env python
"""
_pcoord_dispatch.py — single-Universe pcoord computation.

Usage:
    _pcoord_dispatch.py <topology> <trajectory_or_rst7> <ref_pdb_or_->  [<cv_dir>]

Discovers cv_*.py in <cv_dir> (default: CWD, then $WEST_SIM_ROOT). Each
cv_N.py is imported as a module and is expected to expose a `compute(u, ref)`
callable that, given the Universe positioned at the current frame and the
reference Universe, returns one float.

The dispatcher loads MDAnalysis once, opens the Universe once, loops the
trajectory once, calls each module's compute() per frame, and writes a
whitespace-separated row of floats per frame to stdout — the format
WESTPA's $WEST_PCOORD_RETURN expects.

A single Universe is shared across CVs so that Python interpreter startup
+ MDAnalysis import + parm/dcd parse happens ONCE per walker, not once
per CV. See README → "Reweighting / hardware notes" for the per-walker
cost analysis that motivates this design.

Pass "-" as the ref_pdb if the run dir has no reference structure (some
CVs like Rg or distance metrics don't need one). cv modules that don't
use the reference can ignore the argument; cv modules that require one
will raise AttributeError, which surfaces as a clean WESTPA-side error.
"""
import importlib.util
import glob
import os
import sys

import MDAnalysis as mda


def _load_cv_modules(cv_dir):
    """Import every cv_*.py in cv_dir, sorted lex (cv_0 < cv_1 < ...)."""
    paths = sorted(glob.glob(os.path.join(cv_dir, "cv_*.py")))
    mods = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "compute"):
            sys.exit(f"[dispatch] {path}: missing required compute(u, ref) function")
        mods.append((name, mod))
    return mods


def _open_universe(topology, traj):
    # MDAnalysis can't infer .rst7 → INPCRD from extension alone; spell it
    # out for AMBER restart files.
    if traj.endswith((".rst7", ".rst", ".inpcrd")):
        return mda.Universe(topology, traj, format="INPCRD")
    return mda.Universe(topology, traj)


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: _pcoord_dispatch.py <topology> <trajectory> <ref_pdb_or_-> [<cv_dir>]")

    topology, traj, ref_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    cv_dir = sys.argv[4] if len(sys.argv) > 4 else os.getcwd()

    # Locate cv_*.py: try the given dir first, then $WEST_SIM_ROOT as a
    # fallback (so this script Just Works when invoked from a segment dir
    # without an explicit cv_dir arg).
    if not glob.glob(os.path.join(cv_dir, "cv_*.py")):
        wsr = os.environ.get("WEST_SIM_ROOT", "")
        if wsr and glob.glob(os.path.join(wsr, "cv_*.py")):
            cv_dir = wsr

    modules = _load_cv_modules(cv_dir)
    if not modules:
        sys.exit(
            f"[dispatch] no cv_*.py modules found in {cv_dir} or $WEST_SIM_ROOT — "
            "scaffold the run dir with new_run.sh or add CV files manually"
        )

    u = _open_universe(topology, traj)
    ref = mda.Universe(ref_arg) if ref_arg != "-" else None

    out = sys.stdout.write
    for _ts in u.trajectory:
        out(" ".join(f"{mod.compute(u, ref)}" for _, mod in modules))
        out("\n")


if __name__ == "__main__":
    main()
