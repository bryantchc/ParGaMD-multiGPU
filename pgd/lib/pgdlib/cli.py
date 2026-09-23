"""Shared argparse groups and table rendering for pgd subcommands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import frames as F
from . import paths as P
from . import solvent as S


def base_parser(description: str) -> argparse.ArgumentParser:
    # argv[0] is cmds/<name>.py, so usage reads "pgd load", not bare "pgd".
    p = argparse.ArgumentParser(
        prog="pgd %s" % Path(sys.argv[0]).stem, description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help=argparse.SUPPRESS)   # the dispatcher already set it
    p.add_argument("--max-iter", type=int, default=None,
                   help="cap the iteration range (default: last complete iter)")
    return p


def add_tip_args(p) -> None:
    g = p.add_argument_group("lineage tip")
    g.add_argument("--iter", type=int, help="tip iteration")
    g.add_argument("--seg", type=int, help="tip segment id")
    g.add_argument("--tip", help="tip as ITER:SEG (alternative to --iter/--seg)")
    g.add_argument("--stop-iter", type=int, default=1,
                   help="stop walking at this iteration (default 1 = the root)")


def add_slice_args(p) -> None:
    g = p.add_argument_group("frame selection")
    g.add_argument("--stride", type=int, default=1,
                   help="keep every Nth frame, phase-continuous across seams")
    g.add_argument("--seam", choices=["auto", "keep", "drop"], default="auto",
                   help="whether frame 0 of a child duplicates its parent's "
                        "last frame. 'auto' measures it (default). This run "
                        "measures 'keep' -- see pgd trace.")
    g.add_argument("--solvent", choices=["stripped", "full", "auto"],
                   default="stripped",
                   help="atom set to normalize the lineage to (default stripped)")
    g.add_argument("--tip-frame", type=int, default=None,
                   help="truncate the tip segment at this local frame (0-99)")


def add_output_args(p) -> None:
    g = p.add_argument_group("output")
    g.add_argument("--json", action="store_true", help="machine-readable output")
    g.add_argument("--tsv", action="store_true", help="tab-separated output")


def resolve_tip(args) -> tuple[int, int]:
    if args.tip:
        try:
            it, sg = args.tip.split(":")
            return int(it), int(sg)
        except ValueError:
            raise SystemExit("pgd: --tip must be ITER:SEG, e.g. --tip 115:42")
    if args.iter is None or args.seg is None:
        raise SystemExit(
            "pgd: specify the lineage tip with --iter N --seg S (or --tip N:S)")
    return args.iter, args.seg


def open_run(args):
    """Return (RunPaths, h5file, max_iter).

    The caller MUST call release(h5) as soon as it is done reading, and in
    particular before spawning any long-lived child. HDF5 does not set
    FD_CLOEXEC, so an open handle survives os.execvp into the child and holds
    the HDF5 file lock for that child's whole lifetime -- which is how a VMD
    session ends up blocking `w_run` with BlockingIOError.
    """
    import h5py
    from . import lineage as L
    rp = P.run_paths(args.root)
    if not rp.west_h5.is_file():
        raise SystemExit("pgd: no west.h5 at %s" % rp.west_h5)
    # Read-only, always. PGD_H5_NO_LOCK skips the HDF5 lock entirely so pgd can
    # read while WESTPA holds the file for writing -- opt-in, because an
    # unlocked read of a file being written can tear.
    kw = {}
    if os.environ.get("PGD_H5_NO_LOCK"):
        kw["locking"] = False
    try:
        h5 = h5py.File(rp.west_h5, "r", **kw)
    except OSError as e:
        if "lock" in str(e).lower() or isinstance(e, BlockingIOError):
            raise SystemExit(
                "pgd: cannot open %s -- another process holds the HDF5 lock "
                "(check: lsof %s).\n"
                "     If WESTPA is running and you only want to read, retry "
                "with PGD_H5_NO_LOCK=1." % (rp.west_h5, rp.west_h5))
        raise
    cap = L.max_complete_iter(h5, rp)
    if not rp.traj_segs.is_dir():
        eprint("[pgd] note: %s does not exist -- west.h5 analysis (find, info, "
               "fes) works, but load/export/strip-copy have no coordinates."
               % rp.traj_segs)
    elif not L.trajectories_present(rp, cap):
        eprint("[pgd] note: no on-disk segments for iteration %d; commands "
               "needing coordinates will name the missing files." % cap)
    max_iter = min(args.max_iter, cap) if args.max_iter else cap
    if args.max_iter and args.max_iter > cap:
        eprint("[pgd] iteration %d is not complete; capping at %d"
               % (args.max_iter, cap))
    return rp, h5, max_iter


def release(h5) -> None:
    """Close the west.h5 handle.

    Call this the moment the command is done reading -- always before exec or
    fork. HDF5 has no FD_CLOEXEC, so a live handle is inherited by children and
    keeps the file locked against WESTPA. Safe to call more than once.
    """
    try:
        if h5 is not None and h5.id and h5.id.valid:
            h5.close()
    except Exception:                                       # noqa: BLE001
        pass


def eprint(*a):
    print(*a, file=sys.stderr)


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def table(rows, headers, aligns=None) -> str:
    """Fixed-width table. rows: list of tuples of already-stringified cells."""
    if not rows:
        return "(none)"
    cols = len(headers)
    w = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], len(str(r[i])))
    aligns = aligns or ["<"] * cols
    fmt = "  ".join("{:%s%d}" % (aligns[i], w[i]) for i in range(cols))
    out = [fmt.format(*headers), fmt.format(*["-" * x for x in w])]
    for r in rows:
        out.append(fmt.format(*[str(c) for c in r]))
    return "\n".join(out)


def resolve_seam(h5, chain, requested: str):
    """Turn --seam auto into a measured verdict, with evidence."""
    if requested in (F.SEAM_KEEP, F.SEAM_DROP):
        return requested, {"mode": "explicit"}
    verdict, ev = F.detect_seam(h5, chain)
    ev["mode"] = "auto"
    return verdict, ev


def dry_run() -> bool:
    return bool(os.environ.get("PGD_DRY_RUN"))


def assume_yes() -> bool:
    return bool(os.environ.get("PGD_YES"))
