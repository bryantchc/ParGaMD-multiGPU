#!/usr/bin/env python3
"""The write firewall must reject every route into traj_segs.

If this suite ever passes a traj_segs path, pgd can corrupt the data a live
simulation depends on. Standalone (no pytest in the pargamd env).

    python tests/test_guard.py
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from pgdlib.guard import Guard, UnsafeWrite

FAILS = []


def must_reject(g, p, why):
    try:
        g.safe_out(p)
    except UnsafeWrite:
        return
    raise AssertionError("ACCEPTED a path it must reject (%s): %s" % (why, p))


def must_accept(g, p):
    g.safe_out(p)


def build(tmp):
    """Mimic the real layout: run-dir traj_segs is a SYMLINK to scratch."""
    scratch = tmp / "scratch" / "Ultra_RL2.rna"
    real_ts = scratch / "traj_segs"
    (real_ts / "000058" / "000000").mkdir(parents=True)
    (real_ts / "000058" / "000000" / "output_restart.dcd").write_bytes(b"x")
    cache = scratch / "pgd_cache"
    cache.mkdir()
    run = tmp / "run"
    run.mkdir()
    (run / "traj_segs").symlink_to(real_ts)      # the real run-dir symlink
    return run, real_ts, cache


def test_all():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run, real_ts, cache = build(tmp)
        g = Guard(run / "traj_segs", cache)

        # --- the whole point ---------------------------------------------
        must_reject(g, real_ts / "000058/000000/output_restart.dcd",
                    "direct realpath into traj_segs")
        must_reject(g, run / "traj_segs/000058/000000/output_restart.dcd",
                    "via the run-dir symlink")
        must_reject(g, cache / ".." / "traj_segs" / "000058" / "x.dcd",
                    "'..' traversal out of the cache")
        must_reject(g, real_ts / "000058" / "new_file.dcd",
                    "new file inside a segment dir")
        must_reject(g, real_ts / "Ultra_RL2.stripped.parm7",
                    "topology sitting in traj_segs")

        # a symlinked cache pointing back into traj_segs
        sneaky = tmp / "sneaky_cache"
        sneaky.symlink_to(real_ts)
        g2 = Guard(run / "traj_segs", sneaky)
        must_reject(g2, sneaky / "000058" / "x.dcd",
                    "cache root symlinked into traj_segs")

        # --- legitimate writes -------------------------------------------
        (cache / "runs" / "abc").mkdir(parents=True)
        must_accept(g, cache / "runs" / "abc" / "lineage.dcd")
        (cache / "stripped").mkdir()
        must_accept(g, cache / "stripped" / "x.dcd")

        # explicit user output outside the cache is allowed...
        out = tmp / "elsewhere" / "mine.nc"
        out.parent.mkdir()
        g.allow_explicit(out)
        must_accept(g, out)
        # ...but an explicit path into traj_segs is still refused
        try:
            g.allow_explicit(real_ts / "000058" / "mine.nc")
            raise AssertionError("allow_explicit ACCEPTED a traj_segs path")
        except UnsafeWrite:
            pass
        # unregistered path outside the cache is refused
        must_reject(g, tmp / "elsewhere" / "other.nc", "unregistered outside cache")

        # --- cache_write cannot spell an outside destination --------------
        for bad in ("/etc/passwd", "../traj_segs/x.dcd", "a/../../x"):
            try:
                g.cache_write(bad)
                raise AssertionError("cache_write ACCEPTED %r" % bad)
            except UnsafeWrite:
                pass
        p = g.cache_write("runs/xyz/load.tcl")
        assert p.parent.is_dir()


def main():
    try:
        test_all()
        print("  ok   test_all")
        print("\n1/1 passed")
        return 0
    except Exception:
        traceback.print_exc()
        print("\n0/1 passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
