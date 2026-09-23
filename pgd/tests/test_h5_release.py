#!/usr/bin/env python3
"""west.h5 must never be inherited by a child process.

HDF5 does not set FD_CLOEXEC, so a handle still open at os.execvp survives into
the child and holds the HDF5 file lock for that child's entire lifetime. When
the child is a VMD session, that blocks `w_run` with BlockingIOError -- which is
exactly the bug this guards.

Two checks:
  1. cli.release() actually invalidates the handle.
  2. Every command that spawns a long-lived child releases before doing so.
     Checked by source inspection, because the runtime check needs a real run.

The end-to-end version lives in the README: substitute $PGD_VMD with a probe
that greps its own /proc/self/fd.

    python tests/test_h5_release.py
"""
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_release_invalidates_handle():
    import h5py
    import numpy as np
    from pgdlib import cli
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "west.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("x", data=np.arange(4))
        h5 = h5py.File(p, "r")
        check(bool(h5.id.valid), "handle should start valid")
        cli.release(h5)
        check(not h5.id.valid, "release() left the handle open")
        cli.release(h5)          # must be idempotent, not raise


def test_commands_release_before_spawning():
    """Any command that execs or forks must call cli.release first."""
    spawners = {
        "load.py": r"os\.execvp",
        # the instantiation, not the import line
        "strip-copy.py": r"ProcessPoolExecutor\(",
    }
    for fn, spawn_re in spawners.items():
        src = (ROOT / "cmds" / fn).read_text()
        m = re.search(spawn_re, src)
        check(m is not None, "%s: no %s found -- test is stale" % (fn, spawn_re))
        before = src[:m.start()]
        check("cli.release(" in before,
              "%s calls %s at offset %d with no cli.release() before it -- "
              "west.h5 would be inherited by the child"
              % (fn, spawn_re, m.start()))


def test_every_command_releases():
    """Defensive: each command that opens a run should also release it."""
    for f in sorted((ROOT / "cmds").glob("*.py")):
        src = f.read_text()
        if "cli.open_run(" not in src:
            continue
        check("cli.release(" in src,
              "%s opens a run but never calls cli.release()" % f.name)


def test_no_lock_env_is_honoured():
    from pgdlib import cli
    src = Path(cli.__file__).read_text()
    check("PGD_H5_NO_LOCK" in src and "locking" in src,
          "cli.open_run should honour PGD_H5_NO_LOCK via h5py locking=False")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print("  ok   %s" % t.__name__)
        except Exception:
            FAILS.append(t.__name__)
            print("  FAIL %s" % t.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - len(FAILS), len(tests)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
