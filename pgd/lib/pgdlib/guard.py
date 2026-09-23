"""The write firewall.

Hard requirement: pgd must never modify the original trajectory data. The live
simulation depends on iterations 58/114/115 staying full-atom and intact, and
strip_iter.sh -- which pgd deliberately does NOT invoke -- strips in place.

Every filesystem write in the package goes through `safe_out()`. There is no
other `open(..., 'w')`. Two independent checks:

  1. realpath of the target must not be under traj_segs. realpath on BOTH
     sides is what defeats the run-dir `traj_segs -> /scratch/pool/...`
     symlink, any symlink inside a segment dir, and `..` traversal.
  2. realpath of the target must be under the cache root (or an explicitly
     registered output path the user asked for by name, e.g. `export --out`).

`cache_write()` goes further: it accepts only a cache-RELATIVE path, so a
traj_segs destination cannot even be expressed, let alone executed.
"""

from __future__ import annotations

import os
from pathlib import Path


class UnsafeWrite(RuntimeError):
    pass


class Guard:
    def __init__(self, traj_segs, cache_root):
        self.traj_segs = Path(os.path.realpath(traj_segs))
        self.cache_root = Path(os.path.realpath(cache_root))
        self._explicit: set[Path] = set()

    def allow_explicit(self, p) -> Path:
        """Register a user-named output (e.g. `export --out /tmp/x.nc`).

        Still subject to the traj_segs check -- an explicit path never buys a
        way into the simulation tree.
        """
        rp = Path(os.path.realpath(p))
        self._assert_not_traj_segs(rp, p)
        self._explicit.add(rp)
        return rp

    def _assert_not_traj_segs(self, rp: Path, orig) -> None:
        if rp == self.traj_segs or self.traj_segs in rp.parents:
            raise UnsafeWrite(
                "REFUSING to write under traj_segs: %s (resolves to %s). "
                "pgd never modifies original trajectory data." % (orig, rp))

    def safe_out(self, p) -> Path:
        """Validate an output path. Call immediately before every write."""
        p = Path(p)
        # Resolve the PARENT: the file itself may not exist yet.
        parent = Path(os.path.realpath(p.parent))
        rp = parent / p.name
        self._assert_not_traj_segs(rp, p)
        self._assert_not_traj_segs(parent, p.parent)
        if rp in self._explicit:
            return rp
        if parent == self.cache_root or self.cache_root in parent.parents:
            return rp
        raise UnsafeWrite(
            "output %s (resolves to %s) escapes the cache root %s and was not "
            "explicitly registered" % (p, rp, self.cache_root))

    def cache_write(self, relpath: str) -> Path:
        """Resolve a cache-RELATIVE path and create its parent.

        Rejects absolute paths and `..` so that no caller can even spell a
        destination outside the cache.
        """
        s = str(relpath)
        if s.startswith("/") or ".." in Path(s).parts:
            raise UnsafeWrite(
                "cache_write takes a cache-relative path without '..': %r" % s)
        target = self.cache_root / s
        target.parent.mkdir(parents=True, exist_ok=True)
        return self.safe_out(target)


def from_paths(rp) -> Guard:
    return Guard(rp.traj_segs, rp.cache)
