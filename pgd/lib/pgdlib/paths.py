"""Path resolution for a ParGaMD run directory.

The single most important thing in this module is that iteration numbers are
formatted in exactly two places:

    h5_iter_group(n)  -> 'iterations/iter_%08d'   (west.h5 groups: 8 digits)
    disk_iter_dir(n)  -> '%06d'                   (traj_segs dirs:  6 digits)

west.h5 and the on-disk trajectory tree disagree on zero-padding width. Every
other path in the package composes these two functions, so that bug class has
exactly one place to live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEG_DCD_NAME = "output_restart.dcd"
STRIPPED_MARKER = ".stripped"


# --------------------------------------------------------------------------
# The only two iteration formatters in the package
# --------------------------------------------------------------------------
def h5_iter_group(n_iter: int) -> str:
    return "iterations/iter_%08d" % n_iter


def disk_iter_dir(n_iter: int) -> str:
    return "%06d" % n_iter


def disk_seg_dir(seg_id: int) -> str:
    return "%06d" % seg_id


@dataclass(frozen=True)
class RunPaths:
    root: Path
    west_h5: Path
    traj_segs: Path          # realpath, so the /scratch symlink is resolved
    system_dir: Path
    system_name: str
    full_topo: Path
    stripped_topo: Path
    cache: Path
    strip_mask: str

    # -- segment paths -----------------------------------------------------
    def seg_dir(self, n_iter: int, seg_id: int) -> Path:
        return self.traj_segs / disk_iter_dir(n_iter) / disk_seg_dir(seg_id)

    def seg_dcd(self, n_iter: int, seg_id: int) -> Path:
        return self.seg_dir(n_iter, seg_id) / SEG_DCD_NAME

    def seg_gamd_log(self, n_iter: int, seg_id: int) -> Path:
        return self.seg_dir(n_iter, seg_id) / "gamd.log"

    def seg_stripped_marker(self, n_iter: int, seg_id: int) -> Path:
        return self.seg_dir(n_iter, seg_id) / STRIPPED_MARKER

    def iter_dir(self, n_iter: int) -> Path:
        return self.traj_segs / disk_iter_dir(n_iter)

    def iter_strip_lock(self, n_iter: int) -> Path:
        return self.iter_dir(n_iter) / ".stripping.lock"

    def seg_log_tar(self, n_iter: int) -> Path:
        return self.root / "seg_logs" / (disk_iter_dir(n_iter) + ".tar")

    # -- cache paths -------------------------------------------------------
    def cache_seg_dir(self, n_iter: int, seg_id: int) -> Path:
        return (self.cache / "stripped" / disk_iter_dir(n_iter)
                / disk_seg_dir(seg_id))

    def cache_seg_dcd(self, n_iter: int, seg_id: int) -> Path:
        return self.cache_seg_dir(n_iter, seg_id) / SEG_DCD_NAME

    def cache_run_dir(self, runid: str) -> Path:
        return self.cache / "runs" / runid


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v else default


def run_paths(root: str | os.PathLike | None = None) -> RunPaths:
    """Resolve every path the package needs.

    Values come from the environment that ``pgd`` (the dispatcher) already
    established by sourcing the run's ``env.sh``.  Nothing here re-activates
    conda or re-parses env.sh -- if SYSTEM_NAME is missing, the dispatcher
    did not run and that is a bug worth surfacing loudly.
    """
    r = Path(root or _env("WEST_SIM_ROOT") or ".").resolve()
    if not (r / "west.cfg").is_file():
        raise SystemExit(
            "pgd: %s is not a ParGaMD run dir (no west.cfg). "
            "cd to a run dir, or pass --root." % r
        )

    system_name = _env("SYSTEM_NAME")
    if not system_name:
        raise SystemExit(
            "pgd: $SYSTEM_NAME is unset -- env.sh was not sourced. "
            "Invoke through the 'pgd' dispatcher, not this module directly."
        )

    system_dir = Path(_env("PARGAMD_SYSTEM_DIR") or (r / "system"))
    strip_mask = _env("STRIP_MASK", ":WAT,Na+,Cl-,K+")

    traj_segs = Path(os.path.realpath(r / "traj_segs"))

    cache_env = _env("PGD_CACHE")
    if cache_env:
        cache = Path(cache_env)
    else:
        # Sibling of traj_segs, never inside it: same filesystem (so cpptraj
        # I/O and export's hardlink fast path stay local) without any chance
        # of a stray write landing in the live simulation tree.
        cache = traj_segs.parent / "pgd_cache"

    return RunPaths(
        root=r,
        west_h5=r / "west.h5",
        traj_segs=traj_segs,
        system_dir=system_dir,
        system_name=system_name,
        full_topo=system_dir / ("%s.parm7" % system_name),
        stripped_topo=system_dir / ("%s.stripped.parm7" % system_name),
        cache=cache,
        strip_mask=strip_mask,
    )
