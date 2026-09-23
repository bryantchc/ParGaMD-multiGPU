#!/usr/bin/env python3
# summary: check the run, the tools, the cache and the solvent inventory
# group: 1 inspect the run
"""pgd doctor -- is this run ready for analysis, and what is in it?

Read-only except for --fix-cache. Run it first in any new run directory.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pgdlib import cli
from pgdlib import dcdio
from pgdlib import lineage as L
from pgdlib import paths as P
from pgdlib import sizes as Z
from pgdlib import solvent as S

OK, WARN, BAD = "ok  ", "warn", "FAIL"
_worst = [0]


def say(level, label, detail=""):
    _worst[0] = max(_worst[0], {OK: 0, WARN: 1, BAD: 2}[level])
    print("[%s] %-28s %s" % (level, label, detail))


def build_parser():
    p = cli.base_parser(__doc__)
    p.add_argument("--fix-cache", action="store_true",
                   help="create the cache tree (the only write this makes)")
    p.add_argument("--scan-solvent", action="store_true",
                   help="inventory every iteration's solvent state (stats ~115 "
                        "dirs; a few seconds)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rp, h5, max_iter = cli.open_run(args)

    print("run:   %s" % rp.root)
    print("system: %s\n" % rp.system_name)

    # ---- environment -----------------------------------------------------
    say(OK, "conda env", os.environ.get("CONDA_PREFIX", "?"))
    for mod in ("h5py", "MDAnalysis", "numpy", "westpa"):
        try:
            m = __import__(mod)
            say(OK, "python: %s" % mod, getattr(m, "__version__", "?"))
        except ImportError:
            say(BAD if mod != "westpa" else WARN, "python: %s" % mod, "MISSING")

    for label, path in (("cpptraj", os.environ.get("PGD_CPPTRAJ", "")),
                        ("vmd", os.environ.get("PGD_VMD", "")),
                        ("catdcd", os.environ.get("PGD_CATDCD", ""))):
        if path and Path(path).exists() and os.access(path, os.X_OK):
            say(OK, "tool: %s" % label, path)
        else:
            say(WARN if label == "catdcd" else BAD, "tool: %s" % label,
                "not executable at %r" % path)

    # ---- data ------------------------------------------------------------
    say(OK, "west.h5", "%s (%s)" % (rp.west_h5, Z.human(rp.west_h5.stat().st_size)))
    cur = int(h5.attrs.get("west_current_iteration", -1))
    say(OK, "iterations", "%d complete; west_current_iteration=%d%s"
        % (max_iter, cur,
           " (in flight, excluded)" if cur > max_iter else ""))

    if rp.traj_segs.is_dir():
        du = shutil.disk_usage(rp.traj_segs)
        say(OK, "traj_segs", "%s (%s free on that filesystem)"
            % (rp.traj_segs, Z.human(du.free)))
    else:
        say(BAD, "traj_segs", "missing: %s" % rp.traj_segs)

    for label, topo in (("topology: full", rp.full_topo),
                        ("topology: stripped", rp.stripped_topo)):
        if topo.is_file():
            try:
                say(OK, label, "%s (%s atoms)"
                    % (topo.name, "{:,}".format(S.topo_natoms(str(topo)))))
            except Exception as e:                          # noqa: BLE001
                say(BAD, label, "unreadable: %s" % e)
        else:
            say(BAD, label, "missing: %s" % topo)

    say(OK, "strip mask", rp.strip_mask)

    # ---- cache -----------------------------------------------------------
    if rp.cache.is_dir():
        du = shutil.disk_usage(rp.cache)
        say(OK, "cache", "%s (%s free)" % (rp.cache, Z.human(du.free)))
    elif args.fix_cache:
        rp.cache.mkdir(parents=True, exist_ok=True)
        (rp.cache / "stripped").mkdir(exist_ok=True)
        (rp.cache / "runs").mkdir(exist_ok=True)
        say(OK, "cache", "created %s" % rp.cache)
    else:
        say(WARN, "cache", "%s does not exist yet (pgd creates it on demand, "
                           "or run --fix-cache)" % rp.cache)

    # firewall
    try:
        from pgdlib import guard as G
        gd = G.from_paths(rp)
        try:
            gd.safe_out(rp.seg_dcd(max_iter, 0))
            say(BAD, "write firewall", "ACCEPTED a traj_segs path -- do not run "
                                       "strip-copy until this is fixed")
        except G.UnsafeWrite:
            say(OK, "write firewall", "traj_segs is unreachable from writes")
    except Exception as e:                                  # noqa: BLE001
        say(BAD, "write firewall", str(e))

    # ---- solvent inventory ----------------------------------------------
    if args.scan_solvent:
        full_iters, stripped_iters, missing = [], [], []
        for it in range(1, max_iter + 1):
            d = rp.iter_dir(it)
            if not d.is_dir():
                missing.append(it)
                continue
            segs = sorted(p for p in d.iterdir() if p.is_dir() and p.name.isdigit())
            if not segs:
                missing.append(it)
                continue
            n_marked = sum(1 for p in segs if (p / ".stripped").is_file())
            if n_marked == len(segs):
                stripped_iters.append(it)
            elif n_marked == 0:
                full_iters.append((it, len(segs)))
            else:
                say(WARN, "iter %d" % it,
                    "partially stripped: %d/%d segments" % (n_marked, len(segs)))
        say(OK, "stripped iterations", "%d" % len(stripped_iters))
        if full_iters:
            tot = sum(n for _, n in full_iters)
            cached = 0
            for it, _ in full_iters:
                d = rp.iter_dir(it)
                for p in d.iterdir():
                    if p.is_dir() and p.name.isdigit() and \
                            S.cached_strip_is_valid(rp, it, int(p.name)):
                        cached += 1
            say(OK, "full-atom iterations",
                "%s (%s segments; %d have stripped copies in the cache)"
                % (", ".join("%d(%d segs)" % x for x in full_iters), tot, cached))
            say(OK, "", "a lineage crossing these needs: "
                        "pgd strip-copy --for-lineage ITER:SEG")
        if missing:
            say(WARN, "iterations without segments", str(missing))

    # ---- bin mappers -----------------------------------------------------
    try:
        changes = []
        prev = None
        for it in range(1, max_iter + 1):
            hh = h5[P.h5_iter_group(it)].attrs.get("binhash")
            if hh != prev:
                changes.append(it)
                prev = hh
        say(OK if len(changes) <= 1 else WARN, "bin mappers",
            "%d distinct; changed at iterations %s"
            % (len(changes), changes if len(changes) < 12
               else "%s ... %s" % (changes[:6], changes[-3:])))
        if len(changes) > 1:
            say(WARN, "", "bin indices are NOT comparable across iterations; "
                          "pgd find searches raw CV values instead")
    except Exception as e:                                  # noqa: BLE001
        say(WARN, "bin mappers", str(e))

    # ---- who else holds west.h5 -----------------------------------------
    # An inherited handle in a long-lived child (a VMD session) keeps the HDF5
    # lock and makes `w_run` fail with BlockingIOError. pgd releases its own
    # handle before spawning anything, but say plainly what is holding it.
    cli.release(h5)
    if shutil.which("lsof"):
        try:
            r = subprocess.run(["lsof", "--", str(rp.west_h5)],
                               capture_output=True, text=True, timeout=20)
            holders = [l for l in r.stdout.splitlines()[1:] if l.strip()]
        except (OSError, subprocess.SubprocessError):
            holders = None
        if holders is None:
            say(WARN, "west.h5 lock", "could not run lsof")
        elif holders:
            say(WARN, "west.h5 lock",
                "%d process(es) hold it -- w_run may fail with BlockingIOError"
                % len(holders))
            for h in holders[:5]:
                say(WARN, "", h.split()[0] + " pid " + h.split()[1])
        else:
            say(OK, "west.h5 lock", "no other process holds it")
    else:
        say(OK, "west.h5 lock", "lsof unavailable; skipped")

    # ---- memory ----------------------------------------------------------
    avail = Z.available_ram_bytes()
    warn_gb = float(os.environ.get("PGD_LOAD_WARN_GB", Z.DEFAULT_WARN_GB))
    say(OK if avail else WARN, "memory",
        "%s available; load gate at %s or %d%% of that"
        % (Z.human(avail), Z.human(warn_gb * 2 ** 30),
           round(100 * Z.DEFAULT_RAM_FRACTION)))

    print()
    if _worst[0] == 0:
        print("all checks passed")
    elif _worst[0] == 1:
        print("passed with warnings")
    else:
        print("FAILURES above -- fix them before relying on pgd")
    return _worst[0] if _worst[0] > 1 else 0


if __name__ == "__main__":
    sys.exit(main())
