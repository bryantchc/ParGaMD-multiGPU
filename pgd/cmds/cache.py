#!/usr/bin/env python3
# summary: inspect, verify and prune pgd's cache
# group: 5 housekeeping
"""pgd cache -- what pgd has stored, and whether it is still valid.

Everything pgd writes lives under one cache root (a sibling of traj_segs, on
the same filesystem, never inside it). Nothing here can touch the run
directory or traj_segs: every removal goes through the same write firewall as
every other write.
"""
import shutil
import sys
import time
from pathlib import Path

from pgdlib import cli
from pgdlib import guard as G
from pgdlib import sizes as Z
from pgdlib import solvent as S


def build_parser():
    p = cli.base_parser(__doc__)
    p.add_argument("action", choices=["ls", "du", "verify", "clean"])
    p.add_argument("--prune", action="store_true",
                   help="verify: delete entries that are stale")
    p.add_argument("--older-than", metavar="DAYS", type=float, default=None,
                   help="clean: only remove entries older than this")
    p.add_argument("--what", choices=["all", "stripped", "runs", "fes", "work"],
                   default="all")
    return p


def walk_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main(argv=None):
    args = build_parser().parse_args(argv)
    rp, h5, max_iter = cli.open_run(args)
    gd = G.from_paths(rp)
    cli.release(h5)          # cache inspection never reads west.h5
    cache = rp.cache

    if not cache.is_dir():
        print("cache does not exist yet: %s" % cache)
        return 0
    print("cache: %s\n" % cache)

    if args.action == "du":
        rows = []
        for sub in ("stripped", "runs", "fes", "work", "index"):
            d = cache / sub
            n = sum(1 for _ in d.rglob("*")) if d.is_dir() else 0
            rows.append((sub, "{:,}".format(n), Z.human(walk_size(d))))
        print(cli.table(rows, ["area", "entries", "size"],
                        aligns=["<", ">", ">"]))
        print("\ntotal: %s   (%s free on this filesystem)"
              % (Z.human(walk_size(cache)), Z.human(shutil.disk_usage(cache).free)))
        return 0

    if args.action == "ls":
        sd = cache / "stripped"
        if sd.is_dir() and args.what in ("all", "stripped"):
            rows = []
            for meta in sorted(sd.rglob("meta.json")):
                it = int(meta.parent.parent.name)
                sg = int(meta.parent.name)
                rows.append((it, "%06d" % sg,
                             Z.human((meta.parent / "output_restart.dcd").stat().st_size)
                             if (meta.parent / "output_restart.dcd").is_file() else "-",
                             "valid" if S.cached_strip_is_valid(rp, it, sg)
                             else "STALE"))
            print("stripped copies:")
            print(cli.table(rows, ["iter", "seg", "size", "state"],
                            aligns=[">", ">", ">", "<"]) if rows else "  (none)")
            print()
        rd = cache / "runs"
        if rd.is_dir() and args.what in ("all", "runs"):
            rows = [(d.name, Z.human(walk_size(d)),
                     time.strftime("%Y-%m-%d %H:%M",
                                   time.localtime(d.stat().st_mtime)))
                    for d in sorted(rd.iterdir()) if d.is_dir()]
            print("load/export decks:")
            print(cli.table(rows, ["runid", "size", "modified"],
                            aligns=["<", ">", "<"]) if rows else "  (none)")
            print()
        fd = cache / "fes"
        if fd.is_dir() and args.what in ("all", "fes"):
            rows = [(d.name, Z.human(walk_size(d)),
                     time.strftime("%Y-%m-%d %H:%M",
                                   time.localtime(d.stat().st_mtime)))
                    for d in sorted(fd.iterdir()) if d.is_dir()]
            print("free-energy surfaces:")
            print(cli.table(rows, ["name", "size", "modified"],
                            aligns=["<", ">", "<"]) if rows else "  (none)")
        return 0

    if args.action == "verify":
        sd = cache / "stripped"
        stale = []
        n = 0
        for meta in sorted(sd.rglob("meta.json")) if sd.is_dir() else []:
            it = int(meta.parent.parent.name)
            sg = int(meta.parent.name)
            n += 1
            if not S.cached_strip_is_valid(rp, it, sg):
                stale.append((it, sg, meta.parent))
        print("%d stripped entr%s checked, %d stale"
              % (n, "y" if n == 1 else "ies", len(stale)))
        for it, sg, d in stale:
            print("  STALE %d:%06d  %s" % (it, sg, d))
        if stale and args.prune:
            for _, _, d in stale:
                gd.safe_out(d / ".x")        # firewall check on the parent
                shutil.rmtree(d)
            print("pruned %d stale entr%s"
                  % (len(stale), "y" if len(stale) == 1 else "ies"))
        elif stale:
            print("\nre-run with --prune to remove them "
                  "(originals in traj_segs are never touched)")
        return 0

    if args.action == "clean":
        targets = []
        for sub in (["stripped", "runs", "fes", "work"]
                    if args.what == "all" else [args.what]):
            d = cache / sub
            if not d.is_dir():
                continue
            for e in d.iterdir():
                if args.older_than is not None:
                    age_days = (time.time() - e.stat().st_mtime) / 86400.0
                    if age_days < args.older_than:
                        continue
                targets.append(e)
        if not targets:
            print("nothing to clean")
            return 0
        total = sum(walk_size(t) if t.is_dir() else t.stat().st_size
                    for t in targets)
        print("would remove %d entr%s (%s) from %s"
              % (len(targets), "y" if len(targets) == 1 else "ies",
                 Z.human(total), cache))
        for t in targets[:20]:
            print("  %s" % t)
        if len(targets) > 20:
            print("  ... and %d more" % (len(targets) - 20))
        if cli.dry_run():
            return 0
        if not cli.assume_yes():
            try:
                tty = open("/dev/tty", "r+")
                tty.write("Proceed? [y/N] ")
                tty.flush()
                if not (tty.readline() or "").strip().lower().startswith("y"):
                    print("aborted")
                    return 4
            except OSError:
                print("no tty; pass --yes", file=sys.stderr)
                return 4
        for t in targets:
            gd.safe_out(t / ".x" if t.is_dir() else t)
            shutil.rmtree(t) if t.is_dir() else t.unlink()
        print("removed %d entries" % len(targets))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except G.UnsafeWrite as e:
        cli.eprint("[pgd] %s" % e)
        sys.exit(1)
