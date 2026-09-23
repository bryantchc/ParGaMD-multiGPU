#!/usr/bin/env python3
# summary: strip solvent from COPIES of full-atom segments into the cache
# group: 5 housekeeping
"""pgd strip-copy -- normalize full-atom segments without touching originals.

Iterations 58, 114 and 115 are still full-atom; every lineage crosses at least
one of them. To stitch, they must be reduced to the same 24,790-atom set as
the rest. The running simulation depends on those originals staying intact, so
this command strips COPIES into a cache and never modifies traj_segs.

Why not reuse westpa_scripts/strip_iter.sh
-----------------------------------------
Its contract is exactly what we must forbid: strip_segment() ends with
`mv -f ... output_restart.dcd; touch .stripped`, i.e. in-place replacement.
It is also iteration-scoped (`--iter 58` would strip all 1,368 segments,
~385 GB, when a lineage needs one), and it writes lock/marker files into the
live simulation tree. It is a symlink into the shared repo that the live
post_iter.sh hook invokes, so editing it would risk the running simulation.

We reuse its cpptraj *recipe* -- the same parm/trajin/strip/trajout/go/quit
deck, with STRIP_MASK read from env.sh, never a literal -- with the
destination as an explicit argument.

Why in-place writes are structurally impossible here
----------------------------------------------------
1. The source is COPIED into a private work dir first, so no traj_segs path
   ever appears in the cpptraj deck at all -- not even as `trajin`.
2. Every write goes through pgdlib.guard, which realpath-resolves both sides
   and refuses anything under traj_segs (defeating the run-dir symlink and
   any `..` traversal). `--selftest` proves it.
3. Locks and markers live in the cache, never in traj_segs.
4. A pre/post stat tripwire on every source proves nothing changed, catching
   even an accident from outside this tool.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pgdlib import cli
from pgdlib import dcdio
from pgdlib import guard as G
from pgdlib import lineage as L
from pgdlib import paths as P
from pgdlib import sizes as Z
from pgdlib import solvent as S

STALE_LOCK_SECONDS = 900


def build_parser():
    p = cli.base_parser(__doc__)
    g = p.add_argument_group("what to strip")
    g.add_argument("--for-lineage", metavar="ITER:SEG",
                   help="strip only the full-atom segments on this lineage "
                        "(typically 1-3 segments, seconds)")
    g.add_argument("--segments", metavar="I:S,I:S",
                   help="strip an explicit list of ITER:SEG pairs")
    g.add_argument("--iter", type=int,
                   help="bulk: strip every full-atom segment of an iteration")
    p.add_argument("--jobs", type=int, default=8,
                   help="parallel cpptraj processes (default 8; I/O-bound, and "
                        "the live simulation needs the machine too)")
    p.add_argument("--force", action="store_true",
                   help="re-strip even if a valid cached copy exists")
    p.add_argument("--no-paranoid-copy", action="store_true",
                   help="let cpptraj read the original directly instead of a "
                        "private copy. Faster; drops safety layer 1.")
    p.add_argument("--allow-live-iter", action="store_true",
                   help="permit iterations within 2 of the last complete one")
    p.add_argument("--lock-timeout", type=float, default=300.0)
    p.add_argument("--selftest", action="store_true",
                   help="prove the write firewall rejects traj_segs paths")
    p.add_argument("--verify", action="store_true",
                   help="re-check every cached entry and report staleness")
    return p


# --------------------------------------------------------------------------
def selftest(rp) -> int:
    """Assert the guard refuses every route into traj_segs. Exit non-zero if not."""
    gd = G.from_paths(rp)
    victims = [
        rp.seg_dcd(115, 442),
        rp.root / "traj_segs" / "000115" / "000442" / "output_restart.dcd",
        rp.cache / ".." / "traj_segs" / "000058" / "x.dcd",
        rp.traj_segs / "Ultra_RL2.stripped.parm7",
        rp.iter_dir(58) / "000262" / ".stripped",
    ]
    bad = []
    for v in victims:
        try:
            gd.safe_out(v)
            bad.append(str(v))
        except G.UnsafeWrite:
            print("  refused  %s" % v)
    for rel in ("/etc/passwd", "../traj_segs/x.dcd", "a/../../x"):
        try:
            gd.cache_write(rel)
            bad.append("cache_write(%r)" % rel)
        except G.UnsafeWrite:
            print("  refused  cache_write(%r)" % rel)
    try:
        gd.allow_explicit(rp.seg_dcd(58, 262))
        bad.append("allow_explicit into traj_segs")
    except G.UnsafeWrite:
        print("  refused  allow_explicit(traj_segs path)")

    ok = gd.cache_write("selftest/probe.txt")
    print("  allowed  %s" % ok)

    if bad:
        print("\nSELFTEST FAILED -- the firewall ACCEPTED:", file=sys.stderr)
        for b in bad:
            print("  %s" % b, file=sys.stderr)
        return 1
    print("\nselftest passed: traj_segs is unreachable from every write path")
    return 0


# --------------------------------------------------------------------------
def _stat_tuple(p: Path):
    st = p.stat()
    return st.st_size, st.st_mtime_ns


def _acquire(lock: Path, timeout: float) -> int:
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, json.dumps({"pid": os.getpid(),
                                     "host": socket.gethostname(),
                                     "t": time.time()}).encode())
            return fd
        except FileExistsError:
            try:
                info = json.loads(lock.read_text())
                same_host = info.get("host") == socket.gethostname()
                age = time.time() - float(info.get("t", 0))
                dead = False
                if same_host:
                    try:
                        os.kill(int(info["pid"]), 0)
                    except ProcessLookupError:
                        dead = True
                    except (OSError, ValueError, KeyError):
                        pass
                if (dead and age > 60) or age > STALE_LOCK_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                lock.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise TimeoutError("lock held: %s" % lock)
            time.sleep(1.0)


def strip_one(job) -> dict:
    """Strip one segment into the cache. Runs in a worker process."""
    (root, n_iter, seg_id, paranoid, lock_timeout) = job
    os.environ["OMP_NUM_THREADS"] = "1"      # cpptraj is OpenMP-enabled
    rp = P.run_paths(root)
    gd = G.from_paths(rp)

    src = rp.seg_dcd(n_iter, seg_id)
    before = _stat_tuple(src)

    out_dir = gd.cache_write("stripped/%s/%s/.keep"
                             % (P.disk_iter_dir(n_iter),
                                P.disk_seg_dir(seg_id))).parent
    out = gd.safe_out(out_dir / P.SEG_DCD_NAME)
    meta_p = gd.safe_out(out_dir / "meta.json")
    lock = out_dir / ".lock"

    fd = _acquire(lock, lock_timeout)
    work = None
    try:
        work = Path(tempfile.mkdtemp(prefix="strip-", dir=str(
            gd.cache_write("work/.keep").parent)))

        if paranoid:
            # Layer 1: copy first, so no traj_segs path appears in the deck.
            trajin_src = work / "input.dcd"
            shutil.copyfile(src, trajin_src)
        else:
            trajin_src = src

        tmp_out = work / "stripped.dcd"
        deck = work / "strip.cpptraj"
        deck.write_text(
            "parm %s\n"
            "trajin %s\n"
            "strip %s\n"
            "trajout %s dcd\n"
            "go\n"
            "quit\n" % (rp.full_topo, trajin_src, rp.strip_mask, tmp_out))

        log = work / "cpptraj.out"
        cp = os.environ.get("PGD_CPPTRAJ", "cpptraj")
        with open(log, "wb") as lf:
            rc = subprocess.call([cp, "-i", str(deck)], stdout=lf,
                                 stderr=subprocess.STDOUT)
        if rc != 0 or not tmp_out.is_file() or tmp_out.stat().st_size == 0:
            tail = log.read_text(errors="replace")[-2000:]
            raise RuntimeError("cpptraj failed for %d:%d (rc=%d)\n%s"
                               % (n_iter, seg_id, rc, tail))

        prof = dcdio.probe(tmp_out)
        want_atoms = S.topo_natoms(str(rp.stripped_topo))
        src_prof = dcdio.probe(src)
        if prof.natoms != want_atoms:
            raise RuntimeError(
                "%d:%d stripped to %d atoms, expected %d (mask %r)"
                % (n_iter, seg_id, prof.natoms, want_atoms, rp.strip_mask))
        if prof.nframes != src_prof.nframes:
            raise RuntimeError(
                "%d:%d frame count changed: %d -> %d"
                % (n_iter, seg_id, src_prof.nframes, prof.nframes))

        os.replace(tmp_out, out)              # same filesystem: atomic
        meta_p.write_text(json.dumps({
            "schema": 1,
            "src": str(src),
            "src_size": before[0],
            "src_mtime_ns": before[1],
            "src_natoms": src_prof.natoms,
            "src_nframes": src_prof.nframes,
            "out_natoms": prof.natoms,
            "out_nframes": prof.nframes,
            "out_size": out.stat().st_size,
            "strip_mask": rp.strip_mask,
            "topology": str(rp.full_topo),
            "pgd_version": os.environ.get("PGD_VERSION", "?"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))

        after = _stat_tuple(src)
        if after != before:
            raise RuntimeError(
                "TRIPWIRE: source %s changed during the strip (%s -> %s). "
                "Something outside pgd is writing to traj_segs."
                % (src, before, after))

        return {"iter": n_iter, "seg": seg_id, "ok": True,
                "out": str(out), "bytes": out.stat().st_size}
    except Exception as e:                                  # noqa: BLE001
        # Never leave a partial cache entry that a later run would trust.
        try:
            out.unlink(missing_ok=True)
            meta_p.unlink(missing_ok=True)
        except OSError:
            pass
        return {"iter": n_iter, "seg": seg_id, "ok": False, "error": str(e)}
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)
        os.close(fd)
        lock.unlink(missing_ok=True)


# --------------------------------------------------------------------------
def parse_pairs(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            i, s = tok.split(":")
            out.append((int(i), int(s)))
        except ValueError:
            raise SystemExit("pgd: --segments wants ITER:SEG pairs, got %r" % tok)
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    rp, h5, max_iter = cli.open_run(args)

    if args.selftest:
        cli.release(h5)
        return selftest(rp)

    # ---- work out which segments need stripping --------------------------
    if args.for_lineage:
        try:
            ti, ts = (int(x) for x in args.for_lineage.split(":"))
        except ValueError:
            raise SystemExit("pgd: --for-lineage wants ITER:SEG")
        chain = L.walk_lineage(h5, ti, ts)
        candidates = [(nd.n_iter, nd.seg_id) for nd in chain]
    elif args.segments:
        candidates = parse_pairs(args.segments)
    elif args.iter is not None:
        d = rp.iter_dir(args.iter)
        if not d.is_dir():
            raise SystemExit("pgd: no on-disk directory for iteration %d" % args.iter)
        candidates = sorted((args.iter, int(p.name)) for p in d.iterdir()
                            if p.is_dir() and p.name.isdigit())
    else:
        raise SystemExit(
            "pgd: pick a scope -- --for-lineage ITER:SEG (usual), "
            "--segments I:S,... or --iter N (bulk)")

    if args.verify:
        rows = []
        for it, sg in candidates:
            if S.classify(rp, it, sg) is not S.SolventState.FULL:
                continue
            rows.append((it, "%06d" % sg,
                         "valid" if S.cached_strip_is_valid(rp, it, sg) else "absent/stale"))
        print(cli.table(rows, ["iter", "seg", "cache"]) if rows
              else "no full-atom segments in scope")
        return 0

    todo = []
    for it, sg in candidates:
        st = S.classify(rp, it, sg)
        if st is not S.SolventState.FULL:
            continue
        if not args.force and S.cached_strip_is_valid(rp, it, sg):
            continue
        if it > max_iter:
            raise SystemExit("pgd: iteration %d is not complete" % it)
        if not args.allow_live_iter and it > max_iter - 2:
            raise SystemExit(
                "pgd: iteration %d is within 2 of the last complete iteration "
                "(%d) and may still be referenced by the running simulation's\n"
                "     freeze-fallback. Reading it is safe once the segment is "
                "complete; pass --allow-live-iter to proceed." % (it, max_iter))
        if rp.iter_strip_lock(it).is_file():
            raise SystemExit(
                "pgd: iteration %d has a live .stripping.lock -- "
                "westpa_scripts/strip_iter.sh is working on it. Try later." % it)
        todo.append((it, sg))

    if not todo:
        print("nothing to do: every full-atom segment in scope already has a "
              "valid stripped copy")
        return 0

    full_atoms = S.topo_natoms(str(rp.full_topo))
    read_b = sum(rp.seg_dcd(i, s).stat().st_size for i, s in todo)
    write_b = len(todo) * dcdio.dcd_bytes(S.topo_natoms(str(rp.stripped_topo)), 100)
    print("%d segment(s) to strip: %s read, %s written to %s"
          % (len(todo), Z.human(read_b), Z.human(write_b), rp.cache / "stripped"))
    print("originals in %s are never modified." % rp.traj_segs)

    if cli.dry_run():
        for it, sg in todo:
            print("  would strip %d:%06d -> %s" % (it, sg, rp.cache_seg_dcd(it, sg)))
        return 0

    if len(todo) > 16 and not cli.assume_yes():
        if not _confirm("Proceed?"):
            return 4

    # Release before forking: workers would otherwise inherit the handle and
    # each hold the HDF5 lock for as long as they run.
    cli.release(h5)

    os.nice(5)
    jobs = [(str(rp.root), i, s, not args.no_paranoid_copy, args.lock_timeout)
            for i, s in todo]
    t0 = time.time()
    results = []
    if args.jobs <= 1 or len(jobs) == 1:
        for j in jobs:
            results.append(strip_one(j))
            _report(results[-1])
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as ex:
            futs = [ex.submit(strip_one, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())
                _report(results[-1])

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print("\n%d stripped, %d failed in %.1fs" % (len(ok), len(bad), time.time() - t0))
    return 1 if bad else 0


def _report(r):
    if r["ok"]:
        print("  ok   %d:%06d -> %s" % (r["iter"], r["seg"], r["out"]))
    else:
        print("  FAIL %d:%06d %s" % (r["iter"], r["seg"], r["error"]),
              file=sys.stderr)


def _confirm(prompt):
    try:
        f = open("/dev/tty", "r+")
    except OSError:
        print("no tty; pass --yes to proceed non-interactively", file=sys.stderr)
        return False
    f.write("%s [y/N] " % prompt)
    f.flush()
    return f.readline().strip().lower().startswith("y")


if __name__ == "__main__":
    sys.exit(main())
