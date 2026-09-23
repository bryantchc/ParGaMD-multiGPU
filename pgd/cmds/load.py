#!/usr/bin/env python3
# summary: load a lineage's continuous trajectory into vmd_2
# group: 3 view it
"""pgd load -- open the continuous trajectory from a walker back to its root.

Writes no trajectory file. Emits a Tcl deck and hands it to vmd_2, which
concatenates the ~115 segment DCDs in memory. There is no multi-GB temporary
file to wait for, nothing to invalidate, and nothing to garbage-collect.
`pgd export` is the command that materializes bytes.

Full-atom iterations (58, 114, 115 here) are served from the strip-copy cache
so the whole lineage has one atom count. VMD's atom-count mismatch is SILENT
-- it prints to stderr and loads nothing -- so the generated deck asserts the
frame count after every addfile, and a preflight runs headless first.
"""
import os
import subprocess
import sys
from pathlib import Path

from pgdlib import cli
from pgdlib import frames as F
from pgdlib import guard as G
from pgdlib import lineage as L
from pgdlib import paths as P
from pgdlib import sizes as Z
from pgdlib import solvent as S
from pgdlib import vmdgen


def build_parser():
    p = cli.base_parser(__doc__)
    cli.add_tip_args(p)
    cli.add_slice_args(p)
    p.add_argument("--colvars", action="store_true",
                   help="also attach the run's own progress coordinates as VMD "
                        "Colvars, so cv0/cv1 are live while you scrub the "
                        "trajectory (see: pgd colvars)")
    p.add_argument("--colvars-config", metavar="FILE",
                   help="use this Colvars config instead of generating one")
    p.add_argument("--reps", metavar="FILE",
                   help="Tcl file of representations to source instead of the "
                        "built-in protein/RNA/DNA default")
    p.add_argument("--text", action="store_true",
                   help="run VMD headless (-dispdev text) instead of the GUI")
    p.add_argument("--script-only", action="store_true",
                   help="write the deck and print its path; don't launch VMD")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the headless assertion pass before launching")
    p.add_argument("--no-provenance", action="store_true",
                   help="omit the frame->segment lookup table from the deck")
    p.add_argument("--background", action="store_true",
                   help="launch VMD detached instead of handing it the terminal")
    p.add_argument("--warn-gb", type=float, default=None,
                   help="size-gate threshold in GiB (default 95, "
                        "or $PGD_LOAD_WARN_GB)")
    p.add_argument("--ram-fraction", type=float, default=Z.DEFAULT_RAM_FRACTION,
                   help="fraction of MemAvailable VMD may use (default 0.5)")
    return p


def prompt_size(g, args):
    """Interactive size gate. Returns the stride to use, or exits."""
    sys.stderr.write(Z.render(g) + "\n")
    if not g["over"]:
        return args.stride

    sug = g["suggested_stride"] * args.stride
    msg = ("This exceeds the limits above. Suggested --stride %d -> %s frames, %s in VMD.\n"
           % (sug, "{:,}".format(g["nframes"] // g["suggested_stride"]),
              Z.human(g["est_ram"] // g["suggested_stride"])))
    sys.stderr.write(msg)

    if cli.assume_yes():
        sys.stderr.write("[pgd] --yes given: continuing at stride %d\n" % args.stride)
        return args.stride

    tty = None
    if sys.stdin.isatty():
        tty = (sys.stdin, sys.stderr)
    else:
        try:
            f = open("/dev/tty", "r+")
            tty = (f, f)
        except OSError:
            pass
    if tty is None:
        sys.stderr.write(
            "Refusing to proceed non-interactively. Re-run with --stride %d "
            "(fits), --yes to force, or --warn-gb N to raise the limit.\n" % sug)
        raise SystemExit(3)

    tty[1].write("  [s] stride %d (recommended)  [c] continue anyway  "
                 "[a] abort  [N] enter a stride > " % sug)
    tty[1].flush()
    ans = (tty[0].readline() or "").strip().lower()
    if ans in ("s", ""):
        return sug
    if ans == "c":
        return args.stride
    if ans == "a":
        raise SystemExit(4)
    try:
        return max(1, int(ans))
    except ValueError:
        raise SystemExit(4)


def main(argv=None):
    args = build_parser().parse_args(argv)
    tip_iter, tip_seg = cli.resolve_tip(args)
    rp, h5, max_iter = cli.open_run(args)
    gd = G.from_paths(rp)

    if tip_iter > max_iter:
        raise SystemExit("pgd: iteration %d is not complete (last complete: %d)"
                         % (tip_iter, max_iter))

    chain = L.walk_lineage(h5, tip_iter, tip_seg, stop_iter=args.stop_iter)
    seam, seam_ev = cli.resolve_seam(h5, chain, args.seam)
    # If we are going to show colvars, grab what WESTPA recorded for these
    # segments now -- while h5 is still open -- so the viewer can compare the
    # live Colvars value against the pcoord of record.
    recorded = {}
    if args.colvars:
        import numpy as _np
        for nd in chain:
            recorded[(nd.n_iter, nd.seg_id)] = _np.asarray(
                h5[P.h5_iter_group(nd.n_iter) + "/pcoord"][nd.seg_id])

    # Last use of west.h5. Close it NOW: everything below (solvent resolution,
    # size gate, deck generation, preflight, execvp) needs nothing from it, and
    # HDF5 has no FD_CLOEXEC -- a handle still open at exec is inherited by VMD
    # and locks the file against w_run for the whole session.
    cli.release(h5)
    if seam_ev.get("mode") == "auto":
        cli.eprint("[pgd] seam: %s (%.0f%% of %d links are exact duplicates)"
                   % (seam, 100 * seam_ev["frac_exact"], seam_ev["n_links"]))

    want = {"stripped": S.SolventState.STRIPPED,
            "full": S.SolventState.FULL,
            "auto": S.SolventState.UNKNOWN}[args.solvent]
    topology, resolved = S.resolve_chain(rp, chain, want=want,
                                         tip=(tip_iter, tip_seg))
    natoms = S.topo_natoms(str(topology))

    # Size gate before anything expensive, and it may change the stride.
    probe_slices = F.segment_slices(chain, stride=args.stride, seam=seam,
                                    tip_last_frame=args.tip_frame)
    g = Z.gate(F.total_frames(probe_slices), natoms,
               warn_gb=args.warn_gb, ram_fraction=args.ram_fraction)
    stride = prompt_size(g, args)

    slices = F.segment_slices(chain, stride=stride, seam=seam,
                              tip_last_frame=args.tip_frame)
    nframes = F.total_frames(slices)

    by_key = {(nd.n_iter, nd.seg_id): path for nd, (path, _) in zip(chain, resolved)}
    resolved_slices = [(by_key[(s.n_iter, s.seg_id)], s) for s in slices]

    provenance = None
    if not args.no_provenance:
        provenance = []
        for s in slices:
            for k, lf in enumerate(range(s.first, s.last + 1, s.step)):
                provenance.append((s.n_iter, s.seg_id, lf,
                                   s.t0_ps + k * s.step * F.FRAME_PS))

    title = ("pgd lineage iter%d/seg%06d -> %s  (%s frames, %.2f ns, stride %d)"
             % (tip_iter, tip_seg,
                "root" if chain[0].is_root else "iter%d" % chain[0].n_iter,
                "{:,}".format(nframes), nframes * F.FRAME_PS * stride / 1000.0,
                stride))

    runid = "%d_%06d_%s_s%d_%s" % (tip_iter, tip_seg, seam, stride,
                                   args.solvent)
    deck = gd.cache_write("runs/%s/load.tcl" % runid)
    pre = gd.cache_write("runs/%s/preflight.tcl" % runid)

    if cli.dry_run():
        print("would write %s" % deck)
        print("would load %d segments, %s frames, %d atoms from %s"
              % (len(slices), "{:,}".format(nframes), natoms, topology))
        return 0

    # --- Colvars: the run's own progress coordinates, live in the viewer ---
    extra_tcl = None
    if args.colvars:
        from pgdlib import colvars as CV
        import numpy as _np
        try:
            if args.colvars_config:
                cfg = Path(args.colvars_config).expanduser()
                if not cfg.is_file():
                    raise SystemExit("pgd: no colvars config at %s" % cfg)
                labels = ["(from %s)" % cfg.name]
            else:
                cfg = gd.cache_write("colvars/%s_%s.colvars"
                                     % (rp.system_name, args.solvent))
                cwd = os.getcwd()
                os.chdir(rp.root)      # cv modules resolve relative subset paths
                try:
                    mods = CV.discover_cv_modules(rp.root)
                    import warnings
                    import MDAnalysis as mda
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        uref = mda.Universe(str(topology),
                                            str(resolved_slices[0][0]))
                    specs = [CV.extract(n, pth, m, uref) for n, pth, m in mods]
                finally:
                    os.chdir(cwd)
                cfg.write_text(CV.build_config(specs, topology, rp.root))
                labels = [sp.label for sp in specs]
                cli.eprint("[pgd] colvars: %s -> %s"
                           % (", ".join("%s (%d pairs)" % (sp.label, sp.n)
                                        for sp in specs), cfg))

            # What WESTPA recorded, per emitted frame -- so the live Colvars
            # value can be checked against the pcoord of record.
            ref = gd.cache_write("runs/%s/pcoord_recorded.dat" % runid)
            with open(ref, "w") as fh:
                fh.write("# frame  iter  seg  seg_frame  "
                         "pcoord as recorded in west.h5\n")
                row = 0
                for sl in slices:
                    pc = recorded.get((sl.n_iter, sl.seg_id))
                    for lf in range(sl.first, sl.last + 1, sl.step):
                        vals = ("  ".join("%.6f" % v for v in pc[lf])
                                if pc is not None else "")
                        fh.write("%d  %d  %d  %d  %s\n"
                                 % (row, sl.n_iter, sl.seg_id, lf, vals))
                        row += 1
            cli.eprint("[pgd] west.h5 pcoord of record -> %s" % ref)
            extra_tcl = CV.tcl_snippet(cfg, labels, ref_trace=ref)
        except CV.ColvarsUntranslatable as e:
            cli.eprint("[pgd] colvars: %s" % e)
            cli.eprint("[pgd] continuing without colvars")

    vmdgen.write_tcl(deck, topology, resolved_slices, natoms, nframes,
                     title, reps_file=args.reps, provenance=provenance,
                     extra_tcl=extra_tcl)
    vmdgen.write_preflight_tcl(pre, topology, resolved_slices, natoms, nframes)

    cli.eprint("[pgd] %d segments, %s frames, %.2f ns, %d atoms"
               % (len(slices), "{:,}".format(nframes),
                  nframes * F.FRAME_PS * stride / 1000.0, natoms))
    cli.eprint("[pgd] deck: %s" % deck)

    if args.script_only:
        print(deck)
        return 0

    vmd = os.environ.get("PGD_VMD", "/usr/local/bin/vmd_2")
    if not Path(vmd).exists():
        raise SystemExit("pgd: vmd not found at %s (set PGD_VMD)" % vmd)

    if not args.no_preflight:
        cli.eprint("[pgd] preflight (headless)...")
        r = subprocess.run([vmd, "-dispdev", "text", "-eofexit", "-e", str(pre)],
                           capture_output=True, text=True)
        got = [l for l in r.stdout.splitlines() if l.startswith("PGD_")]
        if r.returncode != 0 or not got:
            sys.stderr.write(r.stdout[-3000:])
            sys.stderr.write(r.stderr[-3000:])
            raise SystemExit("pgd: VMD preflight failed -- not launching the GUI")
        info = dict(l.split("=", 1) for l in got)
        if int(info.get("PGD_FRAMES", -1)) != nframes:
            sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
            raise SystemExit(
                "pgd: preflight loaded %s frames, expected %d. Look for "
                "'Incorrect number of atoms' above -- VMD does not raise on "
                "that." % (info.get("PGD_FRAMES"), nframes))
        cli.eprint("[pgd] preflight ok: %s frames, %s atoms"
                   % (info["PGD_FRAMES"], info["PGD_ATOMS"]))

    # -eofexit with -dispdev text: without it VMD finishes the script and then
    # blocks on the interactive Tcl prompt forever -- a headless run that
    # errors looks identical to one that is working, with an idle CPU. An
    # interactive text session still works; Ctrl-D ends it.
    cmd = [vmd] + (["-dispdev", "text", "-eofexit"] if args.text else []) \
        + ["-e", str(deck)]
    if args.background:
        log = deck.parent / "vmd.log"
        # VMD reads its console from stdin and exits on EOF -- even with a GUI.
        # Handing it /dev/null therefore closes the window the instant the deck
        # finishes, which defeats the whole point of --background. Give it a
        # pty instead, and leak the master end into the child so nothing ever
        # signals EOF; both ends close when VMD exits, so nothing is left
        # behind (unlike holding the pipe open with a stray `sleep`).
        import pty
        master, slave = pty.openpty()
        try:
            with open(log, "wb") as lf:
                proc = subprocess.Popen(cmd, stdin=slave, stdout=lf, stderr=lf,
                                        start_new_session=True,
                                        pass_fds=(master,))
        finally:
            os.close(slave)
            os.close(master)
        cli.eprint("[pgd] launched in background (pid %d); log: %s"
                   % (proc.pid, log))
        cli.eprint("[pgd] close the VMD window, or: kill %d" % proc.pid)
        return 0
    os.execvp(cmd[0], cmd)


def _cli():
    """Expected failures are user-facing messages, not tracebacks."""
    try:
        return main()
    except S.NeedsStripCopy as e:
        cli.eprint("[pgd] %s" % e)
        return 5
    except (L.LineageBroken, G.UnsafeWrite) as e:
        cli.eprint("[pgd] %s" % e)
        return 1
    except KeyboardInterrupt:
        cli.eprint("[pgd] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(_cli())
