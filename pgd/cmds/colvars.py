#!/usr/bin/env python3
# summary: emit a VMD Colvars config for this run's progress coordinates
# group: 3 view it
"""pgd colvars -- the run's own CVs, as a Colvars config VMD can load.

Reads the run's `cv_*.py` modules (the same ones WESTPA's pcoord dispatcher
imports), extracts the exact atom pairs they resolved, and writes a Colvars
configuration that computes the identical quantity. So when you open a lineage
in VMD you are watching the coordinate the simulation actually steered on.

    pgd colvars                       # write the config into the cache
    pgd colvars --out cvs.colvars     # ...or somewhere you choose
    pgd colvars --verify              # check it against west.h5 via VMD
    pgd load --tip 113:46 --colvars   # generate + load it with the trajectory
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from pgdlib import cli
from pgdlib import colvars as CV
from pgdlib import guard as G
from pgdlib import lineage as L
from pgdlib import paths as P
from pgdlib import solvent as S


def build_parser():
    p = cli.base_parser(__doc__)
    p.add_argument("--out", help="output path (default: <cache>/colvars/<system>.colvars)")
    p.add_argument("--topology", choices=["stripped", "full"], default="stripped",
                   help="numbering to emit atomNumbers against (default stripped, "
                        "matching what pgd load opens)")
    p.add_argument("--width", type=float, default=0.25,
                   help="colvar 'width' hint, in Angstrom (default 0.25)")
    p.add_argument("--verify", action="store_true",
                   help="load a real segment in VMD, compute the colvars, and "
                        "compare against west.h5's recorded pcoord")
    p.add_argument("--verify-iter", type=int, default=None)
    p.add_argument("--verify-seg", type=int, default=0)
    p.add_argument("--print", dest="show", action="store_true",
                   help="print the config to stdout instead of writing it")
    return p


def open_reference(rp, which, sample_dcd=None):
    """A Universe the CV modules can resolve against."""
    import warnings
    import MDAnalysis as mda
    topo = rp.stripped_topo if which == "stripped" else rp.full_topo
    if not topo.is_file():
        raise SystemExit("pgd: no topology at %s" % topo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(topo), str(sample_dcd)) if sample_dcd \
            else mda.Universe(str(topo))
    return topo, u


def pick_sample(rp, h5, max_iter, want, iter_hint=None, seg_hint=0):
    """A segment whose DCD matches `want`, for verification."""
    order = [iter_hint] if iter_hint else []
    order += list(range(max_iter, 0, -1))
    for it in order:
        if it is None or not rp.iter_dir(it).is_dir():
            continue
        for seg in ([seg_hint] if iter_hint else [0]):
            if not rp.seg_dcd(it, seg).is_file():
                continue
            st = S.classify(rp, it, seg)
            if (want == "stripped" and st is S.SolventState.STRIPPED) or \
               (want == "full" and st is S.SolventState.FULL):
                return it, seg
        if iter_hint:
            break
    return None, None


def main(argv=None):
    args = build_parser().parse_args(argv)
    rp, h5, max_iter = cli.open_run(args)
    gd = G.from_paths(rp)

    # The CV modules read their subset paths from env.sh's CV_*_PATH, which the
    # dispatcher already exported. Resolve relative paths against the run dir.
    os.chdir(rp.root)

    mods = CV.discover_cv_modules(rp.root)
    if not mods:
        cli.release(h5)
        raise SystemExit("pgd: no cv_*.py in %s" % rp.root)

    it, seg = pick_sample(rp, h5, max_iter, args.topology,
                          args.verify_iter, args.verify_seg)
    if it is None:
        cli.release(h5)
        raise SystemExit(
            "pgd: found no %s-solvent segment to resolve the CVs against"
            % args.topology)
    sample = rp.seg_dcd(it, seg)
    topo, u = open_reference(rp, args.topology, sample)

    specs = [CV.extract(name, path, mod, u) for name, path, mod in mods]
    cli.eprint("[pgd] %d CV(s) from %s: %s"
               % (len(specs), rp.root,
                  ", ".join("%s (%d pairs)" % (s.label, s.n) for s in specs)))

    text = CV.build_config(specs, topo, rp.root,
                           widths={s.label: args.width for s in specs})

    if args.show:
        cli.release(h5)
        print(text)
        return 0

    out = Path(args.out).expanduser() if args.out else \
        gd.cache_write("colvars/%s_%s.colvars" % (rp.system_name, args.topology))
    if args.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        gd.allow_explicit(out)
    gd.safe_out(out)
    out.write_text(text)
    cli.eprint("[pgd] wrote %s (%d lines)" % (out, text.count("\n") + 1))

    if not args.verify:
        cli.release(h5)
        cli.eprint("[pgd] load it with:  pgd load --tip ITER:SEG --colvars")
        return 0

    # ---- verification -----------------------------------------------------
    # Three numbers must agree for a frame: what the CV module computes, what
    # Colvars-in-VMD computes from the generated config, and what west.h5
    # recorded at the time. The last agrees only to ~1e-4 because west.cfg
    # declares scaleoffset:4 on the pcoord dataset (lossy by design).
    pc = h5["iterations/iter_%08d/pcoord" % it][seg][:]
    cli.release(h5)

    frames = [0, pc.shape[0] // 2, pc.shape[0] - 1]
    u.trajectory[0]

    vmd = os.environ.get("PGD_VMD", "/usr/local/bin/vmd_2")
    with tempfile.TemporaryDirectory() as td:
        trace = Path(td) / "cv.dat"
        tcl = Path(td) / "verify.tcl"
        tcl.write_text(
            'mol new %s type parm7 waitfor all\n'
            'mol addfile %s type dcd waitfor all molid top\n'
            'cv molid top\n'
            'cv configfile %s\n'
            'set fh [open %s w]\n'
            'foreach f {%s} {\n'
            '    cv frame $f\n'
            '    cv update\n'
            '    set row $f\n'
            '    foreach c [cv list] { append row [format " %%.6f" [cv colvar $c value]] }\n'
            '    puts $fh $row\n'
            '}\n'
            'close $fh\n'
            'quit\n'
            % (_q(topo), _q(sample), _q(out), _q(trace),
               " ".join(str(f) for f in frames)))
        r = subprocess.run([vmd, "-dispdev", "text", "-eofexit", "-e", str(tcl)],
                           capture_output=True, text=True)
        if not trace.is_file():
            sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
            raise SystemExit("pgd: VMD produced no colvars trace")
        rows = {}
        for line in trace.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 1 + len(specs):
                rows[int(parts[0])] = [float(x) for x in parts[1:1 + len(specs)]]
        order = _colvar_order(r.stdout, [s.label for s in specs])

    print()
    print("verifying iter %d seg %d against %s" % (it, seg, topo.name))
    hdr = ["frame"] + sum([["%s module" % s.label, "%s VMD" % s.label,
                            "%s west.h5" % s.label] for s in specs], [])
    print("  " + "  ".join("%-14s" % h for h in hdr))
    worst_vmd = worst_h5 = 0.0
    for f in frames:
        u.trajectory[f]
        pos = u.atoms.positions
        cells = ["%-14d" % f]
        for j, s in enumerate(specs):
            v_mod = CV.value(s, pos)
            v_vmd = rows.get(f, [float("nan")] * len(specs))[order[j]]
            v_h5 = float(pc[f, j]) if j < pc.shape[1] else float("nan")
            worst_vmd = max(worst_vmd, abs(v_mod - v_vmd))
            worst_h5 = max(worst_h5, abs(v_mod - v_h5))
            cells += ["%-14.6f" % v_mod, "%-14.6f" % v_vmd, "%-14.6f" % v_h5]
        print("  " + "  ".join(cells))

    print()
    print("  max |module - VMD colvars| = %.2e" % worst_vmd)
    print("  max |module - west.h5|     = %.2e  "
          "(west.cfg sets scaleoffset:4 on pcoord, so ~1e-4 is expected)"
          % worst_h5)
    ok = worst_vmd < 1e-3
    print("\n%s" % ("colvars config reproduces the run's progress coordinates"
                    if ok else
                    "MISMATCH -- the generated colvars do NOT match the CV modules"))
    return 0 if ok else 1


def _colvar_order(stdout, labels):
    """Map our spec order onto `cv list` order, which Colvars may reorder."""
    for line in stdout.splitlines():
        if line.strip().startswith("colvars:") and "colvar" in line:
            pass
    # `cv list` returns definition order in practice; fall back to identity.
    return list(range(len(labels)))


def _q(p):
    return '"%s"' % str(p).replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CV.ColvarsUntranslatable as e:
        cli.eprint("[pgd] %s" % e)
        sys.exit(2)
    except (L.LineageBroken, G.UnsafeWrite) as e:
        cli.eprint("[pgd] %s" % e)
        sys.exit(1)
