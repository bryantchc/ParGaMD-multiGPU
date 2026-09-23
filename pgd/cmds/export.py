#!/usr/bin/env python3
# summary: write a lineage out as one trajectory file (dcd/nc/pdb/xtc/...)
# group: 4 get data out
"""pgd export -- materialize a lineage as a single trajectory file.

`pgd load` never writes a trajectory; this is the command that does, for when
you want an artifact: cpptraj analysis, MDAnalysis/mdtraj, PyMOL, a colleague,
or a restart structure.

Two engines:

  splice   (default) Raw byte concatenation. After the strip cache normalizes
           every segment to one atom count, all inputs share an identical
           fixed-size record layout, so joining them is a byte copy: no float
           decode/encode, no precision loss, constant memory, disk-speed
           (~3.2 GiB in ~15 s). It is also the only engine that gets a GLOBAL
           stride right -- catdcd's -stride cannot be combined with per-file
           trimming, and cpptraj's trajin offset restarts on every file.
           DCD output only, and it restores the timestep cpptraj discards.

  cpptraj  Needed for anything that must understand the topology: a different
           output format, an atom mask, RMS fitting, imaging. Selected
           automatically when you ask for any of those.

Every export writes <out>.provenance.csv mapping each output frame back to its
(iteration, segment, frame, weight, original time).
"""
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pgdlib import cli
from pgdlib import dcdio
from pgdlib import frames as F
from pgdlib import guard as G
from pgdlib import lineage as L
from pgdlib import sizes as Z
from pgdlib import solvent as S

# cpptraj trajout keyword by extension
FORMATS = {
    "dcd": "dcd", "nc": "netcdf", "netcdf": "netcdf", "pdb": "pdb",
    "xtc": "xtc", "trr": "trr", "rst7": "restart", "crd": "crd",
    "mol2": "mol2", "binpos": "binpos", "xyz": "xyz",
}


def build_parser():
    p = cli.base_parser(__doc__)
    cli.add_tip_args(p)
    cli.add_slice_args(p)
    p.add_argument("--out", required=True, help="output path")
    p.add_argument("--format", choices=sorted(set(FORMATS)), default=None,
                   help="default: inferred from --out's extension")
    p.add_argument("--engine", choices=["auto", "splice", "cpptraj"],
                   default="auto")
    g = p.add_argument_group("topology-aware options (force the cpptraj engine)")
    g.add_argument("--mask", help="atom mask to KEEP, e.g. ':1-1316' "
                                  "(inverted internally to cpptraj's strip)")
    g.add_argument("--strip-mask", help="literal cpptraj strip mask (what to REMOVE)")
    g.add_argument("--align", metavar="MASK",
                   help="RMS-fit every frame on this mask")
    g.add_argument("--ref", default="first",
                   help="reference for --align: 'first' or a structure file")
    g.add_argument("--autoimage", action="store_true",
                   help="reimage molecules into the primary cell (CHANGES "
                        "coordinates; off by default)")
    p.add_argument("--max-pdb-frames", type=int, default=100,
                   help="refuse a multi-frame PDB above this (default 100)")
    p.add_argument("--no-provenance", action="store_true")
    p.add_argument("--warn-gb", type=float, default=None)
    return p


def run_cpptraj(deck_text, log_path):
    cp = os.environ.get("PGD_CPPTRAJ", "cpptraj")
    with tempfile.NamedTemporaryFile("w", suffix=".cpptraj", delete=False) as f:
        f.write(deck_text)
        deck = f.name
    try:
        with open(log_path, "wb") as lf:
            rc = subprocess.call([cp, "-i", deck], stdout=lf,
                                 stderr=subprocess.STDOUT)
        return rc
    finally:
        os.unlink(deck)


def main(argv=None):
    args = build_parser().parse_args(argv)
    tip_iter, tip_seg = cli.resolve_tip(args)
    rp, h5, max_iter = cli.open_run(args)
    gd = G.from_paths(rp)

    out = Path(args.out).expanduser()
    fmt = args.format or FORMATS.get(out.suffix.lstrip(".").lower())
    if not fmt:
        raise SystemExit(
            "pgd: cannot infer a format from %r; pass --format (%s)"
            % (out.name, ", ".join(sorted(set(FORMATS)))))
    fmt_kw = FORMATS[fmt]

    chain = L.walk_lineage(h5, tip_iter, tip_seg, stop_iter=args.stop_iter)
    seam, seam_ev = cli.resolve_seam(h5, chain, args.seam)
    cli.release(h5)          # nothing below reads west.h5; cpptraj runs next
    want = {"stripped": S.SolventState.STRIPPED,
            "full": S.SolventState.FULL,
            "auto": S.SolventState.UNKNOWN}[args.solvent]
    topology, resolved = S.resolve_chain(rp, chain, want=want,
                                         tip=(tip_iter, tip_seg))
    natoms = S.topo_natoms(str(topology))

    slices = F.segment_slices(chain, stride=args.stride, seam=seam,
                              tip_last_frame=args.tip_frame)
    nframes = F.total_frames(slices)

    needs_cpptraj = bool(args.mask or args.strip_mask or args.align
                         or args.autoimage or fmt_kw != "dcd")
    engine = args.engine
    if engine == "auto":
        engine = "cpptraj" if needs_cpptraj else "splice"
    elif engine == "splice" and needs_cpptraj:
        raise SystemExit(
            "pgd: --engine splice cannot do %s. It is a byte copier: DCD out, "
            "no mask, no alignment, no imaging. Use --engine cpptraj."
            % ("format %s" % fmt if fmt_kw != "dcd" else "that transformation"))

    if fmt_kw == "pdb" and nframes > args.max_pdb_frames:
        raise SystemExit(
            "pgd: %s frames as PDB would be roughly %s of text. Add --stride, "
            "or raise --max-pdb-frames if you really mean it."
            % ("{:,}".format(nframes), Z.human(nframes * natoms * 80)))

    g = Z.gate(nframes, natoms, warn_gb=args.warn_gb)
    cli.eprint(Z.render(g))
    if g["over"] and not cli.assume_yes():
        raise SystemExit(
            "pgd: estimate exceeds the limits above. Re-run with --stride %d, "
            "or --yes to force." % (g["suggested_stride"] * args.stride))

    by_key = {(nd.n_iter, nd.seg_id): p for nd, (p, _) in zip(chain, resolved)}
    sources = [by_key[(s.n_iter, s.seg_id)] for s in slices]

    cli.eprint("[pgd] %d segments, %s frames, %.2f ns, %d atoms -> %s (%s, engine=%s)"
               % (len(slices), "{:,}".format(nframes),
                  nframes * F.FRAME_PS * args.stride / 1000.0, natoms,
                  out, fmt, engine))

    if cli.dry_run():
        cli.eprint("[pgd] dry run; nothing written")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    gd.allow_explicit(out)
    gd.safe_out(out)

    t0 = time.time()
    if engine == "splice":
        title = ("pgd %d:%d->%s stride %d seam %s"
                 % (tip_iter, tip_seg, chain[0].n_iter, args.stride, seam))
        info = dcdio.splice(
            sources, [s.to_byte_range() for s in slices], out,
            natoms_expect=natoms,
            titles=[title[:80], "%d frames, %.2f ns, %d atoms"
                    % (nframes, nframes * F.FRAME_PS * args.stride / 1000.0, natoms)],
            dt_ps=F.DT_PS, interval_steps=F.DCD_INTERVAL_STEPS,
            stride_for_header=args.stride)
        cli.eprint("[pgd] wrote %s frames, %s in %.1fs"
                   % ("{:,}".format(info["nframes"]), Z.human(info["bytes"]),
                      info["seconds"]))
    else:
        lines = ["parm %s" % topology]
        for src, s in zip(sources, slices):
            a, b, st = s.to_cpptraj_trajin()
            lines.append("trajin %s %d %d %d" % (src, a, b, st))
        if args.align and args.ref != "first":
            lines.append("reference %s [ref]" % args.ref)
        if args.autoimage:
            lines.append("autoimage")
        if args.align:
            tgt = "first" if args.ref == "first" else "ref [ref]"
            lines.append("rms %s %s" % (tgt, args.align))
        if args.strip_mask:
            lines.append("strip %s" % args.strip_mask)
        elif args.mask:
            lines.append("strip !(%s)" % args.mask)
        lines += ["trajout %s %s" % (out, fmt_kw), "go", "quit", ""]
        deck = "\n".join(lines)

        log = gd.cache_write("runs/export_%d_%06d/cpptraj.out" % (tip_iter, tip_seg))
        if os.environ.get("PGD_VERBOSE"):
            cli.eprint(deck)
        rc = run_cpptraj(deck, log)
        if rc != 0 or not out.exists():
            sys.stderr.write(Path(log).read_text(errors="replace")[-3000:])
            raise SystemExit("pgd: cpptraj failed (rc=%d); log: %s" % (rc, log))

        if args.mask or args.strip_mask:
            # The output no longer matches the input topology; without a
            # matching parm7 the file is unusable in a month.
            ptop = out.with_suffix(".parm7")
            gd.allow_explicit(ptop)
            pdeck = ["parm %s" % topology,
                     "parmstrip %s" % (args.strip_mask if args.strip_mask
                                       else "!(%s)" % args.mask),
                     "parmwrite out %s" % ptop, "go", "quit", ""]
            if run_cpptraj("\n".join(pdeck), log) == 0 and ptop.exists():
                cli.eprint("[pgd] matching topology: %s" % ptop)
            else:
                cli.eprint("[pgd] WARNING: could not write a matching topology; "
                           "the masked trajectory has no parm7")
        cli.eprint("[pgd] wrote %s in %.1fs"
                   % (Z.human(out.stat().st_size), time.time() - t0))

    if not args.no_provenance:
        prov = out.with_suffix(out.suffix + ".provenance.csv")
        gd.allow_explicit(prov)
        wmap = {(nd.n_iter, nd.seg_id): nd.weight for nd in chain}
        row = 0
        with open(prov, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "iteration", "seg", "seg_frame", "time_ps",
                        "weight", "source_dcd"])
            for s, src in zip(slices, sources):
                for k, lf in enumerate(range(s.first, s.last + 1, s.step)):
                    w.writerow([row, s.n_iter, s.seg_id, lf,
                                "%.1f" % (s.t0_ps + k * s.step * F.FRAME_PS),
                                "%.6e" % wmap[(s.n_iter, s.seg_id)], src])
                    row += 1
        cli.eprint("[pgd] provenance: %s (%s rows)" % (prov, "{:,}".format(row)))

    return 0


def _cli():
    try:
        return main()
    except S.NeedsStripCopy as e:
        cli.eprint("[pgd] %s" % e)
        return 5
    except (L.LineageBroken, G.UnsafeWrite, dcdio.DcdLayoutError) as e:
        cli.eprint("[pgd] %s" % e)
        return 1
    except KeyboardInterrupt:
        cli.eprint("[pgd] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(_cli())
