#!/usr/bin/env python3
# summary: regenerate the free-energy surface, and pick a point off it
# group: 2 find something to look at
"""pgd fes -- PMF/FES via the run's existing pyreweight.py, plus a picker.

A thin, opinionated wrapper. pyreweight.py stays the one free-energy
implementation in the tree; this fixes three things about how it is currently
driven:

  1. Correct CV labels by default. pyreweight_analysis.sh still says
     "RMSD (A)" / "Rg (A)" -- stale. The CVs are mean heavy-atom distances over
     unsatisfied native contacts: RecI CV0 and PI CV1, both in Angstroms.
     max_pyreweight.sh has it right; we follow that.

  2. Nothing in the run directory is ever deleted. generate_png.sh opens with
     `rm *.png; rm ./png/*.png`. pgd writes into the cache and symlinks the
     newest result into png/ instead.

  3. --max-iter defaults to the last COMPLETE iteration, so the in-flight
     iteration can never poison a PMF.

Grid defaults follow max_pyreweight.sh (-Xdim 0 40 -Ydim 0 40 -disc 0.33
-Emax 7), not west.cfg's bin boundaries: the sampled CV range is roughly
3.9-31 / 3.9-37 A, while the configured RectilinearBinMapper grid stops at
10 / 7.5 and puts everything above into the `inf` overflow bin.

--pick renders the resulting PMF and turns a click into a `pgd find` command.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pgdlib import cli
from pgdlib import guard as G

DEFAULT_LABELS = "RecI CV0 (A),PI CV1 (A)"


def build_parser():
    p = cli.base_parser(__doc__)
    p.add_argument("--dim", type=int, choices=[1, 2], default=2)
    p.add_argument("--job", default="noweight",
                   choices=["noweight", "amdweight", "amdweight_MC",
                            "amdweight_CE"])
    p.add_argument("--cv-cols", default=None,
                   help="default: '0,1' for --dim 2, '0' for --dim 1")
    p.add_argument("--cv-labels", default=None,
                   help="default: %r" % DEFAULT_LABELS)
    p.add_argument("--xdim", default="0 40")
    p.add_argument("--ydim", default="0 40")
    p.add_argument("--disc", default="0.33")
    p.add_argument("--emax", default="7")
    p.add_argument("-T", "--temperature", default="300")
    p.add_argument("--out", default=None, help="output basename")
    p.add_argument("--sweep", metavar="LO-HI",
                   help="regenerate one PMF per iteration in the range "
                        "(replaces generate_png.sh, without its rm *.png)")
    p.add_argument("--pick", action="store_true",
                   help="render the PMF and turn a click into a pgd find command")
    p.add_argument("--open", dest="show", action="store_true",
                   help="display the PMF without picking")
    p.add_argument("--link-png", action="store_true",
                   help="also symlink the newest PNGs into <run>/png/")
    return p


def run_one(rp, gd, args, max_iter, it):
    cols = args.cv_cols or ("0,1" if args.dim == 2 else "0")
    labels = args.cv_labels or (DEFAULT_LABELS if args.dim == 2
                                else DEFAULT_LABELS.split(",")[0])
    name = args.out or ("pmf-%s-%dd-iter-%d" % (args.job, args.dim, it))
    workdir = gd.cache_write("fes/%s/.keep" % name).parent

    cmd = [os.environ["PGD_PY"], str(rp.root / "pyreweight.py"),
           "-dim", str(args.dim), "-we", "-we-root", str(rp.root),
           "-job", args.job, "-T", str(args.temperature),
           "-Emax", str(args.emax),
           "--cv-cols", cols, "--cv-labels", labels,
           "--max-iter", str(it),
           "--plots", "--save-npz",
           "--output", name]
    if args.dim >= 1:
        cmd += ["-discX", str(args.disc), "-Xdim"] + args.xdim.split()
    if args.dim >= 2:
        cmd += ["-discY", str(args.disc), "-Ydim"] + args.ydim.split()

    cli.eprint("[pgd] %s" % " ".join(cmd))
    # pyreweight writes its outputs into the cwd; give it the cache dir so the
    # run directory is never touched.
    rc = subprocess.call(cmd, cwd=str(workdir))
    if rc != 0:
        raise SystemExit("pgd: pyreweight.py failed (rc=%d) in %s" % (rc, workdir))
    return workdir, name


def pick(workdir, name, labels):
    """Render the PMF and translate a click into a pgd find invocation."""
    import numpy as np
    import matplotlib

    xvg = None
    for cand in workdir.glob("pmf-*%s*.xvg" % name.split("-")[-1]):
        xvg = cand
        break
    if xvg is None:
        cands = sorted(workdir.glob("*.xvg"))
        if not cands:
            raise SystemExit("pgd: pyreweight wrote no .xvg in %s" % workdir)
        xvg = cands[0]

    data = np.loadtxt(xvg, comments=("#", "@"))
    if data.ndim != 2 or data.shape[1] < 3:
        raise SystemExit("pgd: %s is not a 2-D PMF (need x y value columns)" % xvg)
    x, y, v = data[:, 0], data[:, 1], data[:, 2]
    xs, ys = np.unique(x), np.unique(y)
    grid = np.full((xs.size, ys.size), np.nan)
    xi = {a: i for i, a in enumerate(xs)}
    yi = {a: i for i, a in enumerate(ys)}
    for a, b, c in zip(x, y, v):
        grid[xi[a], yi[b]] = c

    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    m = ax.pcolormesh(xs, ys, np.ma.masked_invalid(grid).T, cmap="jet",
                      shading="nearest")
    fig.colorbar(m, label="PMF (kcal/mol)")
    lx, ly = (labels.split(",") + ["CV1"])[:2]
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)

    populated = ~np.isnan(grid)
    if headless:
        png = workdir / "pick.png"
        ax.set_title("pgd fes")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        cli.eprint("[pgd] no display; wrote %s" % png)
        deep = np.dstack(np.unravel_index(
            np.argsort(np.where(populated, grid, np.inf), axis=None),
            grid.shape))[0][:8]
        print("\ndeepest populated bins:")
        for i, j in deep:
            print("  (%.2f, %.2f)  PMF %.2f kcal/mol   ->  pgd find --cv %.2f,%.2f"
                  % (xs[i], ys[j], grid[i, j], xs[i], ys[j]))
        return 0

    ax.set_title("click a populated bin (close the window when done)")
    picked = []

    def on_click(ev):
        if ev.inaxes is not ax or ev.button != 1:
            return
        i = int(np.argmin(np.abs(xs - ev.xdata)))
        j = int(np.argmin(np.abs(ys - ev.ydata)))
        if not populated[i, j]:
            cli.eprint("[pgd] (%.2f, %.2f) is unpopulated" % (ev.xdata, ev.ydata))
            return
        cx, cy = float(xs[i]), float(ys[j])
        tol = float(max(np.diff(xs).min(), np.diff(ys).min())) / 2.0
        picked.append((cx, cy, tol))
        ax.plot([cx], [cy], "wo", ms=9, mec="k", zorder=5)
        fig.canvas.draw_idle()
        print("picked (%.3f, %.3f)  PMF %.2f kcal/mol" % (cx, cy, grid[i, j]))
        print("    pgd find --cv %.3f,%.3f --tol %.3f --frames any" % (cx, cy, tol))

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if picked:
        cx, cy, tol = picked[-1]
        print("\nlast pick: pgd find --cv %.3f,%.3f --tol %.3f --frames any"
              % (cx, cy, tol))
    return 0


def main(argv=None):
    from pgdlib import paths as P
    args = build_parser().parse_args(argv)
    rp, h5, max_iter = cli.open_run(args)
    gd = G.from_paths(rp)

    # pyreweight.py opens west.h5 itself; don't hold a second handle across it.
    cli.release(h5)

    if not (rp.root / "pyreweight.py").is_file():
        raise SystemExit("pgd: no pyreweight.py in %s" % rp.root)

    labels = args.cv_labels or DEFAULT_LABELS

    if args.sweep:
        try:
            lo, hi = (int(x) for x in args.sweep.split("-"))
        except ValueError:
            raise SystemExit("pgd: --sweep wants LO-HI, e.g. --sweep 1-115")
        hi = min(hi, max_iter)
        cli.eprint("[pgd] sweeping iterations %d..%d (nothing in %s is deleted)"
                   % (lo, hi, rp.root / "png"))
        for it in range(lo, hi + 1):
            run_one(rp, gd, args, max_iter, it)
        return 0

    it = args.max_iter or max_iter
    workdir, name = run_one(rp, gd, args, max_iter, it)
    cli.eprint("[pgd] outputs in %s" % workdir)

    if args.link_png:
        dest = rp.root / "png"
        dest.mkdir(exist_ok=True)
        for png in workdir.glob("*.png"):
            link = dest / png.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(png)
        cli.eprint("[pgd] symlinked PNGs into %s (existing files untouched)" % dest)

    if args.pick or args.show:
        return pick(workdir, name, labels)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
