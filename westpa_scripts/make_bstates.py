#!/usr/bin/env python3
"""
make_bstates.py -- build a run's basis states from a completed ParGaMD run.

Lets a WE run start from MANY different structures instead of a single
equilibrated one, which is the point of the per-basis-state checkpoint support
in runseg.sh and the topology_override.txt support in get_pcoord.sh.

What it does
------------
Takes one iteration of a finished WESTPA run, scores every walker's final frame
by cv_0 (PAM-duplex RMSD to 5B43 after interface-CA superposition), throws away
the walkers that are too far from the docked pose to be worth continuing, and
writes what is left out as WESTPA basis states with renormalised weights.

Each basis state directory gets:

    gamd_restart.checkpoint   the walker's own OpenMM checkpoint -- full
                              solvated positions, velocities and GaMD state.
                              This is what actually seeds the MD (runseg.sh
                              hands it to gamdRunner, which loadCheckpoint()s
                              it over whatever coordinates.rst7 contained).
    output_restart.dcd        single SOLVENT-FREE frame, used only by
                              get_pcoord.sh to report the initial pcoord.
    topology_override.txt     tells get_pcoord.sh to read that frame against
                              the stripped topology.

Why the source frame is solvent-free
------------------------------------
The source run strips solvent from trajectories two iterations behind the front,
so by the time it finished, iterations up to ~120 had stripped DCDs. The
checkpoints were NOT stripped, so nothing is lost for restarting: the solvated
state is in the checkpoint. The stripped frame is only ever used to compute a
pcoord, and every CV in this run touches protein + nucleic atoms only, so that
pcoord is identical either way.

Weights
-------
WESTPA weights within one iteration sum to 1, so they are a genuine probability
distribution over that iteration's walkers. Discarding walkers and renormalising
the survivors to sum to 1 is the natural thing to do -- it conditions that
distribution on "the walker was close enough to keep". All basis states are
taken from a SINGLE iteration on purpose: mixing iterations would double-count
lineages, since an iteration-N walker is a descendant of an iteration-M one.

Caveat, stated plainly: the source weights span ~1e-26 to ~1e-35 because the
source run's fine bin grid fragmented them (see README_PAMDOCK.md). After
renormalisation they are a sensible relative distribution over the retained
structures, but they do NOT carry meaningful absolute free-energy information,
and neither will anything computed from them. This run is for generating a
structure, not for a PMF.

Usage
-----
    python make_bstates.py                      # defaults: iter 105, cut 3.4 A
    python make_bstates.py --cut 3.2
    python make_bstates.py --iter 110 --cut 3.3 --max-states 120
    python make_bstates.py --link               # symlink checkpoints (saves disk,
                                                # but ties this run to the source)
"""

import os
import sys
import json
import shutil
import argparse
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def exact_normalise(w):
    """Return weights that sum to EXACTLY 1.0 in float64, and round-trip through text.

    WESTPA asserts abs(1 - sum(weights)) < EPS * (n_segments + n_active_bins)
    in sim_manager.report_bin_statistics, which for ~100 segments is a tolerance
    of only ~2.6e-14. Writing weights with a fixed number of decimal digits blows
    straight through that: 194 weights written as %.12e summed to
    1.0000000000000273, an error of 2.7e-14, and w_run died on that assertion at
    iteration 1 before propagating anything.

    w_init does have its own "explicitly renormalizing" fallback, but in practice
    it does not persist -- the error in west.h5 came back essentially unchanged
    (2.73e-14 -> 2.71e-14). So the file we write has to be exact to begin with.

    Two things are needed:
      * shortest round-trip repr, so float(repr(x)) == x exactly;
      * an explicit residual correction, since dividing by the sum does not
        by itself give a sequence that re-sums to exactly 1.0.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    w /= w.sum()
    for _ in range(8):
        residual = 1.0 - float(w.sum())
        if residual == 0.0:
            break
        # Put the correction on the largest weight, where it is relatively
        # smallest and cannot make anything negative.
        w[int(np.argmax(w))] += residual
    return w


def format_weight(x):
    """Shortest representation that parses back to the identical float64."""
    return repr(float(x))
RUN_ROOT = os.environ.get("WEST_SIM_ROOT", os.path.dirname(HERE))
SYSTEM_NAME = os.environ.get("SYSTEM_NAME", "")


CV_FILE = []


def load_cv0(root):
    """Import the run's first CV so selection uses exactly the production CV.

    The dispatcher globs cv_*.py and sorts lexically, so dimension 0 is whatever
    sorts first. That may be a bare cv_0.py or a descriptive name such as
    cv_0_pam_readout.py, so match the same way rather than assuming cv_0.py.
    """
    import glob as _glob
    os.environ.setdefault("WEST_SIM_ROOT", root)
    matches = sorted(_glob.glob(os.path.join(root, "cv_0*.py")))
    if not matches:
        sys.exit("no cv_0*.py found in %s -- is that the run directory?" % root)
    path = matches[0]
    print("[cv] scoring with %s" % os.path.basename(path))
    CV_FILE.append(os.path.basename(path))
    spec = importlib.util.spec_from_file_location("cv_0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refresh_weights(out_dir):
    """Rewrite bstates.txt from manifest.json without rebuilding the state dirs."""
    man_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(man_path):
        sys.exit("no manifest at %s -- run a full build first" % man_path)
    with open(man_path) as fh:
        man = json.load(fh)
    states = man["states"]

    wn = exact_normalise([st["weight_source"] for st in states])
    resid = 1.0 - float(wn.sum())
    print("[refresh] %d states, sum-1 = %.3e" % (len(states), resid))
    if resid != 0.0:
        sys.exit("could not normalise weights to exactly 1.0")

    lines = ["# basis states harvested from %s iteration %s"
             % (man.get("source", "?"), man.get("source_iteration", "?")),
             "# selection: %s < %s" % (man.get("cv_file", "cv_0"), man.get("cut_angstrom", "?")),
             "# weights: source WE weights renormalised over the retained set",
             "# name  probability  auxref"]
    for st, p in zip(states, wn):
        st["weight_renormalised"] = float(p)
        lines.append("%s  %s  %s" % (st["name"], format_weight(p), st["name"]))

    with open(os.path.join(out_dir, "bstates.txt"), "w") as fo:
        fo.write("\n".join(lines) + "\n")
    with open(man_path, "w") as fo:
        json.dump(man, fo, indent=2)

    check = np.array([float(l.split()[1]) for l in lines if not l.startswith("#")],
                     dtype=np.float64)
    print("[refresh] re-parsed from disk: n=%d  sum-1 = %.3e" % (len(check), 1.0 - check.sum()))
    print("[refresh] wrote %s/bstates.txt (state directories untouched)" % out_dir)


def build_from_selection(args, mda, h5py):
    """Harvest an explicit list of (iteration, segment) walkers, possibly across
    several iterations.

    Exists because the right selection criterion is often NOT the driven
    coordinate. In this project, selecting basis states by the run's own cv_0
    twice discarded the structurally best frames -- the winners by independent
    contact recovery sat mid-range on cv_0. Weights are forced uniform: WE
    weights from different iterations are not on a common scale, and the
    structures here were chosen on merit rather than probability.
    """
    import shutil, json as _json
    import numpy as _np

    sel = _json.load(open(args.select_file))
    pairs = [(int(a), int(b)) for a, b in sel["pairs"]]
    print("[sel] %d walkers from %s" % (len(pairs), args.select_file))
    if sel.get("criterion"):
        print("[sel] criterion: %s" % sel["criterion"])

    strip_top = os.path.join(args.run_root, "system", args.system + ".stripped.parm7")
    full_top = os.path.join(args.run_root, "system", args.system + ".parm7")

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    # atom count -> topology, computed once
    top_by_natoms = {}
    for t in (strip_top, full_top):
        if os.path.exists(t):
            try:
                top_by_natoms[mda.Universe(t).atoms.n_atoms] = t
            except Exception:
                pass
    if not top_by_natoms:
        sys.exit("no usable topology in %s" % os.path.join(args.run_root, "system"))
    print("[top] known topologies: %s"
          % ", ".join("%s=%d atoms" % (os.path.basename(v), k) for k, v in sorted(top_by_natoms.items())))

    wn = exact_normalise(_np.ones(len(pairs)))
    print("[wts]  uniform, 1/N -> %d effective states" % len(wn))

    lines = ["# basis states harvested from %s" % args.source,
             "# selection: %s" % sel.get("criterion", args.select_file),
             "# weights: uniform, 1/N",
             "# name  probability  auxref"]
    manifest = []
    u = None
    written = 0
    for i, ((it, seg), p) in enumerate(zip(pairs, wn)):
        srcd = os.path.join(args.source, "traj_segs", "%06d" % it, "%06d" % seg)
        ck = os.path.join(srcd, "gamd_restart.checkpoint")
        dcd = os.path.join(srcd, "output_restart.dcd")
        if not (os.path.exists(ck) and os.path.exists(dcd)):
            print("  [skip] iter %d seg %d: missing checkpoint or dcd" % (it, seg))
            continue
        # Pick the topology by ATOM COUNT, not by trying load_new in a loop.
        # Reusing one Universe for detection is wrong: after the first file,
        # load_new succeeds against the topology already bound, so the loop
        # records whichever topology it happened to be testing rather than the
        # one actually in use. That silently wrote the stripped topology into
        # topology_override.txt for every state after the first, and w_init then
        # failed on all of them with an atom-count mismatch.
        try:
            n_dcd = mda.coordinates.DCD.DCDReader(dcd).n_atoms
        except Exception as exc:
            print("  [skip] iter %d seg %d: unreadable dcd (%s)" % (it, seg, str(exc)[:50]))
            continue
        src_top = top_by_natoms.get(n_dcd)
        if src_top is None:
            print("  [skip] iter %d seg %d: %d atoms matches no known topology %s"
                  % (it, seg, n_dcd, sorted(top_by_natoms)))
            continue
        if u is None or u.atoms.n_atoms != n_dcd:
            u = mda.Universe(src_top, dcd)
        else:
            u.load_new(dcd)
        name = "bstate_%04d" % written
        d = os.path.join(args.out, name)
        os.makedirs(d)
        if args.link:
            os.symlink(os.path.realpath(ck), os.path.join(d, "gamd_restart.checkpoint"))
        else:
            shutil.copyfile(ck, os.path.join(d, "gamd_restart.checkpoint"))
        u.trajectory[-1]
        with mda.Writer(os.path.join(d, "output_restart.dcd"), u.atoms.n_atoms) as W:
            W.write(u.atoms)
        with open(os.path.join(d, "topology_override.txt"), "w") as fo:
            fo.write(src_top + "\n")
        lines.append("%s  %s  %s" % (name, format_weight(p), name))
        manifest.append({"name": name, "source_iter": it, "source_seg": seg,
                         "weight_renormalised": float(p)})
        written += 1
        if written % 25 == 0:
            print("  ...wrote %d" % written)

    if not written:
        sys.exit("selection produced no usable basis states")
    # re-normalise exactly over what actually got written
    wn2 = exact_normalise(_np.ones(written))
    lines = lines[:4] + ["%s  %s  %s" % (m["name"], format_weight(w), m["name"])
                         for m, w in zip(manifest, wn2)]
    for m, w in zip(manifest, wn2):
        m["weight_renormalised"] = float(w)
    with open(os.path.join(args.out, "bstates.txt"), "w") as fo:
        fo.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "manifest.json"), "w") as fo:
        _json.dump({"source": args.source, "selection_file": args.select_file,
                    "criterion": sel.get("criterion"), "weighting": "uniform",
                    "n_states": written, "states": manifest}, fo, indent=2)
    print("\n[out] %d basis states -> %s" % (written, args.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="completed run directory to harvest from")
    ap.add_argument("--system", default=SYSTEM_NAME,
                    help="system name; defaults to $SYSTEM_NAME from env.sh")
    ap.add_argument("--run-root", default=RUN_ROOT,
                    help="run dir being seeded; defaults to $WEST_SIM_ROOT")
    ap.add_argument("--iter", type=int, default=105,
                    help="source iteration (must still have per-walker "
                         "gamd_restart.checkpoint files)")
    ap.add_argument("--cut", type=float, default=3.4,
                    help="keep walkers whose cv_0 value is below this")
    ap.add_argument("--max-states", type=int, default=None,
                    help="cap the number of basis states, keeping the best by cv_0")
    ap.add_argument("--link", action="store_true",
                    help="symlink checkpoints instead of copying them")
    ap.add_argument("--select-file", default=None,
                    help="JSON file with {\"pairs\": [[iter, seg], ...]} naming exactly "
                         "which walkers to harvest. Bypasses --iter/--cut scoring, and "
                         "may span several iterations -- use it when the right selection "
                         "criterion is something the run's own CV does not measure "
                         "(e.g. Q over contacts held out of every CV). Implies uniform "
                         "weights, since weights are not comparable across iterations.")
    ap.add_argument("--uniform-weights", action="store_true",
                    help="give every retained state weight 1/N instead of "
                         "renormalising the source WE weights. Use this when the "
                         "source weights span many orders of magnitude -- see the "
                         "effective-state count printed below.")
    ap.add_argument("--out", default=None,
                    help="output bstates dir; default <run-root>/bstates")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh-weights", action="store_true",
                    help="rewrite bstates.txt from an existing manifest.json with "
                         "exactly-normalised, full-precision weights. Does not "
                         "touch the state directories, so none of the (large) "
                         "checkpoints are re-copied.")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.run_root, "bstates")

    if args.refresh_weights:
        refresh_weights(args.out)
        return

    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    import h5py

    src = args.source

    if args.select_file:
        return build_from_selection(args, mda, h5py)

    segdir = os.path.join(src, "traj_segs", "%06d" % args.iter)
    if not os.path.isdir(segdir):
        sys.exit("no such iteration directory: %s" % segdir)

    cv0 = load_cv0(args.run_root)

    # Which topology the source frames need depends on how far the source run's
    # post-iteration stripping had got when it stopped: early iterations have
    # solvent-free DCDs (24790 atoms), the last few still have full ones
    # (234549). Detect it rather than assume, so --iter works for any iteration.
    if not args.system:
        sys.exit("--system not given and $SYSTEM_NAME is unset (source env.sh first)")
    sysdir = os.path.join(args.run_root, "system")
    strip_top = os.path.join(sysdir, args.system + ".stripped.parm7")
    full_top = os.path.join(sysdir, args.system + ".parm7")
    for t in (strip_top, full_top):
        if not os.path.exists(t):
            sys.exit("missing topology: %s" % t)

    probe = None
    for seg in range(4000):
        cand = os.path.join(segdir, "%06d" % seg, "output_restart.dcd")
        if os.path.exists(cand):
            probe = cand
            break
    if probe is None:
        sys.exit("no output_restart.dcd found under %s" % segdir)
    src_top = None
    for t in (strip_top, full_top):
        try:
            mda.Universe(t, probe)
            src_top = t
            break
        except Exception:
            continue
    if src_top is None:
        sys.exit("neither topology matches the atom count in %s" % probe)
    print("[top] source frames read against %s" % os.path.basename(src_top))

    with h5py.File(os.path.join(src, "west.h5"), "r") as fh:
        g = fh["iterations"]["iter_%08d" % args.iter]
        weights = g["seg_index"]["weight"][:]
    n_src = len(weights)
    print("[src] %s iteration %d: %d walkers" % (src, args.iter, n_src))

    # --- score every walker ------------------------------------------------
    u = None
    scored = []
    n_missing_ckpt = 0
    for seg in range(n_src):
        d = os.path.join(segdir, "%06d" % seg)
        dcd = os.path.join(d, "output_restart.dcd")
        ckpt = os.path.join(d, "gamd_restart.checkpoint")
        if not os.path.exists(dcd):
            continue
        if not os.path.exists(ckpt):
            n_missing_ckpt += 1
            continue
        try:
            if u is None:
                u = mda.Universe(src_top, dcd)
            else:
                u.load_new(dcd)
            u.trajectory[-1]
            rmsd = cv0.compute(u)
        except Exception as exc:
            print("  [skip] seg %d: %s" % (seg, str(exc)[:70]))
            continue
        scored.append((seg, float(weights[seg]), rmsd))

    if not scored:
        sys.exit("scored 0 walkers -- check --iter and that checkpoints exist")
    arr = np.array([(s, w, r) for s, w, r in scored])
    print("[scan] scored %d walkers (%d had no checkpoint)" % (len(arr), n_missing_ckpt))
    print("[scan] cv_0 (PAM RMSD): min %.2f  median %.2f  max %.2f A"
          % (arr[:, 2].min(), np.median(arr[:, 2]), arr[:, 2].max()))

    # --- select ------------------------------------------------------------
    keep = arr[arr[:, 2] < args.cut]
    print("[cut]  %d of %d walkers have cv_0 < %.2f A" % (len(keep), len(arr), args.cut))
    if len(keep) == 0:
        sys.exit("cut retained nothing; raise --cut")
    keep = keep[np.argsort(keep[:, 2])]
    if args.max_states and len(keep) > args.max_states:
        keep = keep[: args.max_states]
        print("[cap]  keeping best %d by cv_0" % args.max_states)

    w = keep[:, 1].astype(float)
    if w.sum() <= 0:
        print("[warn] retained weights sum to 0; falling back to uniform weights")
        w = np.ones(len(keep))
    neff_src = 1.0 / np.sum((w / w.sum()) ** 2) if w.sum() > 0 else 0.0
    if args.uniform_weights:
        print("[wts]  --uniform-weights: using 1/N for all %d states" % len(w))
        wn = exact_normalise(np.ones(len(w)))
    else:
        wn = exact_normalise(w)
    neff = 1.0 / np.sum(wn ** 2)
    print("[wts]  effective states: %.1f of %d  (source weights would give %.1f)"
          % (neff, len(wn), neff_src))
    if not args.uniform_weights and neff < 0.25 * len(wn):
        print("[wts]  WARNING: the source weights concentrate on %.0f%% of the retained"
              % (100.0 * neff / len(wn)))
        print("[wts]           states; the rest start with negligible weight and will be")
        print("[wts]           merged away almost immediately. Consider --uniform-weights.")
    resid = 1.0 - float(wn.sum())
    print("[wts]  renormalised %d weights, sum-1 = %.3e, min=%.3e, max=%.3e"
          % (len(wn), resid, wn.min(), wn.max()))
    if resid != 0.0:
        print("[wts]  WARNING: weights do not sum to exactly 1.0; WESTPA may reject them")
    total_src = float(arr[:, 1].sum())
    frac = (w.sum() / total_src) if total_src > 0 else float("nan")
    print("[wts]  retained walkers carried %.3e of the source iteration's weight "
          "(%.3e of 1.0)" % (frac, w.sum()))
    print("[wts]  NOTE: that fraction is ~0 because the source run's fine bin grid")
    print("[wts]        fragmented weight into the 1e-26..1e-35 range. Renormalising")
    print("[wts]        gives a usable RELATIVE distribution over the retained")
    print("[wts]        structures; it does not recover absolute free energies.")

    if args.dry_run:
        print("\n[dry-run] would write %d basis states to %s" % (len(keep), args.out))
        for i, ((seg, _, r), p) in enumerate(zip(keep, wn)):
            if i < 10:
                print("   bstate_%04d  seg %5d  cv_0 %.3f  weight %.6e" % (i, seg, r, p))
        return

    # --- write -------------------------------------------------------------
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    cvname = CV_FILE[0] if CV_FILE else "cv_0"
    wdesc = ("uniform, 1/N" if args.uniform_weights
             else "source WE weights renormalised over the retained set")
    lines = ["# basis states harvested from %s iteration %d" % (src, args.iter),
             "# selection: %s < %.4g" % (cvname, args.cut),
             "# weights: %s" % wdesc,
             "# name  probability  auxref"]
    manifest = []
    for i, ((seg, w_src, r), p) in enumerate(zip(keep, wn)):
        seg = int(seg)
        name = "bstate_%04d" % i
        d = os.path.join(args.out, name)
        os.makedirs(d)
        srcd = os.path.join(segdir, "%06d" % seg)

        ck_src = os.path.join(srcd, "gamd_restart.checkpoint")
        ck_dst = os.path.join(d, "gamd_restart.checkpoint")
        if args.link:
            os.symlink(os.path.realpath(ck_src), ck_dst)
        else:
            shutil.copyfile(ck_src, ck_dst)

        # single-frame stripped DCD for the initial pcoord
        u.load_new(os.path.join(srcd, "output_restart.dcd"))
        u.trajectory[-1]
        with mda.Writer(os.path.join(d, "output_restart.dcd"), u.atoms.n_atoms) as W:
            W.write(u.atoms)

        with open(os.path.join(d, "topology_override.txt"), "w") as fo:
            fo.write(src_top + "\n")

        lines.append("%s  %s  %s" % (name, format_weight(p), name))
        manifest.append({"name": name, "source_iter": args.iter, "source_seg": seg,
                         "cv_0_pam_rmsd": round(float(r), 4),
                         "weight_source": float(w_src), "weight_renormalised": float(p)})
        if (i + 1) % 25 == 0:
            print("  ...wrote %d/%d" % (i + 1, len(keep)))

    with open(os.path.join(args.out, "bstates.txt"), "w") as fo:
        fo.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "manifest.json"), "w") as fo:
        json.dump({"source": src, "source_iteration": args.iter, "cut_angstrom": args.cut,
                   "cv_file": (CV_FILE[0] if CV_FILE else "cv_0"),
                   "weighting": ("uniform" if args.uniform_weights else "renormalised_source"),
                   "n_states": len(manifest), "states": manifest}, fo, indent=2)

    print("\n[out] %d basis states -> %s" % (len(manifest), args.out))
    print("[out] %s/bstates.txt" % args.out)
    print("[out] %s/manifest.json" % args.out)
    print("[out] cv_0 range of retained states: %.2f - %.2f A"
          % (keep[:, 2].min(), keep[:, 2].max()))


if __name__ == "__main__":
    main()
