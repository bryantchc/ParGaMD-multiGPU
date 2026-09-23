#!/usr/bin/env python3
# summary: locate segments populating a point on the explored PES
# group: 2 find something to look at
"""pgd find -- which walkers visited this point on the PES?

Scans every frame's progress coordinate in west.h5 (~0.3 s for the whole run)
and reports the segments within a tolerance of the given CV values, ranked by
weight. Weight is the right default: it answers "show me the TYPICAL structure
here", whereas ranking by CV distance surfaces walkers whose weights are ~1e-34.

Pipe straight into `pgd load`:

    pgd find --cv 5.1,5.9 --tol 0.25 --top 1 --tsv | awk '{print $2":"$3}'
    pgd load --tip $(pgd find --cv 5.1,5.9 --print-tip)
"""
import sys

from pgdlib import cli
from pgdlib import lineage as L
from pgdlib import pes
from pgdlib import solvent as S


def build_parser():
    p = cli.base_parser(__doc__)
    g = p.add_argument_group("the point")
    g.add_argument("--cv", required=True, metavar="CV0,CV1",
                   help="CV values, e.g. --cv 5.1,5.9")
    g.add_argument("--tol", default="0.25",
                   help="tolerance: one number (Euclidean radius) or per-CV "
                        "with --metric box (default 0.25)")
    g.add_argument("--metric", choices=["disk", "box"], default="disk",
                   help="disk = Euclidean radius (default; the CVs share units), "
                        "box = per-CV window")
    g.add_argument("--frames", choices=["last", "any"], default="last",
                   help="'last' matches only the segment endpoint (default); "
                        "'any' matches any of its 100 frames")
    p.add_argument("--iter-lo", type=int, default=1,
                   help="lowest iteration to search")
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--rank", choices=["weight", "distance", "recency"],
                   default="weight")
    p.add_argument("--top", type=int, default=20, help="rows to show (0 = all)")
    p.add_argument("--lineage", action="store_true",
                   help="also report each hit's lineage length (slower)")
    p.add_argument("--print-tip", action="store_true",
                   help="print only the best hit as ITER:SEG, for shell use")
    cli.add_output_args(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cv = [float(x) for x in args.cv.split(",")]
    except ValueError:
        raise SystemExit("pgd: --cv wants comma-separated numbers, e.g. 5.1,5.9")
    try:
        tol = [float(x) for x in str(args.tol).split(",")]
    except ValueError:
        raise SystemExit("pgd: --tol wants a number, or per-CV numbers")
    if args.metric == "disk" and len(tol) > 1:
        raise SystemExit("pgd: --metric disk takes a single radius; "
                         "use --metric box for a per-CV window")

    rp, h5, max_iter = cli.open_run(args)

    hits = pes.find_points(h5, max_iter, cv, tol, metric=args.metric,
                           frames=args.frames, iter_lo=args.iter_lo,
                           min_weight=args.min_weight)

    if not hits:
        near = pes.nearest(h5, max_iter, cv, frames=args.frames,
                           iter_lo=args.iter_lo)
        lo, hi = pes.cv_extent(h5, max_iter, args.iter_lo)
        msg = ["pgd: no segment within %s of (%s) using --frames %s"
               % (args.tol, args.cv, args.frames)]
        if near:
            msg.append("     nearest populated point is %.3f away: "
                       "iter %d seg %d frame %d at (%.3f, %.3f)"
                       % (near["dist"], near["iter"], near["seg"],
                          near["frame"], near["cv"][0], near["cv"][1]))
        msg.append("     sampled CV range: cv0 %.2f-%.2f, cv1 %.2f-%.2f"
                   % (lo[0], hi[0], lo[1], hi[1]))
        raise SystemExit("\n".join(msg))

    if args.rank == "weight":
        hits.sort(key=lambda h: (-h["weight"], h["dist"]))
    elif args.rank == "distance":
        hits.sort(key=lambda h: (h["dist"], -h["weight"]))
    else:
        hits.sort(key=lambda h: (-h["iter"], -h["weight"]))

    if args.print_tip:
        print("%d:%d" % (hits[0]["iter"], hits[0]["seg"]))
        return 0

    shown = hits if args.top == 0 else hits[:args.top]

    for h in shown:
        st = S.classify(rp, h["iter"], h["seg"])
        h["solvent"] = st.value
        h["needs_strip_copy"] = bool(
            st is S.SolventState.FULL
            and not S.cached_strip_is_valid(rp, h["iter"], h["seg"]))
        if args.lineage:
            h["lineage_len"] = len(L.walk_lineage(h5, h["iter"], h["seg"]))
    cli.release(h5)

    if args.json:
        print(cli.dumps({"cv": cv, "tol": tol, "metric": args.metric,
                         "frames": args.frames, "max_iter": max_iter,
                         "n_matches": len(hits), "matches": shown}))
        return 0

    headers = ["rank", "iter", "seg", "frame", "nfr", "dist", "weight",
               "cv0", "cv1", "solvent"]
    aligns = [">", ">", ">", ">", ">", ">", ">", ">", ">", "<"]
    if args.lineage:
        headers.insert(-1, "lin")
        aligns.insert(-1, ">")

    rows = []
    for i, h in enumerate(shown, 1):
        r = [i, h["iter"], "%06d" % h["seg"], h["frame"],
             h["n_matching_frames"], "%.3f" % h["dist"], "%.3e" % h["weight"],
             "%.3f" % h["cv"][0], "%.3f" % h["cv"][1]]
        if args.lineage:
            r.append(h["lineage_len"])
        r.append(h["solvent"] + (" NEEDS-STRIP-COPY" if h["needs_strip_copy"] else ""))
        rows.append(r)

    if args.tsv:
        for r in rows:
            print("\t".join(str(c) for c in r))
        return 0

    print(cli.table(rows, headers, aligns))
    print()
    print("%d segment(s) match; showing %d, ranked by %s."
          % (len(hits), len(shown), args.rank))
    best = shown[0]
    print("To open the continuous trajectory from the top hit back to its root:")
    print("    pgd load --tip %d:%d" % (best["iter"], best["seg"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
