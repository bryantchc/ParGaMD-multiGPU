#!/usr/bin/env python3
# summary: walk a segment's ancestry back to the initial parent
# group: 1 inspect the run
"""pgd trace -- show the lineage of a walker, root to tip.

This is the debuggable core of `load` and `export`. It touches only west.h5
plus a stat per segment, so it answers "what would be stitched, and from
where?" without reading any of the 3.2 TB of trajectory data.
"""
import sys

from pgdlib import cli
from pgdlib import frames as F
from pgdlib import lineage as L
from pgdlib import solvent as S
from pgdlib import sizes as Z


def build_parser():
    p = cli.base_parser(__doc__)
    cli.add_tip_args(p)
    cli.add_slice_args(p)
    cli.add_output_args(p)
    p.add_argument("--flags", action="store_true",
                   help="also scan seg_logs for FROZEN and retried segments "
                        "(slower: greps the per-iteration tars)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    tip_iter, tip_seg = cli.resolve_tip(args)
    rp, h5, max_iter = cli.open_run(args)

    if tip_iter > max_iter:
        raise SystemExit(
            "pgd: iteration %d is not complete (last complete is %d). "
            "Iteration %d has no on-disk segments."
            % (tip_iter, max_iter, tip_iter))

    chain = L.walk_lineage(h5, tip_iter, tip_seg, stop_iter=args.stop_iter)
    seam, seam_ev = cli.resolve_seam(h5, chain, args.seam)
    slices = F.segment_slices(chain, stride=args.stride, seam=seam,
                              tip_last_frame=args.tip_frame)

    want = {"stripped": S.SolventState.STRIPPED,
            "full": S.SolventState.FULL,
            "auto": S.SolventState.UNKNOWN}[args.solvent]

    # Per-segment state, tolerating the not-yet-stripped case so trace still
    # prints a full picture instead of failing like load would.
    states = [S.classify(rp, nd.n_iter, nd.seg_id) for nd in chain]
    needs_strip = [(nd.n_iter, nd.seg_id)
                   for nd, st in zip(chain, states)
                   if st is S.SolventState.FULL
                   and want is S.SolventState.STRIPPED
                   and not S.cached_strip_is_valid(rp, nd.n_iter, nd.seg_id)]

    frozen_by_iter, retried_by_iter = {}, {}
    if args.flags:
        for nd in chain:
            frozen_by_iter.setdefault(
                nd.n_iter, L.frozen_segments(str(rp.seg_log_tar(nd.n_iter)),
                                             nd.n_iter))
            retried_by_iter.setdefault(nd.n_iter, L.retried_segments(rp, nd.n_iter))

    root = chain[0]
    bstate = L.basis_state_for(h5, rp, root)
    cli.release(h5)          # only rendering below

    nframes = F.total_frames(slices)
    natoms = (S.topo_natoms(str(rp.stripped_topo))
              if want is not S.SolventState.FULL
              else S.topo_natoms(str(rp.full_topo)))

    sl_by_key = {(s.n_iter, s.seg_id): s for s in slices}

    records = []
    for i, (nd, st) in enumerate(zip(chain, states)):
        sl = sl_by_key.get((nd.n_iter, nd.seg_id))
        flags = []
        if nd.is_root:
            if bstate and "path" in bstate:
                flags.append("root(istate %s -> %s)"
                             % (bstate["istate_id"], bstate["path"]))
            else:
                flags.append("root(istate %s)" % (nd.istate_id,))
        elif i == 0:
            flags.append("truncated(--stop-iter %d)" % args.stop_iter)
        if nd.merge_parents:
            flags.append("merge-parents=%s (weight-flow only, not followed)"
                         % list(nd.merge_parents))
        if st is S.SolventState.FULL and want is S.SolventState.STRIPPED:
            flags.append("cached-strip" if (nd.n_iter, nd.seg_id) not in needs_strip
                         else "NEEDS-STRIP-COPY")
        if args.flags:
            if nd.seg_id in frozen_by_iter.get(nd.n_iter, ()):
                flags.append("FROZEN(parent DCD copied verbatim)")
            if nd.seg_id in retried_by_iter.get(nd.n_iter, ()):
                flags.append("retry(attempt2)")
        if nd.endpoint_type == 2:
            flags.append("merged")
        elif nd.endpoint_type == 3:
            flags.append("recycled")

        records.append({
            "n": i + 1,
            "iter": nd.n_iter,
            "seg": nd.seg_id,
            "weight": nd.weight,
            "parent_id": nd.parent_id,
            "first": sl.first if sl else None,
            "last": sl.last if sl else None,
            "step": sl.step if sl else None,
            "nframes": sl.n_frames if sl else 0,
            "t0_ps": sl.t0_ps if sl else None,
            "solvent": st.value,
            "dcd": str(rp.seg_dcd(nd.n_iter, nd.seg_id)),
            "flags": flags,
        })

    if args.json:
        print(cli.dumps({
            "tip": [tip_iter, tip_seg],
            "stop_iter": args.stop_iter,
            "max_complete_iter": max_iter,
            "seam": {"verdict": seam, **seam_ev},
            "stride": args.stride,
            "solvent": args.solvent,
            "topology": str(rp.stripped_topo if want is not S.SolventState.FULL
                            else rp.full_topo),
            "natoms": natoms,
            "n_segments": len(chain),
            "n_frames": nframes,
            "time_ps": F.total_time_ps(slices),
            "est_bytes": Z.estimate_disk_bytes(natoms, nframes),
            "needs_strip_copy": [{"iter": i, "seg": s} for i, s in needs_strip],
            "segments": records,
        }))
        return 0

    rows = [(r["n"], r["iter"], "%06d" % r["seg"], "%.3e" % r["weight"],
             r["first"] if r["first"] is not None else "-",
             r["last"] if r["last"] is not None else "-",
             r["step"] if r["step"] is not None else "-",
             r["nframes"],
             "%.1f" % r["t0_ps"] if r["t0_ps"] is not None else "-",
             r["solvent"], "; ".join(r["flags"]))
            for r in records]
    if args.tsv:
        for r in rows:
            print("\t".join(str(c) for c in r))
        return 0

    print(cli.table(
        rows,
        ["#", "iter", "seg", "weight", "first", "last", "step", "nfr",
         "t0_ps", "solvent", "flags"],
        aligns=[">", ">", ">", ">", ">", ">", ">", ">", ">", "<", "<"]))

    print()
    if seam_ev.get("mode") == "auto":
        print("seam: %s (measured over %d links: %.1f%% exact duplicates, "
              "max |dpcoord| = %.4f)"
              % (seam, seam_ev["n_links"], 100 * seam_ev["frac_exact"],
                 seam_ev["max_diff"]))
        if seam == F.SEAM_KEEP:
            print("      -> frame 0 of a child is NOT a copy of its parent's last "
                  "frame in this run,")
            print("         so no frames are dropped. Dropping them would delete "
                  "%d real frames" % (len(chain) - 1))
            print("         and open a %.0f ps gap at every seam."
                  % (2 * F.FRAME_PS))
    else:
        print("seam: %s (explicit)" % seam)

    print("total: %d segments, %s frames, %.2f ns, %s as %s (%d atoms)"
          % (len(chain), "{:,}".format(nframes),
             F.total_time_ps(slices) / 1000.0,
             Z.human(Z.estimate_disk_bytes(natoms, nframes)),
             args.solvent, natoms))

    if needs_strip:
        print()
        print("%d segment(s) still full-atom with no stripped copy: %s"
              % (len(needs_strip),
                 ", ".join("%d:%d" % x for x in needs_strip)))
        print("  originals are never modified. To normalize copies:")
        print("    pgd strip-copy --for-lineage %d:%d" % (tip_iter, tip_seg))
    return 0


def _cli():
    try:
        return main()
    except L.LineageBroken as e:
        cli.eprint("[pgd] %s" % e)
        return 1
    except KeyboardInterrupt:
        cli.eprint("[pgd] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(_cli())
