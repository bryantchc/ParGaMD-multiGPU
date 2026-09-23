#!/usr/bin/env python3
# summary: report on an iteration or a single segment
# group: 1 inspect the run
"""pgd info -- everything west.h5 and the disk know about an iteration or segment."""
import sys

import numpy as np

from pgdlib import cli
from pgdlib import dcdio
from pgdlib import lineage as L
from pgdlib import paths as P
from pgdlib import sizes as Z
from pgdlib import solvent as S


def build_parser():
    p = cli.base_parser(__doc__)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--seg", type=int, default=None)
    p.add_argument("--flags", action="store_true",
                   help="scan seg_logs for FROZEN / retried segments (slower)")
    cli.add_output_args(p)
    return p


def bin_index(h5, rp, n_iter, cv):
    """Bin index under THIS iteration's mapper. Display only.

    The mapper changed 21 times mid-run, so indices are not comparable across
    iterations -- which is why pgd find searches raw CV values instead.
    """
    try:
        import pickle
        hh = h5[P.h5_iter_group(n_iter)].attrs["binhash"]
        # attrs give a str; bin_topologies/index stores S64 bytes.
        if isinstance(hh, str):
            hh = hh.encode()
        idx = h5["bin_topologies/index"][:]
        pk = h5["bin_topologies/pickles"]
        for i, row in enumerate(idx):
            if row["hash"] == hh:
                blob = bytes(pk[i][:int(row["pickle_len"])])
                mapper = pickle.loads(blob)
                return (int(mapper.assign(np.asarray([cv]))[0]),
                        int(mapper.nbins))
    except Exception:                                       # noqa: BLE001
        pass
    return None, None


def seg_info(h5, rp, n_iter, seg_id, want_flags):
    nd = L.node(h5, n_iter, seg_id)
    pc = h5[P.h5_iter_group(n_iter) + "/pcoord"][seg_id]
    st = S.classify(rp, n_iter, seg_id)
    dcd = rp.seg_dcd(n_iter, seg_id)

    out = {
        "iter": n_iter, "seg": seg_id, "weight": nd.weight,
        "parent_id": nd.parent_id,
        "endpoint_type": L.ENDPOINT_TYPE_NAMES.get(nd.endpoint_type,
                                                   nd.endpoint_type),
        "status": "complete" if nd.status == L.SEG_STATUS_COMPLETE
                  else "incomplete(%d)" % nd.status,
        "pcoord_first": [float(x) for x in pc[0]],
        "pcoord_last": [float(x) for x in pc[-1]],
        "solvent": st.value,
        "dcd": str(dcd),
    }
    if nd.is_root:
        out["root"] = L.basis_state_for(h5, rp, nd)
    else:
        out["parent"] = {"iter": n_iter - 1, "seg": nd.parent_id}
    if nd.merge_parents:
        out["merge_parents"] = {
            "ids": list(nd.merge_parents),
            "note": "weight-flow parents of a merge; pgd trace/load follow "
                    "parent_id only, which is the single dynamical parent",
        }
    out["children_next_iter"] = L.children_of(h5, n_iter, seg_id)
    out["lineage_length"] = len(L.walk_lineage(h5, n_iter, seg_id))

    if dcd.is_file():
        try:
            pr = dcdio.probe(dcd)
            out["dcd_natoms"] = pr.natoms
            out["dcd_nframes"] = pr.nframes
            out["dcd_bytes"] = dcd.stat().st_size
        except dcdio.DcdLayoutError as e:
            out["dcd_error"] = str(e)
    if st is S.SolventState.FULL:
        out["cached_strip"] = S.cached_strip_is_valid(rp, n_iter, seg_id)

    b, nb = bin_index(h5, rp, n_iter, pc[-1])
    if b is not None:
        out["bin"] = {"index": b, "nbins": nb,
                      "note": "under iteration %d's own mapper; NOT comparable "
                              "across iterations" % n_iter}

    gl = rp.seg_gamd_log(n_iter, seg_id)
    if gl.is_file():
        try:
            d = np.loadtxt(gl, skiprows=3)
            if d.ndim == 2 and d.shape[1] >= 8:
                dv = d[:, 6] + d[:, 7]      # NonBonded + Dihedral boost
                out["boost_dV_kcal"] = {"mean": float(dv.mean()),
                                        "sd": float(dv.std()),
                                        "min": float(dv.min()),
                                        "max": float(dv.max())}
        except (OSError, ValueError):
            pass

    if want_flags:
        out["frozen"] = seg_id in L.frozen_segments(
            str(rp.seg_log_tar(n_iter)), n_iter)
        out["retried"] = seg_id in L.retried_segments(rp, n_iter)
    return out


def iter_info(h5, rp, n_iter, want_flags):
    si = h5[P.h5_iter_group(n_iter) + "/seg_index"][:]
    pc = h5[P.h5_iter_group(n_iter) + "/pcoord"]
    summary = h5["summary"][n_iter - 1]
    d = rp.iter_dir(n_iter)

    flat = pc[:].reshape(-1, pc.shape[-1])
    out = {
        "iter": n_iter,
        "n_segments": int(len(si)),
        "sum_weights": float(si["weight"].sum()),
        "summary_norm": float(summary["norm"]),
        "cputime_s": float(summary["cputime"]),
        "walltime_s": float(summary["walltime"]),
        "n_roots": int((si["parent_id"] < 0).sum()),
        "n_merges": int((si["wtg_n_parents"] > 1).sum()),
        "n_incomplete": int((si["status"] != L.SEG_STATUS_COMPLETE).sum()),
        "pcoord_range": [[float(flat[:, k].min()), float(flat[:, k].max())]
                         for k in range(flat.shape[1])],
        "weight_range": [float(si["weight"].min()), float(si["weight"].max())],
        "on_disk": d.is_dir(),
        "strip_lock": rp.iter_strip_lock(n_iter).is_file(),
    }
    if d.is_dir():
        segs = [p for p in d.iterdir() if p.is_dir() and p.name.isdigit()]
        marked = sum(1 for p in segs if (p / ".stripped").is_file())
        out["disk_segments"] = len(segs)
        out["stripped_segments"] = marked
        out["solvent"] = ("stripped" if marked == len(segs) and segs
                          else "full" if marked == 0
                          else "MIXED (%d/%d)" % (marked, len(segs)))
        if marked == 0 and segs:
            out["cached_strips"] = sum(
                1 for p in segs if S.cached_strip_is_valid(rp, n_iter, int(p.name)))
    try:
        out["binhash"] = h5[P.h5_iter_group(n_iter)].attrs["binhash"].decode() \
            if isinstance(h5[P.h5_iter_group(n_iter)].attrs["binhash"], bytes) \
            else str(h5[P.h5_iter_group(n_iter)].attrs["binhash"])
        if n_iter > 1:
            prev = h5[P.h5_iter_group(n_iter - 1)].attrs["binhash"]
            out["bin_mapper_changed_here"] = bool(
                h5[P.h5_iter_group(n_iter)].attrs["binhash"] != prev)
    except (KeyError, AttributeError):
        pass
    if want_flags:
        out["n_frozen"] = len(L.frozen_segments(str(rp.seg_log_tar(n_iter)), n_iter))
        out["n_retried"] = len(L.retried_segments(rp, n_iter))
    return out


def render(d, indent=0):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print("%s%-22s" % (pad, k + ":"))
            render(v, indent + 1)
        elif isinstance(v, list) and v and isinstance(v[0], (list, dict)):
            print("%s%-22s %s" % (pad, k + ":", v))
        else:
            print("%s%-22s %s" % (pad, k + ":", v))


def main(argv=None):
    args = build_parser().parse_args(argv)
    rp, h5, max_iter = cli.open_run(args)
    if args.iter > max_iter:
        cli.eprint("[pgd] note: iteration %d is not complete (last complete: %d)"
                   % (args.iter, max_iter))

    d = (seg_info(h5, rp, args.iter, args.seg, args.flags) if args.seg is not None
         else iter_info(h5, rp, args.iter, args.flags))
    cli.release(h5)

    if args.json:
        print(cli.dumps(d))
        return 0
    render(d)
    if args.seg is not None:
        print()
        print("open its lineage:  pgd load --tip %d:%d" % (args.iter, args.seg))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except L.LineageBroken as e:
        cli.eprint("[pgd] %s" % e)
        sys.exit(1)
