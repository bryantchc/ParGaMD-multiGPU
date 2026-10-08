#!/usr/bin/env python3
"""
make_target_counts.py -- generate a non-uniform `bin_target_counts` list for west.cfg.

WHY THIS EXISTS
---------------
west.cfg's `bin_target_counts` may be either a scalar (every bin gets the same
number of walkers) or a list of length nbins. The list form lets you spend
compute where it matters instead of spreading it evenly, but it is a FLAT list
over a multi-dimensional grid, so it cannot sensibly be hand-maintained: a
2D 15x17 grid is 255 entries, and changing a single bin boundary invalidates
all of them. This script regenerates the list from a policy function, so the
policy is what lives in version control and the list is derived.

WESTPA SUPPORT, VERIFIED (2022.15)
----------------------------------
src/westpa/core/_rc.py accepts an iterable and asserts len == mapper.nbins.
Targets are indexed by ABSOLUTE bin index in we_driver.py:

    for ibin, bin in enumerate(self.next_iter_binning):
        if len(bin) == 0: continue
        target_count = self.bin_target_counts[ibin]

Empty bins are skipped before the target is read, so the list covers occupied
and unoccupied cells alike and nothing needs recomputing as bins fill.

WHERE IT IS *NOT* SAFE
----------------------
  - Adaptive mappers (MAB, adaptive Voronoi). Their boundaries move every
    iteration, so flat index i denotes a different region each time and a
    static list silently mis-targets. westext/adaptvoronoi rewrites
    system.bin_target_counts per iteration for exactly this reason.
  - RecursiveBinMapper. Flattening is not the simple C-order product below.
  - Runs WITH TARGET STATES. Two places assume uniformity:
      we_driver.py:296    bin_target_counts[tstate_assignment] = 0
                          -> silently overwrites your value for that bin
      sim_manager.py:389  total_replicas = sum(...) - bin_target_counts[-1]
                                           * len(target_states)
                          -> uses the LAST bin's count for every target-state
                             bin; correct only if all counts are equal.
    Both are inert when there are no target states (gen_istates: false and
    w_init reporting "0 target state(s) present").

So: static RectilinearBinMapper, no recursion, no target states. That is this
project's configuration; check yours before using the output.

FLATTENING ORDER
----------------
Verified empirically against RectilinearBinMapper.assign() across all cells:
C order, index = i_dim0 * n_dim1 * ... + ... + i_dimN. For the 2D case here
that is  index = dock_bin * n_clearance_bins + clearance_bin.
--assert-order re-verifies this against the live mapper before emitting.

USAGE
-----
    python make_target_counts.py --west-cfg west.cfg --policy corner-weighted
    python make_target_counts.py --west-cfg west.cfg --policy uniform --value 4
    python make_target_counts.py ... --emit yaml     # paste-ready block
    python make_target_counts.py ... --assert-order  # verify with WESTPA itself
"""
import argparse, sys
import numpy as np
import yaml


# --------------------------------------------------------------------------
# Policies.  Each takes the per-dimension bin INDEX tuple plus the boundary
# lists and returns a walker count.  Add new ones here; keep them pure.
# --------------------------------------------------------------------------

def policy_uniform(idx, bounds, value=4):
    return value


def policy_corner_weighted(idx, bounds, value=None):
    """Spend walkers on the open+docked corner; starve dissociation.

    Tuned for Ultra_RL2.rna.preloop3, whose dim 0 is PI/WED docking (drive
    DOWN) and dim 1 is REC1 clearance (drive UP).  Rationale, measured over
    1630 parent/child pairs at tau = 200 ps:

      - clearance carries a NEGATIVE drift (-0.193 A per child per tau): REC1
        wants to shut.  Progress comes only from the upper tail, so a frontier
        bin needs enough walkers for that tail to clear zero.  P(all N children
        retreat) is 32% at N=2 but 3.4% at N=6, which over 10 iterations is the
        difference between a 2% and a 71% chance the bin still exists.
      - dock is essentially diffusive (-0.004 A, sd 0.274), so it does not need
        the same protection.
      - the closed bulk held thousands of walkers against tens in the corner:
        it is oversampled and can afford 2.
      - dock > 9 A is the duplex leaving the cleft.  Catch it, do not reward it.

    Net effect is roughly walker-NEUTRAL versus uniform 4 -- it redistributes
    rather than saves.  The payoff is frontier-bin survival, not throughput.
    """
    d, c = idx[0], idx[1]
    nd, nc = len(bounds[0]) - 1, len(bounds[1]) - 1
    dock_lo, clear_lo = bounds[0][d], bounds[1][c]

    if dock_lo >= 9.0:                     # dissociating
        return 1
    if clear_lo >= 6.6:                    # open corner, fights the drift
        return 6
    if clear_lo >= 5.8:                    # approach to the corner
        return 4
    if clear_lo < 4.6 and dock_lo >= 6.0:  # closed + poorly docked bulk
        return 2
    return 3


def policy_dock_frontier(idx, bounds, value=None):
    """Drive dim 0 (docking) DOWN; leave dim 1 NEUTRAL.

    Written for Ultra_RL2.tmd_noMg, whose duplex equilibrated to ~16 A on the
    PI/WED docking CV (native ~4 A) with the lobes already open (REC1-RuvC COM
    ~46 A, 9CJH 46.7). Counts depend on the docking bin only, so dim 1 gets
    population diversity but no preferred direction:

      - dock < 8 A   : 6   frontier, never visited yet; protect it from
                           collapsing back (survival, as in corner-weighted)
      - 8 - 14 A     : 4   the approach
      - 14 - 22 A    : 2   where the run starts; oversampled by construction
      - >= 22 A      : 1   duplex leaving; catch it, do not reward it
    """
    dock_lo = bounds[0][idx[0]]
    if dock_lo >= 22.0:
        return 1
    if dock_lo >= 14.0:
        return 2
    if dock_lo >= 8.0:
        return 4
    return 6


def policy_dock_slope(idx, bounds, value=None):
    """Climb a steep docking slope: walkers by docking bin only (dim 1 neutral).

    For Ultra_RL2.tmd2_withMg after 37 MAB iterations. Measured over 5908
    parent/child pairs (tau = 200 ps): the docking CV has a restoring drift
    toward the ~7.9 A bulk of +0.18 A/tau at the 6.5-6.9 A frontier (sd 0.18,
    only 2% of children advance > 0.2 A), i.e. ~1 kT per 0.1 A. WE climbs that
    only with bins about one step wide (0.1 A; median |step| 0.15 A) and enough
    walkers per frontier bin that some child advances every tau:

      - dock < 7.0 A   : 8   the frontier and everything beyond it
      - 7.0 - 7.6 A    : 5   just behind the frontier
      - 7.6 - 8.4 A    : 3   the bulk (~7.9 A), oversampled for free
      - 8.4 - 9.6 A    : 2
      - >= 9.6 A       : 1   duplex backing out; keep it visible, do not feed it
    """
    lo = bounds[0][idx[0]]
    if lo >= 9.6:
        return 1
    if lo >= 8.4:
        return 2
    if lo >= 7.6:
        return 3
    if lo >= 7.0:
        return 5
    return 8


POLICIES = {"uniform": policy_uniform, "corner-weighted": policy_corner_weighted,
            "dock-frontier": policy_dock_frontier, "dock-slope": policy_dock_slope}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--west-cfg", required=True)
    ap.add_argument("--policy", default="corner-weighted", choices=sorted(POLICIES))
    ap.add_argument("--value", type=int, default=4, help="count for --policy uniform")
    ap.add_argument("--emit", choices=["yaml", "list", "summary"], default="summary")
    ap.add_argument("--assert-order", action="store_true",
                    help="verify the C-order assumption against WESTPA's own mapper")
    args = ap.parse_args()

    cfg = yaml.unsafe_load(open(args.west_cfg))
    so = cfg["west"]["system"]["system_options"]
    bins = so["bins"]
    if bins.get("type") != "RectilinearBinMapper":
        sys.exit("refusing: bin mapper is %r, not RectilinearBinMapper. A "
                 "non-uniform list is only well-defined for a STATIC mapper "
                 "(see module docstring)." % bins.get("type"))
    bounds = [[float(x) for x in row] for row in bins["boundaries"]]
    shape = [len(b) - 1 for b in bounds]
    nbins = int(np.prod(shape))

    fn = POLICIES[args.policy]
    counts = np.zeros(nbins, dtype=int)
    for flat in range(nbins):
        idx = np.unravel_index(flat, shape)          # C order
        counts[flat] = fn(idx, bounds, args.value)

    if args.assert_order:
        from westpa.core.binning import RectilinearBinMapper
        m = RectilinearBinMapper(bounds)
        if m.nbins != nbins:
            sys.exit("nbins mismatch: computed %d, mapper says %d" % (nbins, m.nbins))
        bad = 0
        for flat in range(nbins):
            idx = np.unravel_index(flat, shape)
            p = np.zeros((1, len(bounds)))
            for dim, i in enumerate(idx):
                lo, hi = bounds[dim][i], bounds[dim][i + 1]
                if np.isfinite(lo) and np.isfinite(hi):
                    p[0, dim] = (lo + hi) / 2.0
                elif np.isfinite(lo):
                    p[0, dim] = lo + 1.0
                elif np.isfinite(hi):
                    p[0, dim] = hi - 1.0
                else:
                    p[0, dim] = 0.0
            if int(m.assign(p)[0]) != flat:
                bad += 1
        if bad:
            sys.exit("FLATTENING ORDER MISMATCH in %d/%d cells -- do NOT use "
                     "this output" % (bad, nbins))
        print("# flattening order verified against RectilinearBinMapper: "
              "%d/%d cells OK" % (nbins, nbins), file=sys.stderr)

    if args.emit == "yaml":
        # One YAML line per dim-0 bin, so the block stays readable and each row
        # is visibly the minor dimension. YAML flow sequences may span lines.
        print("      # generated by westpa_scripts/make_target_counts.py"
              " --policy %s" % args.policy)
        print("      # grid %s = %d bins; regenerate after ANY boundary change."
              % (" x ".join(map(str, shape)), nbins))
        print("      bin_target_counts: [")
        row_len = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        for i in range(0, nbins, row_len):
            row = ", ".join("%d" % x for x in counts[i:i + row_len])
            tail = "," if i + row_len < nbins else ""
            print("        %s%s   # dim0 bin %d" % (row, tail, i // row_len))
        print("      ]")
    elif args.emit == "list":
        print(" ".join(str(int(x)) for x in counts))
    else:
        import collections
        print("west.cfg      : %s" % args.west_cfg)
        print("grid          : %s = %d bins" % (" x ".join(map(str, shape)), nbins))
        print("policy        : %s" % args.policy)
        print("total walkers if every bin fills : %d (uniform 4 would be %d)"
              % (counts.sum(), 4 * nbins))
        print("distribution  :")
        for v, n in sorted(collections.Counter(counts.tolist()).items()):
            print("   %d walkers/bin : %4d bins" % (v, n))


if __name__ == "__main__":
    main()
