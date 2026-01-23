#!/usr/bin/env python3
"""
maclaurin_we_gamd_2D.py
=======================

A standalone script for reweighting Gaussian Accelerated MD (GaMD) data 
using a Maclaurin-series expansion *and* Weighted Ensemble (WE) segment weights 
to produce a 2D Potential of Mean Force (PMF).

Usage example:
  python maclaurin_we_gamd_2D.py \
    --input merged.dat \
    --order 10 \
    --T 300 \
    --discX 0.2 \
    --discY 0.2 \
    --Xdim 0 10 \
    --Ydim 0 12 \
    --Emax 10.0

Where "merged.dat" has columns:
  col0 = x-coordinate
  col1 = y-coordinate
  col2 = GaMD boost potential (DeltaV)
  col3 = Weighted Ensemble weight (w_WE)

Output:
 - A file named "pmf-merged.dat.xvg" containing columns [x_center, y_center, PMF(kcal/mol)].
"""

import numpy as np
import math
import sys
from argparse import ArgumentParser

k_B = 0.001987  # Boltzmann constant in kcal/(mol*K)

def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1) Load data from file: shape => (N,4)
    #    [ x, y, DeltaV, WE_weight ]
    # -------------------------------------------------------------------------
    data = np.loadtxt(args.input)
    if data.shape[1] < 4:
        sys.exit("ERROR: input file must have at least 4 columns: x, y, dV, WE_weight")

    xvals   = data[:,0]
    yvals   = data[:,1]
    dV      = data[:,2]
    we_w    = data[:,3]

    N = len(xvals)
    print(f"Loaded {N} snapshots from {args.input}")

    # -------------------------------------------------------------------------
    # 2) Maclaurin expansion of e^{beta * dV}
    # -------------------------------------------------------------------------
    T = args.T
    beta = 1.0/(k_B * T)
    order = args.order

    # e^{beta * dV} approx = sum_{k=0}^order ( (beta*dV)^k / k! )
    # We'll build it for each snapshot
    mc_weight = np.zeros(N)
    beta_dV   = beta * dV

    for k in range(order+1):
        # (beta_dV)^k / k!
        term = np.power(beta_dV, k) / math.factorial(k)
        mc_weight += term  # vector addition

    # -------------------------------------------------------------------------
    # 3) Combine with WE weight
    # -------------------------------------------------------------------------
    combined_weight = we_w * mc_weight
    #combined_weight = 1* mc_weight


    # -------------------------------------------------------------------------
    # 4) Build 2D histogram
    #    We'll define bin edges based on user-provided discX/Y or range
    # -------------------------------------------------------------------------
    binsX = define_bins(xvals, args.discX, args.Xdim)
    binsY = define_bins(yvals, args.discY, args.Ydim)

    hist2d, xedges, yedges = np.histogram2d(
        xvals, yvals,
        bins=(binsX, binsY),
        weights=combined_weight
    )

    # -------------------------------------------------------------------------
    # 5) Convert histogram to Free Energy
    #    add a tiny offset to avoid log(0)
    # -------------------------------------------------------------------------
    hist2d += 1e-15

    # free energy
    fe = k_B*T * np.log(hist2d)
    # invert => shift so min = 0
    fe = np.max(fe) - fe

    # cap at Emax
    emax = args.Emax
    fe[fe > emax] = emax

    # -------------------------------------------------------------------------
    # 6) Write out the 2D PMF as an .xvg file
    # -------------------------------------------------------------------------
    outname = f"pmf-{args.input}.xvg"
    write_pmf2d(outname, fe, xedges, yedges)
    print(f"Saved 2D PMF to {outname}")
    print("DONE.")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Input file with columns: x, y, DeltaV, WE_weight")
    parser.add_argument("--order", type=int, default=10,
                        help="Order of Maclaurin expansion (default=10)")
    parser.add_argument("--T", type=float, default=300.0,
                        help="Temperature in K (default=300)")
    parser.add_argument("--discX", type=float, default=0.5,
                        help="Bin width in X dimension (default=0.5)")
    parser.add_argument("--discY", type=float, default=0.5,
                        help="Bin width in Y dimension (default=0.5)")
    parser.add_argument("--Xdim", nargs=2, type=float, default=None,
                        help="X range: Xmin Xmax")
    parser.add_argument("--Ydim", nargs=2, type=float, default=None,
                        help="Y range: Ymin Ymax")
    parser.add_argument("--Emax", type=float, default=8.0,
                        help="Max free energy cutoff (default=8.0 kcal/mol)")
    return parser.parse_args()

def define_bins(values, disc, xy_range):
    """
    Return bin edges for np.histogram based on user-provided range or data range.
    """
    if xy_range is not None:
        xmin, xmax = xy_range
    else:
        vmin, vmax = np.min(values), np.max(values)
        # round outward to multiples of disc for neat bins
        xmin = disc * (int(vmin/disc) - 1)
        xmax = disc * (int(vmax/disc) + 1)

    nbins = int((xmax - xmin)/disc + 0.9999)
    edges = np.arange(nbins+1)*disc + xmin
    return edges

def write_pmf2d(filename, fe2d, xedges, yedges):
    """
    Write 2D free energy grid to file:
      # X_center  Y_center  FE(kcal/mol)
    """
    nx = len(xedges)-1
    ny = len(yedges)-1

    with open(filename, 'w') as f:
        f.write("# X   Y   PMF(kcal/mol)\n@TYPE xy\n")
        for i in range(nx):
            xcenter = 0.5*(xedges[i]+xedges[i+1])
            for j in range(ny):
                ycenter = 0.5*(yedges[j]+yedges[j+1])
                val = fe2d[i,j]
                f.write(f"{xcenter:12.4f} {ycenter:12.4f} {val:12.4f}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
