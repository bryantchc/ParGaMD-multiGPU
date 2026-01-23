#!/usr/bin/env python

import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

########################################################################
# Example Weighted Ensemble + GaMD reweighting script using
# Cumulative Expansion (CE) approach, extended for:
#   - 1D or 2D reaction coordinates
#
# Original Author: [Your Name]
# Extended for 2D by: [Your Name or collaborator]
#
# Inspired by PyReweighting (Miao, Sinko, et al.)
#
# Requirements:
#   numpy, matplotlib
#
# Example Usage (1D):
#   python amd_reweighing_WE_2D.py \
#       -input your_data_1D.dat \
#       -job amdweight_CE_WE \
#       -T 300 -disc 0.2 -Xdim 0 10 -cutoff 5 -Emax 8
#
# Example Usage (2D):
#   python amd_reweighing_WE_2D.py \
#       -input your_data_2D.dat \
#       -job amdweight_CE_WE_2D \
#       -T 300 \
#       -Xdim 0 10 -Ydim -2 2 \
#       -discX 0.2 -discY 0.2 \
#       -cutoff 10 -Emax 8
########################################################################

def cmdlineparse():
    parser = ArgumentParser(description="Reweight Weighted Ensemble + GaMD data via Cumulative Expansion (1D or 2D).")
    parser.add_argument("-input", dest="input", required=True,
                        help="Input text file. For 1D: col0=RC, col1=dV, col2=w_WE. "
                             "For 2D: col0=cv1, col1=cv2, col2=dV, col3=w_WE")
    parser.add_argument("-job", dest="job", required=True,
                        help="Job type: 'amdweight_CE_WE' (1D) or 'amdweight_CE_WE_2D' (2D).")
    parser.add_argument("-T", dest="T", required=False, default=300.0, type=float,
                        help="Temperature (K). Default=300")
    parser.add_argument("-Xdim", dest="Xdim", required=False, nargs=2, default=None,
                        help="Xmin Xmax for binning the 1D RC or the first dimension in 2D.")
    parser.add_argument("-Ydim", dest="Ydim", required=False, nargs=2, default=None,
                        help="Ymin Ymax for the second dimension (only required for 2D jobs).")
    parser.add_argument("-disc", dest="disc", required=False, default=0.2, type=float,
                        help="Bin width for the 1D RC. Default=0.2. (1D only)")
    parser.add_argument("-discX", dest="discX", required=False, default=0.2, type=float,
                        help="Bin width in X dimension (2D job only). Default=0.2")
    parser.add_argument("-discY", dest="discY", required=False, default=0.2, type=float,
                        help="Bin width in Y dimension (2D job only). Default=0.2")
    parser.add_argument("-cutoff", dest="cutoff", required=False, default=10.0, type=float,
                        help="Minimum total weight in a bin to compute cumulants. Default=10")
    parser.add_argument("-Emax", dest="Emax", required=False, default=8.0, type=float,
                        help="Max free energy cutoff to treat as 'infinite'. Default=8 kcal/mol")
    args = parser.parse_args()
    return args

def main():
    args = cmdlineparse()
    T = float(args.T)
    beta = 1.0/(0.001987 * T)
    
    # Decide if it's 1D or 2D reweighting
    if args.job == "amdweight_CE_WE":
        # 1D data must have columns: RC, dV, w_WE
        data_in = np.loadtxt(args.input)
        rc  = data_in[:,0]
        dV  = data_in[:,1]
        wWE = data_in[:,2]

        # If user gave Xdim, use them, else auto from data
        if args.Xdim:
            x_min = float(args.Xdim[0])
            x_max = float(args.Xdim[1])
        else:
            x_min = np.min(rc)
            x_max = np.max(rc)
        discX = float(args.disc)

        # Make bin edges
        binsX = np.arange(x_min, x_max + discX, discX)

        # Reweight using Weighted Ensemble + GaMD Cumulative Expansion in 1D
        hist, newedges, c1, c2, c3 = reweight_CE_WE_1D(
            rc, dV, wWE, binsX, discX, args.cutoff, T
        )
        
        pmf_raw = histogram_to_pmf_1D(hist, T)
        # c1, c2, c3 are dimensionless expansions
        pmf_c1 = pmf_raw + (-1.0/beta)*c1
        pmf_c2 = pmf_raw + (-1.0/beta)*(c1 + c2)
        pmf_c3 = pmf_raw + (-1.0/beta)*(c1 + c2 + c3)

        # Normalize and saturate
        pmf_c1 = normalize_pmf(pmf_c1, args.Emax)
        pmf_c2 = normalize_pmf(pmf_c2, args.Emax)
        pmf_c3 = normalize_pmf(pmf_c3, args.Emax)

        # Output 1D
        output_pmf_1D("pmf_c1.xvg", pmf_c1, newedges)
        output_pmf_1D("pmf_c2.xvg", pmf_c2, newedges)
        output_pmf_1D("pmf_c3.xvg", pmf_c3, newedges)

        print("Finished 1D CE reweighting with Weighted Ensemble + GaMD.")

    elif args.job == "amdweight_CE_WE_2D":
        # 2D data must have columns: cv1, cv2, dV, w_WE
        data_in = np.loadtxt(args.input)
        cv1 = data_in[:,0]
        cv2 = data_in[:,1]
        dV  = data_in[:,2]
        wWE = data_in[:,3]
        #wWE = np.ones_like(cv1)
        wWE /= wWE.sum() 

        # If user gave Xdim, Ydim, use them; else auto
        if not args.Xdim:
            x_min = np.min(cv1)
            x_max = np.max(cv1)
        else:
            x_min = float(args.Xdim[0])
            x_max = float(args.Xdim[1])
        if not args.Ydim:
            y_min = np.min(cv2)
            y_max = np.max(cv2)
        else:
            y_min = float(args.Ydim[0])
            y_max = float(args.Ydim[1])

        discX = float(args.discX)
        discY = float(args.discY)

        binsX = np.arange(x_min, x_max + discX, discX)
        binsY = np.arange(y_min, y_max + discY, discY)

        # Reweight in 2D
        hist2D, edgesX, edgesY, c1_2D, c2_2D, c3_2D = reweight_CE_WE_2D(
            cv1, cv2, dV, wWE, binsX, binsY, discX, discY, args.cutoff, T
        )

        pmf_raw_2D = histogram_to_pmf_2D(hist2D, T)
        # Apply expansions
        pmf_c1_2D = pmf_raw_2D + (-1.0/beta)*c1_2D
        pmf_c2_2D = pmf_raw_2D + (-1.0/beta)*(c1_2D + c2_2D)
        pmf_c3_2D = pmf_raw_2D + (-1.0/beta)*(c1_2D + c2_2D + c3_2D)

        # Normalize each
        pmf_c1_2D = normalize_pmf_2D(pmf_c1_2D, args.Emax)
        pmf_c2_2D = normalize_pmf_2D(pmf_c2_2D, args.Emax)
        pmf_c3_2D = normalize_pmf_2D(pmf_c3_2D, args.Emax)

        # Output 2D
        output_pmf_2D("pmf_c1_2D.xvg", pmf_c1_2D, edgesX, edgesY)
        output_pmf_2D("pmf_c2_2D.xvg", pmf_c2_2D, edgesX, edgesY)
        output_pmf_2D("pmf_c3_2D.xvg", pmf_c3_2D, edgesX, edgesY)

        print("Finished 2D CE reweighting with Weighted Ensemble + GaMD.")

    else:
        print("ERROR: job type not recognized. Use 'amdweight_CE_WE' (1D) or 'amdweight_CE_WE_2D' (2D).")

##############################
# 1D reweighting subroutines
##############################
def reweight_CE_WE_1D(rc, dV, wWE, binsX, discX, cutoff, T):
    """
    Compute the Weighted Ensemble + GaMD Cumulative Expansion in each 1D bin.
    rc   : reaction coordinate (1D)
    dV   : GaMD boost potential (kcal/mol)
    wWE  : Weighted Ensemble weights
    binsX: bin edges in 1D
    discX: bin width in X
    cutoff: min total weight to compute cumulants
    T    : temperature (K)
    """
    beta = 1.0/(0.001987*T)
    nbins = len(binsX) - 1

    sum_w     = np.zeros(nbins)
    sum_w_dV  = np.zeros(nbins)
    sum_w_dV2 = np.zeros(nbins)
    sum_w_dV3 = np.zeros(nbins)

    # Bin assignment & accumulation
    for i in range(len(rc)):
        bx = int((rc[i] - binsX[0])//discX)
        if bx < 0 or bx >= nbins:
            continue
        w = wWE[i]
        dv_i = dV[i]
        sum_w[bx]     += w
        sum_w_dV[bx]  += w*dv_i
        sum_w_dV2[bx] += w*(dv_i**2)
        sum_w_dV3[bx] += w*(dv_i**3)

    # Now compute c1, c2, c3 (dimensionless expansions) in each bin
    c1 = np.zeros(nbins)
    c2 = np.zeros(nbins)
    c3 = np.zeros(nbins)

    for j in range(nbins):
        if sum_w[j] >= cutoff:
            mean_dV  = sum_w_dV[j]/sum_w[j]
            mean_dV2 = sum_w_dV2[j]/sum_w[j]
            mean_dV3 = sum_w_dV3[j]/sum_w[j]

            var_dV   = mean_dV2 - mean_dV**2
            # 1st cumulant: c1 = beta * <dV>
            c1[j] = beta * mean_dV
            # 2nd cumulant: c2 = 1/2 * beta^2 * variance
            c2[j] = 0.5 * (beta**2) * var_dV
            # 3rd cumulant: c3 = (1/6)*beta^3*(<dV^3>-3<dV^2><dV>+2<dV>^3)
            c3[j] = (1.0/6.0)* (beta**3)* (mean_dV3 - 3.0*mean_dV2*mean_dV + 2.0*(mean_dV**3))
        else:
            # not enough weight => skip
            c1[j] = 0.0
            c2[j] = 0.0
            c3[j] = 0.0

    # "Raw" histogram is sum_w
    hist, newedges = np.histogram(rc, bins=binsX, weights=wWE)
    return hist, newedges, c1, c2, c3

def histogram_to_pmf_1D(hist, T):
    """
    Convert a weighted 1D histogram -> free energy in kcal/mol.
    We'll do a log, ignoring normalization factor except as
    an additive constant. Then we shift in normalize_pmf().
    """
    kT = 0.001987*T
    # avoid zero
    hist_safe = hist + 1e-30
    # pmf_temp = -kT ln(hist_bin)
    # We'll do an inverted form: pmf_temp = kT * ln(hist_bin),
    # then subtract from max to get free energies
    pmf_temp = kT * np.log(hist_safe)
    # flip sign so the most-populated bin is near 0
    pmf = np.max(pmf_temp) - pmf_temp
    return pmf

def output_pmf_1D(filename, pmf, bin_edges):
    """
    Write a simple 2-column file: RC_center  PMF
    """
    with open(filename, 'w') as f:
        f.write("# RC   PMF(kcal/mol)\n")
        for i in range(len(pmf)):
            rc_mid = 0.5*(bin_edges[i] + bin_edges[i+1])
            f.write(f"{rc_mid:12.5f}  {pmf[i]:12.5f}\n")
    print(f"1D PMF written to {filename}")

##############################
# 2D reweighting subroutines
##############################
def reweight_CE_WE_2D(cv1, cv2, dV, wWE, binsX, binsY, discX, discY, cutoff, T):
    """
    Compute Weighted Ensemble + GaMD Cumulative Expansion in 2D bins.
    cv1, cv2 : coordinates
    dV       : GaMD boost potential
    wWE      : Weighted Ensemble weights
    binsX    : array of bin edges in X
    binsY    : array of bin edges in Y
    discX, discY : bin widths
    cutoff   : min total weight for cumulant calculation
    T        : temperature
    """
    beta = 1.0/(0.001987*T)
    nx = len(binsX) - 1
    ny = len(binsY) - 1

    sum_w     = np.zeros((nx, ny))
    sum_w_dV  = np.zeros((nx, ny))
    sum_w_dV2 = np.zeros((nx, ny))
    sum_w_dV3 = np.zeros((nx, ny))

    # Bin assignment
    for i in range(len(cv1)):
        bx = int((cv1[i] - binsX[0]) // discX)
        by = int((cv2[i] - binsY[0]) // discY)
        if bx < 0 or bx >= nx: 
            continue
        if by < 0 or by >= ny:
            continue
        w = wWE[i]
        dv_i = dV[i]
        sum_w[bx, by]     += w
        sum_w_dV[bx, by]  += w*dv_i
        sum_w_dV2[bx, by] += w*(dv_i**2)
        sum_w_dV3[bx, by] += w*(dv_i**3)

    # c1, c2, c3 are now 2D arrays
    c1_2D = np.zeros((nx, ny))
    c2_2D = np.zeros((nx, ny))
    c3_2D = np.zeros((nx, ny))

    for ix in range(nx):
        for iy in range(ny):
            if sum_w[ix, iy] >= cutoff:
                mean_dV  = sum_w_dV[ix, iy]/sum_w[ix, iy]
                mean_dV2 = sum_w_dV2[ix, iy]/sum_w[ix, iy]
                mean_dV3 = sum_w_dV3[ix, iy]/sum_w[ix, iy]

                var_dV   = mean_dV2 - mean_dV**2
                c1_2D[ix, iy] = beta * mean_dV
                c2_2D[ix, iy] = 0.5 * (beta**2) * var_dV
                c3_2D[ix, iy] = (1.0/6.0)* (beta**3)* (
                    mean_dV3 - 3.0*mean_dV2*mean_dV + 2.0*(mean_dV**3)
                )
            else:
                c1_2D[ix, iy] = 0.0
                c2_2D[ix, iy] = 0.0
                c3_2D[ix, iy] = 0.0

    # 2D "raw" histogram from sum_w
    hist2D, edgesX, edgesY = np.histogram2d(cv1, cv2, bins=[binsX, binsY], weights=wWE)
    return hist2D, edgesX, edgesY, c1_2D, c2_2D, c3_2D

def histogram_to_pmf_2D(hist2D, T):
    """
    Convert a weighted 2D histogram -> free energy (kcal/mol).
    Similar approach as 1D: pmf = max(...) - kT ln(hist).
    """
    kT = 0.001987 * T
    hist_safe = hist2D + 1e-30
    pmf_temp = kT * np.log(hist_safe)
    pmf_2D = np.max(pmf_temp) - pmf_temp
    return pmf_2D

##############################
# Normalization and output
##############################
def normalize_pmf(pmf_1D, Emax):
    """
    Shift so min(pmf) = 0 and saturate large energies (1D).
    """
    finite_mask = np.isfinite(pmf_1D)
    if np.any(finite_mask):
        pmf_min = np.min(pmf_1D[finite_mask])
    else:
        pmf_min = 0.0
    pmf_1D -= pmf_min
    pmf_1D[~finite_mask] = Emax
    pmf_1D[pmf_1D > Emax] = Emax
    return pmf_1D

def normalize_pmf_2D(pmf_2D, Emax):
    """
    Shift so min(pmf) = 0 and saturate large energies (2D).
    """
    finite_mask = np.isfinite(pmf_2D)
    if np.any(finite_mask):
        pmf_min = np.min(pmf_2D[finite_mask])
    else:
        pmf_min = 0.0
    pmf_2D -= pmf_min
    pmf_2D[~finite_mask] = Emax
    pmf_2D[pmf_2D > Emax] = Emax
    return pmf_2D

def output_pmf_2D(filename, pmf2D, edgesX, edgesY):
    """
    Write a simple 3-column file for each bin: X_center  Y_center  PMF(kcal/mol).
    """
    with open(filename, 'w') as f:
        f.write("# X   Y   PMF(kcal/mol)\n")
        nx = len(edgesX) - 1
        ny = len(edgesY) - 1
        for ix in range(nx):
            x_mid = 0.5*(edgesX[ix] + edgesX[ix+1])
            for iy in range(ny):
                y_mid = 0.5*(edgesY[iy] + edgesY[iy+1])
                val   = pmf2D[ix, iy]
                f.write(f"{x_mid:12.5f}  {y_mid:12.5f}  {val:12.5f}\n")
            f.write("\n")  # blank line to separate rows (optional)
    print(f"2D PMF written to {filename}")

if __name__ == '__main__':
    main()
