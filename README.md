# Parallelizable Gaussian Accelerated Molecular Dynamics (ParGaMD)

> **Local-workstation fork**: this fork has been adapted to run on a
> single Linux box with NVIDIA GPUs (no SLURM, no module loads). For
> the local-execution story — launcher, MPS configuration, NaN
> robustness, system swapping, troubleshooting — see
> **[WORKSTATION.md](WORKSTATION.md)**. The original cluster
> launchers are preserved in `_tacc_original/` for reference. The
> theory and reweighting documentation below is unchanged from
> upstream.

A hybrid enhanced sampling method integrating Gaussian Accelerated Molecular Dynamics (GaMD) with the Weighted Ensemble (WE) framework for efficient multi-GPU parallelization.

## Overview

ParGaMD leverages the accelerated sampling capabilities of GaMD and combines them with the multi-GPU parallelization framework of the Weighted Ensemble method. This integration enables:

- **Super-Linear GPU scaling** for molecular dynamics simulations
- **Guided exploration** along user-defined collective variables (CVs)
- **Enhanced barrier crossing** through GaMD's harmonic boost potential
- **Rigorous free energy recovery** via established reweighting protocols

For theoretical details, please refer to:
> Siddharth Sonti, Anugraha Thyagatur , Hung-Yu Wan, et al. Accelerating free energy exploration using parallelizable Gaussian accelerated molecular dynamics (ParGaMD). ChemRxiv. 28 May 2025.
DOI: https://doi.org/10.26434/chemrxiv-2025-rr5v9

---

## Table of Contents

1. [Installation](#installation)
2. [Running ParGaMD with OpenMM](#running-pargamd-with-openmm)
3. [Reweighting Methods](#reweighting-methods)
4. [Usage Examples](#usage-examples)
5. [Output Files](#output-files)
6. [Citation](#citation)

---

## Installation

### Prerequisites

- Python ≥ 3.8
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- CUDA-compatible GPU(s)

### Step 1: Clone this Repository

```bash
git clone https://github.com/anugrahat/ParGaMD_MPS_TACC.git
cd ParGaMD_MPS_TACC
```

> **Important:** Use this repository rather than the standard `gamd-openmm` package, as modifications have been made to ensure compatibility with the WESTPA framework for ParGaMD simulations.

### Step 2: Install Dependencies

```bash
conda create -n pargamd python=3.9
conda activate pargamd
conda install -c conda-forge openmm mdtraj ambertools
pip install westpa numpy matplotlib
```

---

## Running ParGaMD with OpenMM

ParGaMD simulations follow a two-phase protocol: (1) Conventional GaMD equilibration run and (2) ParGaMD production with WESTPA.

### Phase 1: GaMD Parameter Equilibration

Before initiating ParGaMD, run a short GaMD simulation (default: 4 ns) to obtain the finalized boost potential parameters (*E*, *V*<sub>max</sub>, *V*<sub>min</sub>, *k*).

```bash
gamdRunner xml input.xml or sbatch common_files/gamd_prerun.sh
```

The equilibration protocol comprises:
1. Conventional MD preparatory stage
2. Conventional MD stage
3. GaMD pre-equilibration stage
4. GaMD equilibration stage

Upon completion, the GaMD parameters are written to output files and serve as input for the ParGaMD production phase.

### Phase 2: ParGaMD Production with WESTPA

#### Configuration

1. **Define collective variables (CVs):** Specify the reaction coordinates for bin partitioning.

2. **Set WE parameters:**
   - Resampling time τ (recommended: 100 ps – 1 ns depending on system size)
   - Target walkers per bin *n*<sub>w</sub> (typical: 4–6)
   - Bin boundaries along each CV dimension

3. **Prepare WESTPA configuration files** (`west.cfg`, `env.sh`, propagator scripts).

#### Running with NVIDIA Multi-Process Service (MPS)

For optimal GPU utilization, enable MPS to run multiple WESTPA segments per GPU:

```bash
# Start MPS daemon (execute once per job) if using MPS (you might need to check if MPS enabled in your cluster or org)
nvidia-cuda-mps-control -d

run run_WE.sh
```

MPS enables concurrent kernel execution from multiple OpenMM instances on a single GPU, yielding approximately **4-fold throughput improvement** compared to single-segment-per-GPU execution.

#### Key Modifications in This Repository

Two critical modifications ensure ParGaMD compatibility with WESTPA:

1. **Stochastic Independence:** Each WE iteration creates a new GaMD integrator and simulation context with a unique random seed, ensuring walkers evolve stochastically.

2. **Segment Duration Control:** The `<extension-steps>` XML attribute specifies the exact number of MD steps per WE segment, enabling precise resampling time τ.

---

## Reweighting Methods

ParGaMD applies a harmonic boost potential Δ*V*(**r**) when the system potential *V*(**r**) falls below a threshold *E*:

$$\Delta V(\mathbf{r}) = \frac{k}{2}(E - V(\mathbf{r}))^2$$

Recovery of the unbiased free energy landscape requires reweighting to account for both the GaMD boost potential and WE trajectory weights.

### Method 1: Maclaurin Series Expansion (`reweigh.py`)

The Maclaurin series method approximates the exponential Boltzmann factor directly:

$$e^{\beta \Delta V} \approx \sum_{k=0}^{n} \frac{(\beta \Delta V)^k}{k!}$$

where β = 1/(*k*<sub>B</sub>*T*) and *n* is the expansion order (default: 10).

**Combined reweighting with WE weights:**

For each frame *i*, the total reweighting factor is:

$$w_{\text{total},i} = w_{\text{WE},i} \times \sum_{k=0}^{n} \frac{(\beta \Delta V_i)^k}{k!}$$

The reweighted histogram is constructed as:

$$H(A_j) = \sum_{i \in \text{bin } j} w_{\text{total},i}$$

and the potential of mean force (PMF) is obtained:

$$F(A_j) = -k_B T \ln H(A_j) + F_0$$

**When to use:** The Maclaurin series is recommended when the boost potential distribution deviates significantly from Gaussian behavior or exhibits broad variance, as observed in chignolin and PPARα systems.

#### Usage

```bash
python reweigh.py \
    --input merged_data.dat \
    --order 10 \
    --T 300 \
    --discX 0.5 \
    --discY 0.5 \
    --Xdim 0 10 \
    --Ydim 0 12 \
    --Emax 10.0
```

**Input file format** (4 columns, whitespace-delimited):
```
# CV1    CV2    DeltaV(kcal/mol)    WE_weight
0.5      1.2    3.45                0.00125
...
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input data file | Required |
| `--order` | Maclaurin expansion order | 10 |
| `--T` | Temperature (K) | 300 |
| `--discX` | Bin width in X dimension | 0.5 |
| `--discY` | Bin width in Y dimension | 0.5 |
| `--Xdim` | X range: Xmin Xmax | Auto |
| `--Ydim` | Y range: Ymin Ymax | Auto |
| `--Emax` | Maximum free energy cutoff (kcal/mol) | 8.0 |

---

### Method 2: Cumulant Expansion (`reweigh_CE.py`)

When the boost potential follows a near-Gaussian distribution, the cumulant expansion to second (or third) order provides accurate and efficient reweighting.

**Cumulant expansion:**

$$\langle e^{\beta \Delta V} \rangle_j \approx \exp(C_1 + C_2 + C_3)$$

where the cumulants are defined as:

$$C_1 = \beta \langle \Delta V \rangle_j$$

$$C_2 = \frac{1}{2} \beta^2 \sigma^2_{\Delta V,j}$$

$$C_3 = \frac{1}{6} \beta^3 \left( \langle \Delta V^3 \rangle_j - 3\langle \Delta V^2 \rangle_j \langle \Delta V \rangle_j + 2\langle \Delta V \rangle_j^3 \right)$$

Here, ⟨ΔV⟩<sub>j</sub> is the mean boost potential in bin *j*, and σ²<sub>ΔV,j</sub> = ⟨ΔV²⟩<sub>j</sub> − ⟨ΔV⟩²<sub>j</sub> is the variance.

**PMF with cumulant expansion:**

$$F(A_j) = -k_B T \ln P^*(A_j) - C_1 - C_2 + F_0$$

**Integration with WE weights:**

The WE weights are incorporated by computing weighted averages within each bin:

$$\langle \Delta V \rangle_j = \frac{\sum_{i \in \text{bin } j} w_{\text{WE},i} \cdot \Delta V_i}{\sum_{i \in \text{bin } j} w_{\text{WE},i}}$$

**When to use:** The cumulant expansion is recommended when the boost potential distribution exhibits near-Gaussian behavior with low anharmonicity, as observed in the α-synuclein–PAL system.

#### Usage

**1D Reweighting:**

```bash
python reweigh_CE.py \
    -input data_1D.dat \
    -job amdweight_CE_WE \
    -T 300 \
    -disc 0.2 \
    -Xdim 0 10 \
    -cutoff 5 \
    -Emax 8
```

**2D Reweighting:**

```bash
python reweigh_CE.py \
    -input data_2D.dat \
    -job amdweight_CE_WE_2D \
    -T 300 \
    -discX 0.2 \
    -discY 0.2 \
    -Xdim 0 10 \
    -Ydim -2 2 \
    -cutoff 10 \
    -Emax 8
```

**Input file format:**

For 1D: `RC  DeltaV  WE_weight`

For 2D: `CV1  CV2  DeltaV  WE_weight`

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-input` | Input data file | Required |
| `-job` | Job type: `amdweight_CE_WE` (1D) or `amdweight_CE_WE_2D` (2D) | Required |
| `-T` | Temperature (K) | 300 |
| `-disc` | Bin width for 1D | 0.2 |
| `-discX`, `-discY` | Bin widths for 2D | 0.2 |
| `-Xdim`, `-Ydim` | Dimension ranges | Auto |
| `-cutoff` | Minimum total weight per bin for cumulant calculation | 10 |
| `-Emax` | Maximum free energy cutoff (kcal/mol) | 8 |

---

### Selection of Reweighting Method

| System Characteristics | Recommended Method |
|------------------------|-------------------|
| Broad ΔV distribution, high variance | Maclaurin series (`reweigh.py`) |
| Near-Gaussian ΔV distribution | Cumulant expansion (`reweigh_CE.py`) |
| Uncertain distribution shape | Compare both; assess PMF convergence |

---

## Usage Examples

### Example: Chignolin Folding/Unfolding

```bash
# 1. Equilibrate GaMD parameters (4 ns)
gamdRunner xml chignolin_equil.xml

# 2. Run ParGaMD with WESTPA (using MPS on 12 GPUs, 8 segments/GPU)
nvidia-cuda-mps-control -d
w_run --work-manager processes --n-workers 96

# 3. Extract trajectory data (CV1=RMSD, CV2=Rg, dV, WE_weight)
python extract_pargamd_data.py --west-h5 west.h5 --output merged_data.dat

# 4. Reweight using Maclaurin series
python reweigh.py --input merged_data.dat --order 10 --T 300 \
    --discX 0.5 --discY 0.5 --Xdim 0 10 --Ydim 3 10 --Emax 8
```

---

## Output Files

### Maclaurin Series (`reweigh.py`)

- `pmf-<input>.xvg`: 2D PMF file with columns `X_center  Y_center  PMF(kcal/mol)`

### Cumulant Expansion (`reweigh_CE.py`)

- `pmf_c1.xvg` / `pmf_c1_2D.xvg`: PMF with 1st-order cumulant correction
- `pmf_c2.xvg` / `pmf_c2_2D.xvg`: PMF with 2nd-order cumulant correction
- `pmf_c3.xvg` / `pmf_c3_2D.xvg`: PMF with 3rd-order cumulant correction



<img width="1006" height="486" alt="image" src="https://github.com/user-attachments/assets/31893eb2-889a-4022-b60e-cc02054a3e07" />

---

## Citation

If you use this software, please cite:

```bibtex
@article{sonti2025pargamd,
  title={Accelerating free energy exploration using parallelizable Gaussian accelerated molecular dynamics (ParGaMD)},
  author={Sonti, Siddharth and Thyagatur, Anugraha and Wan, Hung-Yu and Hamelynck, Maxen and Faller, Roland and Ahn, Surl-Hee},
  journal={https://doi.org/10.26434/chemrxiv-2025-rr5v9},
  year={2025},
  publisher={ChemRxiv: chemistry preprints}
}
```

---

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
