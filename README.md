# Hamiltonian-Informed Quantum Machine Learning

This repository contains the code and data for the paper:
"Electronic-Structure-Informed Many-Body Encoding for Quantum Machine Learning"



## Goal

Predict molecular correlation energies, $\Delta E = E_{\mathrm{FCI}} - E_{\mathrm{RHF}}$, using quantum machine learning. Electronic-density descriptors encode molecular structure in parameterized quantum circuits; eight many-body (MB) models compare density products and reduced-density-matrix features of increasing order. Energies are expressed in Hartree.

## Structure

```text
paper_source_v5/
├── registry.py            # Molecules, AO ordering, and default paths
├── initialdata.py         # RHF equilibrium, harmonic scans, and frozen-core FCI
├── descriptor.py          # Native MP2 density matrices and Löwdin transformations
├── data.py                # Descriptor loading, AO selection, and data splitting
├── features.py            # Density products and permutation-cycle sums
├── preprocessing.py       # Training statistics and model definitions
├── models.py              # Quantum encodings, variational circuit, and readout
├── train.py               # Optimization and checkpoint selection
├── checkpoints.py         # Saved-checkpoint validation and inference
├── bond_extrapolation.py  # Bond-length extrapolation with fixed parameters
├── prepare.py             # Input validation
└── requirements.txt       # Pinned dependencies
```

## Algorithm / Technical Detail

HF, H2O, NH3, N2, CO, and H2O2 are treated as neutral singlets in STO-3G. RHF equilibrium geometries and harmonic curvatures define 30 equally spaced bond-stretch geometries spanning the harmonic $n=1$ interval. PySCF evaluates RHF and frozen-core FCI energies, freezing one canonical core orbital for HF/H2O/NH3 and two for N2/CO/H2O2. Full-AO Löwdin transformations, $P_L=S^{1/2}P_{\mathrm{AO}}S^{1/2}$ and $F_L=S^{-1/2}F_{\mathrm{AO}}S^{-1/2}$, precede exclusion of heavy-atom 1s AOs. Native unrelaxed RMP2 (`frozen=0`) supplies the spin-summed two-particle density: all four indices are transformed from MO to AO to Löwdin coordinates before extracting $\Gamma_L[I,I,J,J]$ for selected pairs $i<j$. The $\Gamma^0$ hierarchy is constructed from the RHF density; Fock matrices are archived but are not model inputs.

| Model | Density encoding |
| --- | --- |
| `MB1` | Standardized RHF diagonal density only. |
| `MB2` | Adds diagonal and off-diagonal products of separately standardized density entries. |
| `MB3`, `MB4` | Extend `MB2` with independently weighted density-cycle groups through order three or four. |
| `MB2_gamm0` | Adds standardized $\Gamma^0_2=P_{ii}P_{jj}-\tfrac12 P_{ij}P_{ji}$ from raw selected $P_L$. |
| `MB2_gamma` | Adds standardized native MP2 $\Gamma_L[I,J,I,J]$. |
| `MB3_gamm0`, `MB4_gamma0` | Add separately standardized RHF-derived $\Gamma^0_k$ for orders two through three or four. |

- Circuits: one qubit per selected AO; $(\pi/2)\tanh(u)$ encoding angles with `RY`, `IsingZZ`, and `MultiRZ` gates; two layers of shared `RY` rotations and CNOT rings. A linear readout uses local Pauli-Z expectations and their first three moments. PennyLane `default.qubit` uses exact expectations and PyTorch backpropagation.
- Preprocessing: 15 odd-indexed geometries (`001,003,...,029`) for training and 15 even-indexed geometries for testing; all normalization statistics use training data only.
- Training defaults: seeds 0–63, 1000 epochs, Adam, learning rate 0.02, batch size 8, and mean squared error loss. Parameters use float64 Gaussian initialization with standard deviation 0.05; the readout bias starts at the training-target mean.
- Evaluation: select the epoch with minimum full-training MAE, then evaluate held-out predictions. MAE, RMSE, maximum absolute error, and signed-error amplitude are reported in Hartree and mHa.

## Example Command

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
python -B -m paper_source_v5.train \
  --molecules HF --models MB2_gamma --seeds 0
```

## Requirement
```bash
numpy==2.5.1
scipy==1.18.0
pyscf==2.14.0
torch==2.13.0
pennylane==0.45.1
matplotlib==3.11.1
pandas==3.0.5
```
