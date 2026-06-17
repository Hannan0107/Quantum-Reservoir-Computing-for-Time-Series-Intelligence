# Quantum Reservoir Computing for Time-Series Intelligence


This repository implements a Quantum Reservoir Computing (QRC) framework for financial time-series forecasting, specifically targeting 5-day-ahead rolling log-volatility prediction on equity ETFs (SPY, QQQ, GLD). The QRC model is benchmarked against classical baselines (GARCH, ESN, LSTM) and validated on real IBM Quantum hardware.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
  - [qrc\_core.py](#qrc_corepy)
  - [experiment\_finance\_simulation.py](#experiment_finance_simulationpy)
  - [experiment\_finance\_simulation\_Noise\_Test.py](#experiment_finance_simulation_noise_testpy)
  - [experiment\_finance\_hardware\_SPY.py](#experiment_finance_hardware_spypy)
  - [benchmark\_classical.py](#benchmark_classicalpy)
- [Configuration](#configuration)
- [Key Results](#key-results)
- [Results Files](#results-files)
- [Citation](#citation)

---

## Overview

Quantum Reservoir Computing leverages the natural dynamics of a fixed, randomly-initialised quantum circuit as a non-linear feature extractor. A classical linear readout (RidgeCV) is trained on top of these quantum features to forecast volatility. The reservoir weights are never updated — only the readout layer is trained.

Key contributions of this work:

- A multi-reservoir QRC ensemble with Hankel (delay-embedding) stacking for improved temporal memory
- A full 1,000-combination grid search per ticker over circuit depth, ensemble size, and data horizon (1–10 years)
- Hardware validation on IBM Quantum (`ibm_kingston`) using `EstimatorV2` with full transpilation and observable layout mapping
- A noise robustness sweep across depolarizing error levels 0%–5% via `qiskit-aer`
- Systematic comparison against GARCH(1,1), Echo State Networks (ESN), and LSTM across three metrics: RMSE, QLIKE, and Mincer-Zarnowitz (α, β, R²)

---

## Repository Structure

```
.
├── Code/
│   ├── qrc_core.py                                  # Core QRC module
│   ├── experiment_finance_simulation.py             # Simulation grid search
│   ├── experiment_finance_simulation_Noise_Test.py  # Noise robustness sweep
│   ├── experiment_finance_hardware_SPY.py           # IBM hardware execution
│   └── benchmark_classical.py                       # GARCH / ESN / LSTM baselines
│
└── Results/
    ├── 4K_Configuration_Test_SPY.csv                # Grid results — SPY, 4 qubits
    ├── 4K_Configuration_Test_QQQ.csv                # Grid results — QQQ, 4 qubits
    ├── 4K_Configuration_Test_GLD_Data.csv           # Grid results — GLD, 4 qubits
    ├── 4Qubits1Layer2Reservoir4YearHardware.csv     # IBM hardware results
    ├── Claasical_Test_SPY.csv                       # Classical benchmarks — SPY
    ├── Classical_Test_QQQ.csv                       # Classical benchmarks — QQQ
    ├── Classical_Test_GLD.csv                       # Classical benchmarks — GLD
    └── Noise_Test.csv                               # Noise sweep results
```

---

## Pipeline Architecture

```
Market Data (yfinance)
        │
        ▼
Daily Returns  ──►  Rolling 5-day Log-Volatility Targets
        │
        ▼
Lookback Windows (length = 5)
        │
        ▼
┌──────────────────────────────────────────────┐
│          Quantum Reservoir Ensemble          │
│                                              │
│  Reservoir 1 (seed=42)  ──►  Latent Z_t    │
│  Reservoir 2 (seed=43)  ──►  Latent Z_t    │
│          ...              ──►  ...          │
│  Reservoir N (seed=N+41) ──►  Latent Z_t   │
│                                              │
│  Each circuit: RY encoding + RZ + CNOT      │
│  Observables:  ⟨Z_i⟩ and ⟨Z_i Z_j⟩         │
└──────────────────────────────────────────────┘
        │
        ▼
Hankel Stacking (window = 3)
        │
        ▼
StandardScaler  ──►  RidgeCV (TimeSeriesSplit, 5 folds)
        │
        ▼
Predictions in log-vol space  ──►  exp(·)  ──►  Raw volatility
        │
        ▼
Evaluation: RMSE | QLIKE | Mincer-Zarnowitz (α, β, R²)
```

For `n_qubits = 4` and `n_reservoirs = 2`, the Hankel feature matrix has shape `(T, 2 × 10 × 3) = (T, 60)` — 10 observables (4 single-Z + 6 ZZ) per reservoir, stacked across 3 Hankel delays.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hannan0107/Quantum-Reservoir-Computing-for-Time-Series-Intelligence.git
cd Quantum-Reservoir-Computing-for-Time-Series-Intelligence
```

### 2. Install core dependencies

```bash
pip install qiskit qiskit-aer yfinance scikit-learn statsmodels numpy pandas arch tensorflow
```

For IBM hardware execution, also install:

```bash
pip install qiskit-ibm-runtime
```

### Dependency reference

| Package | Purpose |
|---|---|
| `qiskit` | Quantum circuit construction and statevector simulation |
| `qiskit-aer` | Noisy shot-based simulation (AerSimulator) |
| `qiskit-ibm-runtime` | IBM Quantum hardware access (EstimatorV2) |
| `yfinance` | Market data download |
| `scikit-learn` | RidgeCV readout, StandardScaler, TimeSeriesSplit |
| `statsmodels` | Mincer-Zarnowitz OLS regression |
| `numpy` / `pandas` | Numerical computation and results management |
| `arch` | GARCH(1,1) baseline |
| `tensorflow` | LSTM baseline |

---

## Quick Start

```bash
cd Code

# 1. Run the simulation grid search (SPY, 4 qubits)
python experiment_finance_simulation.py

# 2. Run the noise robustness sweep
#    (set RUN_MODE = "noise_sweep" inside the file first)
python experiment_finance_simulation_Noise_Test.py

# 3. Run classical baselines
python benchmark_classical.py
```

The grid search checkpoints results to `qrc_results_q{n_qubits}_logvol.csv` after every combination and skips already-completed rows on restart, so it is safe to interrupt and resume.

---

## Scripts

### `qrc_core.py`

The central module used by all other scripts. Provides three simulation backends, selectable at construction time:

| Backend | Description |
|---|---|
| `"statevector"` | Exact Qiskit `Statevector` — no shots, no noise (default) |
| `"aer_ideal"` | `AerSimulator` shot-based, zero noise — validates Aer matches statevector |
| `"aer_noise"` | `AerSimulator` + `NoiseModel` — depolarizing gate + readout errors |

**Usage example:**

```python
from qrc_core import QuantumReservoir, prepare_hankel_features

# Exact simulation (default)
reservoir = QuantumReservoir(
    n_qubits=4, n_layers=2, seeds=[42, 43, 44],
    backend="statevector"
)

# With noise (requires qiskit-aer)
noisy_reservoir = QuantumReservoir(
    n_qubits=4, n_layers=1, seeds=[42, 43],
    backend="aer_noise", shots=1024,
    depolar_error=0.01,   # 1% single-qubit gate error
    cx_error=0.02,        # 2% CX gate error
    readout_error=0.01    # 1% measurement bit-flip
)

# Extract Hankel-stacked features
X = prepare_hankel_features(returns, reservoir, lookback=5, window=3)
```

Each reservoir produces `n_qubits + n_qubits*(n_qubits-1)//2` expectation values — for 4 qubits: 4 single-qubit Z observables and 6 two-qubit ZZ observables = 10 features per reservoir.

---

### `experiment_finance_simulation.py`

Runs a full grid search over:

| Axis | Values |
|---|---|
| `data_period` | 1y, 2y, 3y, 4y, 5y, 6y, 7y, 8y, 9y, 10y |
| `n_layers` | 1 – 10 |
| `n_reservoirs` | 1 – 10 |

`n_qubits` is set manually in `CFG` before each run (paper tested: 4, 6, 8, 10). Metrics saved per combination: `rmse`, `qlike`, `mz_alpha`, `mz_beta`, `mz_r2`, `best_alpha`, `train_size`, `test_size`.

Change the ticker manually in `CFG["ticker"]` to run on QQQ or GLD.

---

### `experiment_finance_simulation_Noise_Test.py`

Two run modes, controlled by `RUN_MODE` at the top of the file:

`"grid"` — same grid search as above, for a specific ticker.

`"noise_sweep"` — holds architecture fixed and sweeps depolarizing noise levels across three backends:

```
Backends:       statevector  →  aer_ideal  →  aer_noise
Depolar levels: [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
CX error:       2 × depolar_error  (per each level)
Readout error:  = depolar_error    (per each level)
```

The summary output reports RMSE degradation (%) and R² drop relative to the noiseless statevector baseline.

---

### `experiment_finance_hardware_SPY.py`

Executes the QRC pipeline on real IBM Quantum hardware using `EstimatorV2`. Before running, fill in your credentials:

```python
IBM_TOKEN        = "YOUR_IBM_TOKEN"
IBM_INSTANCE     = "YOUR_CRN_INSTANCE"
IBM_SERVICE_NAME = "open-instance"
IBM_BACKEND_NAME = "ibm_kingston"   # or your available backend
```

The script builds `n_sequences × n_reservoirs` circuits, transpiles them with `optimization_level=1`, applies per-circuit observable layout mapping, and submits a single batched job. It uses the same `SEED_POOL` as the simulation scripts so reservoir weights are identical across both runs.

An optional smoke test function (`smoke_test()`) can be uncommented to verify a single sequence before submitting the full job.

---

### `benchmark_classical.py`

Evaluates three classical baselines across all 10 data periods:

| Model | Details |
|---|---|
| `GARCH(1,1)` | Rolling one-step-ahead forecast via `arch`; output log-transformed before evaluation |
| `ESN(lb=k)` | Echo State Network (300 reservoir units, spectral radius 0.9); lookbacks k ∈ {1, 5, 7, 21} |
| `LSTM(lb=k)` | LSTM (64 units, Dropout 0.2, early stopping); lookbacks k ∈ {5, 7, 21} |

All models predict in log-vol space. RMSE and QLIKE are computed after exponentiating to raw-vol space. Mincer-Zarnowitz is computed in log-vol space — identical to the QRC evaluation protocol, ensuring a fair comparison.

Lookback rationale:

| Lookback | Rationale |
|---|---|
| `lb=1` | ESN single-step (recurrent memory intact; original ESN baseline) |
| `lb=5` | Matches QRC raw lookback window |
| `lb=7` | Matches QRC effective memory (`lookback + hankel_window − 1 = 5 + 3 − 1`) |
| `lb=21` | Standard ~1-month financial window |

---

## Configuration

All simulation scripts share the same core hyperparameters in `CFG`:

```python
CFG = {
    "ticker":           "SPY",   # SPY | QQQ | GLD
    "n_qubits":         4,       # set manually per run (4 | 6 | 8 | 10)
    "lookback":         5,       # input sequence length fed to each reservoir
    "hankel_window":    3,       # Hankel (delay-embedding) stacking window
    "forecast_horizon": 5,       # days ahead to forecast
    "train_split":      0.8,     # 80/20 chronological split
    "seed":             42,      # global RNG seed
}
```

**Seed pool:** Reservoir weight matrices are drawn from `np.random.seed(SEED_POOL[i])` for each reservoir `i`, using the fixed pool `[42, 43, 44, ..., 51]`. This guarantees identical weights between simulation and hardware runs, making results directly comparable.

**Data period start dates** used across all scripts:

| Period | Start date | End date |
|---|---|---|
| 1y | 2024-02-07 | 2025-01-01 |
| 2y | 2023-02-16 | 2025-01-01 |
| 3y | 2022-02-24 | 2025-01-01 |
| 4y | 2021-03-05 | 2025-01-01 |
| 5y | 2020-03-13 | 2025-01-01 |
| 6y | 2019-03-22 | 2025-01-01 |
| 7y | 2018-04-02 | 2025-01-01 |
| 8y | 2017-04-10 | 2025-01-01 |
| 9y | 2016-04-19 | 2025-01-01 |
| 10y | 2015-04-27 | 2025-01-01 |

---

## Key Results

All results below use 4 qubits, `lookback=5`, `hankel_window=3`, `forecast_horizon=5`.

### Best QRC simulation results

| Ticker | Best RMSE | Period | Layers | Reservoirs |
|---|---|---|---|---|
| SPY | 0.002663 | 8y | 8 | 1 |
| QQQ | 0.003592 | 8y | 8 | 1 |
| GLD | 0.003155 | 8y | 6 | 10 |

### QRC vs best classical (RMSE) — win/loss by ticker

| Ticker | QRC wins | Classical wins | Notes |
|---|---|---|---|
| SPY | 9 / 10 periods | 1 / 10 periods | QRC competitive across nearly all horizons |
| QQQ | 7 / 10 periods | 3 / 10 periods | Classical wins at 1y, 9y, 10y |
| GLD | 0 / 10 periods | 10 / 10 periods | Classical (GARCH/LSTM) outperforms QRC on GLD |

### IBM hardware validation (4y period, 4 qubits, 1 layer, 2 reservoirs)

| Ticker | Hardware RMSE | Simulation RMSE | QLIKE | MZ β | MZ R² | Runtime |
|---|---|---|---|---|---|---|
| SPY | 0.003609 | 0.003149 | 0.1349 | 0.464 | 0.0405 | ~470s |
| QQQ | 0.004336 | 0.003943 | 0.1134 | 0.419 | 0.0233 | ~590s |
| GLD | 0.004044 | 0.003565 | 0.1447 | 0.226 | 0.0064 | ~590s |

Hardware RMSE is 7–14% above simulation. SPY hardware RMSE (0.003609) still beats GARCH(1,1) (0.003891) on the same 4-year period.

### Noise robustness (SPY, 4y, 1 layer, 2 reservoirs)

| Run type | Depolar | RMSE | QLIKE |
|---|---|---|---|
| Statevector baseline | — | 0.003277 | 0.1202 |
| Aer ideal (shot noise only) | 0% | 0.003418 | 0.1284 |
| Aer noisy | 1% | 0.003292 | 0.1205 |
| Aer noisy | 2% | 0.003716 | 0.1472 |
| Aer noisy | 5% | 0.003602 | 0.1424 |

RMSE degradation at 5% depolarizing noise is ~10% for SPY, ~21% for QQQ, and ~2% for GLD — indicating GLD's QRC features are intrinsically more noise-resilient despite lower predictive accuracy overall.

---

## Results Files

| File | Rows | Contents |
|---|---|---|
| `4K_Configuration_Test_SPY.csv` | 1,000 | Grid: 10 periods × 10 layers × 10 reservoirs — SPY |
| `4K_Configuration_Test_QQQ.csv` | 1,000 | Grid: same axes — QQQ |
| `4K_Configuration_Test_GLD_Data.csv` | 1,000 | Grid: same axes — GLD |
| `4Qubits1Layer2Reservoir4YearHardware.csv` | 3 | Hardware results for SPY, QQQ, GLD |
| `Claasical_Test_SPY.csv` | 80 | 8 models × 10 periods — SPY |
| `Classical_Test_QQQ.csv` | 80 | 8 models × 10 periods — QQQ |
| `Classical_Test_GLD.csv` | 80 | 8 models × 10 periods — GLD |
| `Noise_Test.csv` | 30 | 3 tickers × (statevector + aer_ideal + 8 noise levels) |

Column definitions shared across grid result files: `period`, `n_qubits`, `n_layers`, `n_reservoirs`, `seeds`, `n_features`, `best_alpha`, `rmse`, `qlike`, `mz_alpha`, `mz_beta`, `mz_r2`, `train_size`, `test_size`.

---

## Citation

If you use this code or results in your research, please cite.

---

*For questions or issues, please open a GitHub issue or contact the author via the repository.*
