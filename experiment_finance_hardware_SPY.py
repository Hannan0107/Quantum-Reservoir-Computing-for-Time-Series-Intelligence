"""
experiment_finance_hardware_SPY.py
===================================

Pipeline (Hardware Version - Multi-Reservoir)
---------------------------------------------
  1. Download price data via yfinance
  2. Compute daily returns and rolling log-volatility targets
  3. Build quantum circuits for multiple reservoirs and extract features via IBM hardware (EstimatorV2)
  4. Apply Hankel stacking on concatenated latent features
  5. Fit RidgeCV readout in log-vol space
  6. Evaluate: RMSE, QLIKE, Mincer-Zarnowitz (alpha, beta, R2)
"""

import warnings
import numpy as np
import statsmodels.api as sm
import yfinance as yf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Only import these two utilities from qrc_core (no QuantumReservoir needed)
from qrc_core import build_observables, apply_hankel

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION — adjust before each run
# ============================================================
CFG = {
    "ticker":           "SPY",       # Change manually to QQQ, GLD
    "n_qubits":         4,
    "n_layers":         1,
    "n_reservoirs":     2,
    "lookback":         5,
    "hankel_window":    3,
    "forecast_horizon": 5,
    "train_split":      0.8,
    "seed":             42,
    "shots":            1024,        # Shots per circuit on hardware
    "data_period":      "4y",
}

PERIOD_STARTS = {
    "1y":  "2024-02-07",
    "2y":  "2023-02-16",
    "3y":  "2022-02-24",
    "4y":  "2021-03-05",
    "5y":  "2020-03-13",
    "6y":  "2019-03-22",
    "7y":  "2018-04-02",
    "8y":  "2017-04-10",
    "9y":  "2016-04-19",
    "10y": "2015-04-27",
}
END_DATE = "2025-01-01"

# Seeds pool matching the simulation script to guarantee identical weight generation
SEED_POOL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


# ============================================================
# IBM QUANTUM CREDENTIALS — fill in before running
# ============================================================
IBM_TOKEN        = "API"   # <-- CHANGE
IBM_INSTANCE     = "CRN"         # <-- CHANGE
IBM_SERVICE_NAME = "open-instance"        # <-- CHANGE
IBM_BACKEND_NAME = "ibm_kingston"     # <-- CHANGE if needed

try:
    QiskitRuntimeService.save_account(
        token=IBM_TOKEN,
        channel="ibm_cloud",
        instance=IBM_INSTANCE,
        name=IBM_SERVICE_NAME,
        overwrite=True,
    )
except Exception as e:
    print(f"  [WARN] Could not save account: {e}")

service    = QiskitRuntimeService(name=IBM_SERVICE_NAME)
hw_backend = service.backend(IBM_BACKEND_NAME)

print(f"\n  Target Backend : {hw_backend.name}")
print(f"  Qubits         : {hw_backend.num_qubits}")
print(f"  Basis gates    : {hw_backend.operation_names}")


# ============================================================
# OBSERVABLES
# ============================================================
OBSERVABLES = build_observables(CFG["n_qubits"])
N_OBS       = len(OBSERVABLES)
print(f"\n  Observables per circuit : {N_OBS} "
      f"(Z×{CFG['n_qubits']} + ZZ×{CFG['n_qubits'] * (CFG['n_qubits'] - 1) // 2})")


# ============================================================
# DATA
# ============================================================
def get_market_data():
    """Download market data and return (returns, log_vol_targets)."""
    df = yf.download(
        CFG["ticker"],
        start=PERIOD_STARTS[CFG["data_period"]],
        end=END_DATE,
        interval="1d",
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data for {CFG['ticker']}.")

    close      = df["Close"].values.flatten()
    returns    = np.diff(close) / close[:-1]

    vol_window = 5
    raw_vol    = np.array(
        [np.std(returns[i - vol_window:i]) for i in range(vol_window, len(returns))]
    )
    targets = np.log(np.clip(raw_vol, 1e-10, None))
    returns = returns[vol_window:]

    min_len         = min(len(returns), len(targets))
    returns, targets = returns[:min_len], targets[:min_len]

    mask = ~(np.isnan(returns) | np.isnan(targets))
    return returns[mask], targets[mask]

# ============================================================
# CIRCUIT BUILDER
# ============================================================
# Generate a distinct weight matrix for each reservoir based on the SEED_POOL
HW_WEIGHTS_LIST = []
for i in range(CFG["n_reservoirs"]):
    np.random.seed(SEED_POOL[i])
    weights = np.random.uniform(
        0, 2 * np.pi,
        size=(CFG["n_layers"], CFG["n_qubits"])
    )
    HW_WEIGHTS_LIST.append(weights)


def build_circuit(return_sequence, res_idx):
    """Build a single parameterised reservoir circuit for one lookback window and specific reservoir."""
    qc = QuantumCircuit(CFG["n_qubits"])
    current_weights = HW_WEIGHTS_LIST[res_idx]

    for r_t in return_sequence:
        val = float(r_t)
        for q in range(CFG["n_qubits"] // 2):
            qc.ry(np.pi * val, q)
        for q in range(CFG["n_qubits"] // 2, CFG["n_qubits"]):
            qc.ry(0.0, q)

        for layer in range(CFG["n_layers"]):
            for q in range(CFG["n_qubits"]):
                qc.rz(current_weights[layer, q], q)
            for q in range(CFG["n_qubits"] - 1):
                qc.cx(q, q + 1)
    return qc


# ============================================================
# HARDWARE FEATURE EXTRACTION
# ============================================================
def extract_features_hardware(return_sequences, backend):
    """
    Submit circuits to IBM hardware and return expectation value features.
    Builds (N_sequences * n_reservoirs) circuits, then concatenates the features.
    """
    n_seq = len(return_sequences)
    r_count = CFG["n_reservoirs"]

    print(f"\n  Building {n_seq} sequences x {r_count} reservoirs = {n_seq * r_count} circuits...")

    raw_circuits = []
    # Loop sequences first, then reservoirs, so we can reshape cleanly later
    for seq in return_sequences:
        for res_idx in range(r_count):
            raw_circuits.append(build_circuit(seq, res_idx))

    print(f"  Transpiling circuits for {backend.name}...")
    pm                  = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_circuits = pm.run(raw_circuits)
    print(f"  Transpilation complete. Example depth: {transpiled_circuits[0].depth()}")

    # Build pubs with per-circuit layout transformation
    pubs = []
    for tc in transpiled_circuits:
        layout     = tc.layout
        mapped_obs = [obs.apply_layout(layout) for obs in OBSERVABLES] if layout else OBSERVABLES
        pubs.append((tc, mapped_obs))

    print(f"  Submitting {len(pubs)} total circuits x {N_OBS} observables to {backend.name}...")
    estimator                        = EstimatorV2(mode=backend)
    estimator.options.default_shots  = CFG["shots"]
    job    = estimator.run(pubs)

    print(f"  Job ID : {job.job_id()}")
    print("  Waiting for results...")
    result = job.result()
    print("  Results received.")

    # Extract all expectation values as a flat array of arrays
    flat_features = np.array([res.data.evs for res in result])

    # Reshape: group every `r_count` circuits back into a single row for that time sequence
    # Original: (n_seq * r_count, N_OBS) -> New: (n_seq, r_count * N_OBS)
    features = flat_features.reshape(n_seq, r_count * N_OBS)

    print(f"  Feature matrix shape : {features.shape} (Sequences, Reservoirs × Observables)")
    return features


# ============================================================
# SMOKE TEST (optional — uncomment call in main to run)
# ============================================================
def smoke_test(sequences, backend):
    print("\n  Running smoke test with 1 sequence...")
    test_features = extract_features_hardware(sequences[:1], backend)
    expected_dim = CFG["n_reservoirs"] * N_OBS
    assert test_features.shape == (1, expected_dim), \
        f"Smoke test failed: got {test_features.shape}, expected (1, {expected_dim})"
    print("  Smoke test passed. Ready for full job.\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(f"  QRC ON IBM HARDWARE — {CFG['ticker']} | LOG VOL TARGET | HANKEL")
    print("=" * 60)

    # 1. Load data
    returns, targets = get_market_data()
    print(f"\n[1/7] Data : returns {returns.shape}, targets {targets.shape}")

    # 2. Build lookback sequences
    sequences = [
        returns[i - CFG["lookback"]:i]
        for i in range(CFG["lookback"], len(returns))
    ]
    print(f"[2/7] Sequences built : {len(sequences)}")

    # 3. Optional smoke test — uncomment to verify 1 sequence (all reservoirs) before full job
    # smoke_test(sequences, hw_backend)

    # 4. Extract features from hardware
    latent_history = extract_features_hardware(sequences, hw_backend)
    print(f"[3/7] Raw latent shape : {latent_history.shape}")

    # 5. Apply Hankel stacking
    X = apply_hankel(latent_history, CFG["hankel_window"])
    print(f"[4/7] Hankel features  : {X.shape}")

    # 6. Align targets (log-vol targets, same as simulation file)
    offset  = CFG["lookback"] + CFG["hankel_window"] - 1 + CFG["forecast_horizon"]
    y       = targets[offset:]
    min_len = min(len(X), len(y))
    X       = X[:min_len]
    y       = y[:min_len]
    print(f"[5/7] Aligned          : X {X.shape} | y {y.shape}")

    # 7. Train / test split + scaling
    split      = int(len(X) * CFG["train_split"])
    X_train    = X[:split];  X_test = X[split:]
    y_train    = y[:split];  y_test = y[split:]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 8. RidgeCV (loose alphas — same as simulation file for fair comparison)
    tscv  = TimeSeriesSplit(n_splits=5)
    reg   = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], cv=tscv)
    reg.fit(X_train_sc, y_train)
    preds = reg.predict(X_test_sc)
    print(f"[6/7] Ridge best alpha : {reg.alpha_}")

    # 9. Convert back to raw vol for RMSE / QLIKE
    preds_raw      = np.exp(preds)
    y_test_raw     = np.exp(y_test)
    preds_clipped  = np.clip(preds_raw,  1e-8, None)
    y_test_clipped = np.clip(y_test_raw, 1e-8, None)

    rmse  = np.sqrt(mean_squared_error(y_test_raw, preds_clipped))
    qlike = np.mean(
        y_test_clipped / preds_clipped
        - np.log(y_test_clipped / preds_clipped)
        - 1
    )

    # 10. Mincer-Zarnowitz in log space
    mz_X = np.column_stack([np.ones(len(preds)), preds])
    mz   = sm.OLS(y_test, mz_X).fit()
    print(f"[7/7] Evaluation complete.")

    # ---- Results ----
    print("\n" + "=" * 60)
    print(f"  RESULTS — {hw_backend.name}")
    print(f"  {CFG['ticker']} | n_qubits={CFG['n_qubits']} | "
          f"n_layers={CFG['n_layers']} | n_reservoirs={CFG['n_reservoirs']} | period={CFG['data_period']}")
    print(f"  Total circuits executed : {len(sequences) * CFG['n_reservoirs']}")
    print("=" * 60)
    print(f"  Best Alpha : {reg.alpha_}")
    print(f"  RMSE       : {rmse:.6f}")
    print(f"  QLIKE      : {qlike:.6f}")
    print(f"\n  Mincer-Zarnowitz (log space):")
    print(f"    alpha = {mz.params[0]:.6f}  (ideal: 0)")
    print(f"    beta  = {mz.params[1]:.6f}  (ideal: 1)")
    print(f"    R2    = {mz.rsquared:.4f}")
    print("=" * 60)