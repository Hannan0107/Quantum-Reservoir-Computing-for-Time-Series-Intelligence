"""
experiment_finance_simulation_QQQ.py
=====================================

Pipeline
--------
  1. Load price data from the local snapshot (market_data.py)
  2. Compute daily returns and rolling log-volatility targets
  3. Extract Hankel-stacked quantum reservoir features
  4. Fit RidgeCV readout in log-vol space
  5. Evaluate: RMSE, QLIKE, Mincer-Zarnowitz (α, β, R²)

Two run modes (set RUN_MODE below):
------------------------------------
  "grid"        — original grid search over n_layers × n_reservoirs
                  (uses statevector backend, unchanged from original)

  "noise_sweep" — hold n_layers / n_reservoirs / period fixed,
                  sweep over a range of depolarizing noise levels
                  to measure how model performance degrades with noise.

Backends available in qrc_core.QuantumReservoir
------------------------------------------------
  "statevector"  — exact simulation, zero noise          (original)
  "aer_ideal"    — Aer shot-based, zero noise            (sanity check)
  "aer_noise"    — Aer shot-based + NoiseModel           (the degradation study)

Noise parameters (aer_noise)
-----------------------------
  depolar_error : single-qubit gate depolarizing probability
  cx_error      : CX gate depolarizing probability  (set = 2× depolar by default)
  readout_error : symmetric measurement bit-flip probability
  shots         : number of measurement shots (more → less shot noise)
"""

import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from market_data import DATA_DIR, load_close
from qrc_core import QuantumReservoir, prepare_hankel_features

warnings.filterwarnings("ignore")


# ============================================================
# RUN MODE  ← change this before each run
# ============================================================
RUN_MODE = "noise_sweep"   # "grid"  or  "noise_sweep"


# ============================================================
# MANUAL CONFIGURATION
# ============================================================
CFG = {
    "ticker":           "SPY",    # change manually (QQQ, GLD)
    "n_qubits":         4,        # change manually (e.g. 4, 6, 8, 10)
    "lookback":         5,
    "hankel_window":    3,
    "forecast_horizon": 5,
    "train_split":      0.8,
    "seed":             42,
    "data_dir":         DATA_DIR,  # price snapshots (see market_data.py)
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

SEED_POOL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# ── Grid search settings ──────────────────────────────────────
GRID = {
    "data_period":  ["4y"],
    "n_layers":     list(range(1, 11)),
    "n_reservoirs": list(range(1, 11)),
}
RESULTS_FILE = f"qrc_results_q{CFG['n_qubits']}_logvol_QQQ.csv"

# ── Noise sweep settings ─────────────────────────────────────
NOISE_CFG = {
    # Fixed architecture for the sweep — pick your best combo from grid results
    "data_period":  "4y",
    "n_layers":     1,
    "n_reservoirs": 2,

    # Backends to compare:
    #   "statevector"  — exact baseline (no shots, no noise)
    #   "aer_ideal"    — Aer, high shots, zero noise  (validates Aer vs statevector)
    #   "aer_noise"    — Aer + noise model
    "backends": ["statevector", "aer_ideal", "aer_noise"],

    # Shots for Aer backends (higher = less shot noise, but slower)
    "shots": 1024,

    # Depolarizing error levels to sweep (single-qubit gate error probability).
    # CX error is set to 2× this value; readout error equals this value.
    # 0.0 = no noise; 0.05 = very noisy (5 % per gate)
    "depolar_levels": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
}
NOISE_RESULTS_FILE = f"qrc_noise_sweep_q{CFG['n_qubits']}_QQQ.csv"


# ============================================================
# DATA  (local snapshot, cached per period)
# ============================================================
_data_cache: dict = {}


def get_market_data(period: str):
    if period in _data_cache:
        return _data_cache[period]

    close   = load_close(CFG["ticker"], PERIOD_STARTS[period], END_DATE, CFG["data_dir"]).values
    returns = np.diff(close) / close[:-1]

    vol_window = 5
    raw_vol    = np.array(
        [np.std(returns[i - vol_window:i]) for i in range(vol_window, len(returns))]
    )
    targets = np.log(np.clip(raw_vol, 1e-10, None))
    returns = returns[vol_window:]

    min_len          = min(len(returns), len(targets))
    returns, targets = returns[:min_len], targets[:min_len]

    mask   = ~(np.isnan(returns) | np.isnan(targets))
    result = returns[mask], targets[mask]

    _data_cache[period] = result
    return result


# ============================================================
# CORE EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    period:        str,
    n_layers:      int,
    n_reservoirs:  int,
    backend:       str   = "statevector",
    shots:         int   = 1024,
    depolar_error: float = 0.0,
    cx_error:      float = 0.0,
    readout_error: float = 0.0,
) -> dict:
    """
    Run one experiment and return a metrics dict.

    Parameters
    ----------
    backend       : "statevector", "aer_ideal", or "aer_noise"
    shots         : shots for Aer backends (ignored for statevector)
    depolar_error : single-qubit gate depolarizing probability
    cx_error      : CX gate depolarizing probability
    readout_error : measurement bit-flip probability
    """
    seeds = SEED_POOL[:n_reservoirs]

    returns, targets = get_market_data(period)

    reservoir = QuantumReservoir(
        n_qubits      = CFG["n_qubits"],
        n_layers      = n_layers,
        seeds         = seeds,
        global_seed   = CFG["seed"],
        backend       = backend,
        shots         = shots,
        depolar_error = depolar_error,
        cx_error      = cx_error,
        readout_error = readout_error,
    )

    X_hankel = prepare_hankel_features(
        returns,
        reservoir,
        lookback = CFG["lookback"],
        window   = CFG["hankel_window"],
    )

    offset  = CFG["lookback"] + CFG["hankel_window"] - 1 + CFG["forecast_horizon"]
    y       = targets[offset:]
    min_len = min(len(X_hankel), len(y))
    X       = X_hankel[:min_len]
    y       = y[:min_len]

    split           = int(len(X) * CFG["train_split"])
    X_train, X_test = X[:split],  X[split:]
    y_train, y_test = y[:split],  y[split:]

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=5)
    reg  = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], cv=tscv)
    reg.fit(X_train_sc, y_train)
    preds = reg.predict(X_test_sc)

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

    mz_X = np.column_stack([np.ones(len(preds)), preds])
    mz   = sm.OLS(y_test, mz_X).fit()

    return {
        "period":        period,
        "n_qubits":      CFG["n_qubits"],
        "n_layers":      n_layers,
        "n_reservoirs":  n_reservoirs,
        "backend":       backend,
        "shots":         shots if backend != "statevector" else "N/A",
        "depolar_error": depolar_error,
        "cx_error":      cx_error,
        "readout_error": readout_error,
        "seeds":         str(seeds),
        "n_features":    X_hankel.shape[1],
        "best_alpha":    reg.alpha_,
        "rmse":          round(rmse,                6),
        "qlike":         round(qlike,               6),
        "mz_alpha":      round(float(mz.params[0]), 6),
        "mz_beta":       round(float(mz.params[1]), 6),
        "mz_r2":         round(float(mz.rsquared),  4),
        "train_size":    len(X_train),
        "test_size":     len(X_test),
    }


# ============================================================
# GRID SEARCH  (original behaviour, unchanged)
# ============================================================

def load_existing_results(filepath: str) -> pd.DataFrame:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} existing results from {filepath}")
        return df
    return pd.DataFrame()


def already_done_grid(existing_df, period, n_layers, n_reservoirs) -> bool:
    if existing_df.empty:
        return False
    mask = (
        (existing_df["period"]       == period)
        & (existing_df["n_layers"]   == n_layers)
        & (existing_df["n_reservoirs"] == n_reservoirs)
        & (existing_df["n_qubits"]   == CFG["n_qubits"])
    )
    return mask.any()


def run_grid_search() -> pd.DataFrame:
    combos      = list(itertools.product(
        GRID["data_period"], GRID["n_layers"], GRID["n_reservoirs"],
    ))
    total       = len(combos)
    existing_df = load_existing_results(RESULTS_FILE)
    results     = existing_df.to_dict("records") if not existing_df.empty else []

    done_count = sum(1 for (p, l, r) in combos if already_done_grid(existing_df, p, l, r))
    remaining  = total - done_count

    print(f"\n{'='*60}")
    print(f"  QRC GRID SEARCH — {CFG['n_qubits']} qubits | backend=statevector")
    print(f"  Ticker: {CFG['ticker']}")
    print(f"  Total combos : {total}  |  Done : {done_count}  |  Remaining : {remaining}")
    print(f"{'='*60}\n")

    run_idx = 0
    for period, n_layers, n_reservoirs in combos:
        if already_done_grid(existing_df, period, n_layers, n_reservoirs):
            continue

        run_idx += 1
        seeds = SEED_POOL[:n_reservoirs]
        print(f"  [{run_idx}/{remaining}] period={period}  layers={n_layers}  "
              f"reservoirs={n_reservoirs}  seeds={seeds}")
        t0 = time.time()
        try:
            result  = run_experiment(period, n_layers, n_reservoirs, backend="statevector")
            elapsed = time.time() - t0
            print(
                f"          RMSE={result['rmse']:.6f}  "
                f"QLIKE={result['qlike']:.6f}  "
                f"R²={result['mz_r2']:.4f}  "
                f"MZ_a={result['mz_alpha']:.4f}  "
                f"MZ_b={result['mz_beta']:.4f}  "
                f"({elapsed:.1f}s)"
            )
            results.append(result)
        except Exception as e:
            print(f"          ERROR: {e}")
            results.append({
                "period": period, "n_qubits": CFG["n_qubits"],
                "n_layers": n_layers, "n_reservoirs": n_reservoirs,
                "backend": "statevector", "error": str(e),
                "rmse": np.nan, "qlike": np.nan, "mz_r2": np.nan,
            })

        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    return pd.DataFrame(results)


# ============================================================
# NOISE SWEEP
# ============================================================

def run_noise_sweep() -> pd.DataFrame:
    """
    Hold architecture fixed. Sweep over:
      - backend: statevector, aer_ideal, aer_noise
      - depolar_error level (aer_noise only)

    This isolates the effect of quantum noise on model performance.
    """
    period        = NOISE_CFG["data_period"]
    n_layers      = NOISE_CFG["n_layers"]
    n_reservoirs  = NOISE_CFG["n_reservoirs"]
    shots         = NOISE_CFG["shots"]
    depolar_levels = NOISE_CFG["depolar_levels"]
    backends      = NOISE_CFG["backends"]

    existing_df = load_existing_results(NOISE_RESULTS_FILE)
    results     = existing_df.to_dict("records") if not existing_df.empty else []

    def already_done(backend, depolar):
        if existing_df.empty:
            return False
        mask = (
            (existing_df["backend"]       == backend)
            & (existing_df["depolar_error"] == depolar)
            & (existing_df["n_layers"]      == n_layers)
            & (existing_df["n_reservoirs"]  == n_reservoirs)
            & (existing_df["period"]        == period)
        )
        return mask.any()

    print(f"\n{'='*60}")
    print(f"  QRC NOISE SWEEP — {CFG['n_qubits']} qubits")
    print(f"  Ticker: {CFG['ticker']}  |  period={period}")
    print(f"  Architecture: layers={n_layers}  reservoirs={n_reservoirs}")
    print(f"  Shots (Aer): {shots}")
    print(f"  Backends: {backends}")
    print(f"  Depolar levels: {depolar_levels}")
    print(f"{'='*60}\n")

    # ── statevector baseline (once, no noise params) ──────────
    if "statevector" in backends:
        if already_done("statevector", 0.0):
            print("  [SKIP] statevector baseline")
        else:
            print("  [RUN ] statevector baseline (exact, no noise)")
            t0 = time.time()
            try:
                r = run_experiment(period, n_layers, n_reservoirs, backend="statevector")
                elapsed = time.time() - t0
                print(
                    f"         RMSE={r['rmse']:.6f}  QLIKE={r['qlike']:.6f}"
                    f"  R²={r['mz_r2']:.4f}"
                    f"  MZ_a={r['mz_alpha']:.4f}(ideal:0)"
                    f"  MZ_b={r['mz_beta']:.4f}(ideal:1)"
                    f"  ({elapsed:.1f}s)"
                )
                results.append(r)
            except Exception as e:
                import traceback
                print(f"         ERROR: {e}")
                print(f"         TRACEBACK:\n{traceback.format_exc()}")
            pd.DataFrame(results).to_csv(NOISE_RESULTS_FILE, index=False)

    # ── aer_ideal: shot noise only (depolar = 0) ─────────────
    if "aer_ideal" in backends:
        if already_done("aer_ideal", 0.0):
            print(f"  [SKIP] aer_ideal  shots={shots}")
        else:
            print(f"  [RUN ] aer_ideal  shots={shots}  (shot noise only, no gate noise)")
            t0 = time.time()
            try:
                r = run_experiment(
                    period, n_layers, n_reservoirs,
                    backend="aer_ideal", shots=shots,
                    depolar_error=0.0, cx_error=0.0, readout_error=0.0,
                )
                elapsed = time.time() - t0
                print(
                    f"         RMSE={r['rmse']:.6f}  QLIKE={r['qlike']:.6f}"
                    f"  R²={r['mz_r2']:.4f}"
                    f"  MZ_a={r['mz_alpha']:.4f}(ideal:0)"
                    f"  MZ_b={r['mz_beta']:.4f}(ideal:1)"
                    f"  ({elapsed:.1f}s)"
                )
                results.append(r)
            except Exception as e:
                import traceback
                print(f"         ERROR: {e}")
                print(f"         TRACEBACK:\n{traceback.format_exc()}")
            pd.DataFrame(results).to_csv(NOISE_RESULTS_FILE, index=False)

    # ── aer_noise: sweep depolarizing levels ─────────────────
    if "aer_noise" in backends:
        total_noise_runs = len(depolar_levels)
        print(f"\n  aer_noise sweep ({total_noise_runs} levels):")
        for idx, depolar in enumerate(depolar_levels, 1):
            cx_err      = 2.0 * depolar   # CX errors typically ~2× single-qubit
            readout_err = depolar          # symmetric, same scale

            if already_done("aer_noise", depolar):
                print(f"    [{idx:2d}/{total_noise_runs}] SKIP  depolar={depolar:.4f}")
                continue

            print(
                f"    [{idx:2d}/{total_noise_runs}] RUN   depolar={depolar:.4f}  "
                f"cx={cx_err:.4f}  readout={readout_err:.4f}  shots={shots}"
            )
            t0 = time.time()
            try:
                r = run_experiment(
                    period, n_layers, n_reservoirs,
                    backend="aer_noise", shots=shots,
                    depolar_error=depolar,
                    cx_error=cx_err,
                    readout_error=readout_err,
                )
                elapsed = time.time() - t0
                print(
                    f"            RMSE={r['rmse']:.6f}  QLIKE={r['qlike']:.6f}"
                    f"  R²={r['mz_r2']:.4f}"
                    f"  MZ_a={r['mz_alpha']:.4f}(ideal:0)"
                    f"  MZ_b={r['mz_beta']:.4f}(ideal:1)"
                    f"  ({elapsed:.1f}s)"
                )
                results.append(r)
            except Exception as e:
                import traceback
                print(f"            ERROR: {e}")
                print(f"            TRACEBACK:\n{traceback.format_exc()}")
                results.append({
                    "period": period, "n_qubits": CFG["n_qubits"],
                    "n_layers": n_layers, "n_reservoirs": n_reservoirs,
                    "backend": "aer_noise", "shots": shots,
                    "depolar_error": depolar, "cx_error": cx_err, "readout_error": readout_err,
                    "error": str(e),
                    "rmse": np.nan, "qlike": np.nan, "mz_r2": np.nan,
                    "mz_alpha": np.nan, "mz_beta": np.nan,
                })

            pd.DataFrame(results).to_csv(NOISE_RESULTS_FILE, index=False)

    return pd.DataFrame(results)


# ============================================================
# SUMMARIES
# ============================================================

def print_grid_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"  GRID COMPLETE — {CFG['n_qubits']} qubits | LOG VOL | {CFG['ticker']}")
    print(f"{'='*60}\n")

    valid = df.dropna(subset=["rmse"])
    cols  = ["period", "n_layers", "n_reservoirs", "rmse", "qlike", "mz_r2", "best_alpha"]

    for label, key, ascending in [
        ("RMSE (lower is better)",                 "rmse",   True),
        ("QLIKE (lower is better)",                "qlike",  True),
        ("Mincer-Zarnowitz R² (higher is better)", "mz_r2",  False),
    ]:
        fn  = valid.nsmallest if ascending else valid.nlargest
        top = fn(5, key)[cols]
        print(f"  Top 5 by {label}:")
        print("  " + "-" * 56)
        print(top.to_string(index=False))
        print()

    print(f"  Full results saved to: {RESULTS_FILE}")


def print_noise_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"  NOISE SWEEP SUMMARY — {CFG['n_qubits']} qubits | {CFG['ticker']}")
    print(f"{'='*60}\n")

    valid = df.dropna(subset=["rmse"])
    cols  = ["backend", "depolar_error", "shots", "rmse", "qlike", "mz_r2", "mz_beta"]

    print("  Results by backend / noise level:")
    print("  " + "-" * 70)
    print(valid[cols].sort_values(["backend", "depolar_error"]).to_string(index=False))

    # Show degradation vs statevector baseline
    sv_rows = valid[valid["backend"] == "statevector"]
    if not sv_rows.empty:
        baseline_rmse  = sv_rows["rmse"].values[0]
        baseline_r2    = sv_rows["mz_r2"].values[0]
        noisy_rows     = valid[valid["backend"] == "aer_noise"].copy()

        if not noisy_rows.empty:
            noisy_rows["rmse_degradation_%"] = (
                (noisy_rows["rmse"] - baseline_rmse) / baseline_rmse * 100
            ).round(2)
            noisy_rows["r2_drop"] = (baseline_r2 - noisy_rows["mz_r2"]).round(4)

            print(f"\n  Degradation vs statevector baseline "
                  f"(RMSE={baseline_rmse:.6f}, R²={baseline_r2:.4f}):")
            print("  " + "-" * 60)
            deg_cols = ["depolar_error", "rmse", "rmse_degradation_%", "mz_r2", "r2_drop"]
            print(noisy_rows[deg_cols].sort_values("depolar_error").to_string(index=False))

    print(f"\n  Full results saved to: {NOISE_RESULTS_FILE}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n  RUN_MODE = '{RUN_MODE}'")
    print(f"  Ticker   = {CFG['ticker']}  |  n_qubits = {CFG['n_qubits']}")

    if RUN_MODE == "grid":
        results_df = run_grid_search()
        print_grid_summary(results_df)

    elif RUN_MODE == "noise_sweep":
        results_df = run_noise_sweep()
        print_noise_summary(results_df)

    else:
        raise ValueError(f"Unknown RUN_MODE '{RUN_MODE}'. Use 'grid' or 'noise_sweep'.")

