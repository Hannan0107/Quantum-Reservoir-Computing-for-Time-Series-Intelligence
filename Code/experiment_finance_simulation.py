"""
experiment_finance_simulation.py
=================================

Pipeline
--------
  1. Download SPY price data via yfinance
  2. Compute daily returns and rolling log-volatility targets
  3. Extract Hankel-stacked quantum reservoir features
  4. Fit RidgeCV readout in log-vol space
  5. Evaluate: RMSE, QLIKE, Mincer-Zarnowitz (α, β, R²)
  6. Grid-search over data_period × n_layers × n_reservoirs
     (n_qubits set manually in CFG)
"""

import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from qrc_core import QuantumReservoir, prepare_hankel_features

warnings.filterwarnings("ignore")


# ============================================================
# MANUAL CONFIGURATION — set n_qubits before each run
# ============================================================
CFG = {
    "ticker":           "SPY",        # ← Change manually (QQQ, GLD)
    "n_qubits":         10,          # ← Change manually (eg. 4,6,8,10)
    "lookback":         5,
    "hankel_window":    3,
    "forecast_horizon": 5,
    "train_split":      0.8,
    "seed":             42,
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

# Grid axes — n_qubits is set manually above
GRID = {
    "data_period":  ["1y", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "10y"],  # <---- Change manually (eg. 1y, 2y---10y)
    "n_layers":     list(range(1, 11)),
    "n_reservoirs": list(range(1, 11)),
}

# Seeds grow as n_reservoirs increases (accumulating diversity)
SEED_POOL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

RESULTS_FILE = f"qrc_results_q{CFG['n_qubits']}_logvol.csv"


# ============================================================
# DATA  (cached so yfinance is not hit repeatedly)
# ============================================================
_data_cache: dict = {}


def get_market_data(period: str):
    """
    Download SPY data for `period` and return (returns, log_vol_targets).

    Returns
    -------
    returns : np.ndarray, shape (N,)   — daily returns
    targets : np.ndarray, shape (N,)   — rolling 5-day log-volatility
    """
    if period in _data_cache:
        return _data_cache[period]

    df = yf.download(
        CFG["ticker"],
        start=PERIOD_STARTS[period],
        end=END_DATE,
        interval="1d",
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data for {CFG['ticker']} period={period}.")

    close      = df["Close"].values.flatten()
    returns    = np.diff(close) / close[:-1]

    vol_window = 5
    raw_vol    = np.array(
        [np.std(returns[i - vol_window:i]) for i in range(vol_window, len(returns))]
    )
    targets = np.log(np.clip(raw_vol, 1e-10, None))
    returns = returns[vol_window:]

    min_len = min(len(returns), len(targets))
    returns, targets = returns[:min_len], targets[:min_len]

    mask = ~(np.isnan(returns) | np.isnan(targets))
    result = returns[mask], targets[mask]

    _data_cache[period] = result
    return result


# ============================================================
# SINGLE EXPERIMENT RUN
# ============================================================

def run_experiment(period: str, n_layers: int, n_reservoirs: int) -> dict:
    seeds = SEED_POOL[:n_reservoirs]

    # ----- Data -----
    returns, targets = get_market_data(period)

    # ----- Reservoir (from qrc_core) -----
    reservoir = QuantumReservoir(
        n_qubits=CFG["n_qubits"],
        n_layers=n_layers,
        seeds=seeds,
        global_seed=CFG["seed"],
    )

    # ----- Features (from qrc_core) -----
    X_hankel = prepare_hankel_features(
        returns,
        reservoir,
        lookback=CFG["lookback"],
        window=CFG["hankel_window"],
    )

    # ----- Align targets -----
    offset  = CFG["lookback"] + CFG["hankel_window"] - 1 + CFG["forecast_horizon"]
    y       = targets[offset:]
    min_len = min(len(X_hankel), len(y))
    X       = X_hankel[:min_len]
    y       = y[:min_len]

    # ----- Train / test split -----
    split              = int(len(X) * CFG["train_split"])
    X_train, X_test    = X[:split],  X[split:]
    y_train, y_test    = y[:split],  y[split:]

    # ----- Scale -----
    scaler             = StandardScaler()
    X_train_sc         = scaler.fit_transform(X_train)
    X_test_sc          = scaler.transform(X_test)

    # ----- RidgeCV (loose alphas → better MZ beta) -----
    tscv  = TimeSeriesSplit(n_splits=5)
    reg   = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], cv=tscv)
    reg.fit(X_train_sc, y_train)
    preds = reg.predict(X_test_sc)

    # ----- Convert back to raw vol for RMSE / QLIKE -----
    preds_raw         = np.exp(preds)
    y_test_raw        = np.exp(y_test)
    preds_clipped     = np.clip(preds_raw,   1e-8, None)
    y_test_clipped    = np.clip(y_test_raw,  1e-8, None)

    rmse  = np.sqrt(mean_squared_error(y_test_raw, preds_clipped))
    qlike = np.mean(
        y_test_clipped / preds_clipped
        - np.log(y_test_clipped / preds_clipped)
        - 1
    )

    # ----- Mincer-Zarnowitz in log space -----
    mz_X  = np.column_stack([np.ones(len(preds)), preds])
    mz    = sm.OLS(y_test, mz_X).fit()

    return {
        "period":       period,
        "n_qubits":     CFG["n_qubits"],
        "n_layers":     n_layers,
        "n_reservoirs": n_reservoirs,
        "seeds":        str(seeds),
        "n_features":   X_hankel.shape[1],
        "best_alpha":   reg.alpha_,
        "rmse":         round(rmse,                   6),
        "qlike":        round(qlike,                  6),
        "mz_alpha":     round(float(mz.params[0]),    6),
        "mz_beta":      round(float(mz.params[1]),    6),
        "mz_r2":        round(float(mz.rsquared),     4),
        "train_size":   len(X_train),
        "test_size":    len(X_test),
    }


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_existing_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        print(f"  Loaded {len(df)} existing results from {RESULTS_FILE}")
        return df
    return pd.DataFrame()


def already_done(existing_df: pd.DataFrame, period, n_layers, n_reservoirs) -> bool:
    if existing_df.empty:
        return False
    mask = (
        (existing_df["period"]       == period)
        & (existing_df["n_layers"]     == n_layers)
        & (existing_df["n_reservoirs"] == n_reservoirs)
        & (existing_df["n_qubits"]     == CFG["n_qubits"])
    )
    return mask.any()


# ============================================================
# GRID SEARCH
# ============================================================

def run_grid_search() -> pd.DataFrame:
    combos      = list(itertools.product(
        GRID["data_period"],
        GRID["n_layers"],
        GRID["n_reservoirs"],
    ))
    total       = len(combos)
    existing_df = load_existing_results()
    results     = existing_df.to_dict("records") if not existing_df.empty else []

    done_count = sum(
        1 for (p, l, r) in combos if already_done(existing_df, p, l, r)
    )
    remaining = total - done_count

    print(f"\n{'='*60}")
    print(f"  QRC GRID SEARCH — {CFG['n_qubits']} qubits (fixed)")
    print(f"  Target: LOG VOLATILITY  |  Ridge: loose alphas → MZ focus")
    print(f"  Total combos : {total}  |  Done : {done_count}  |  Remaining : {remaining}")
    print(f"{'='*60}\n")

    run_idx = 0
    for period, n_layers, n_reservoirs in combos:
        seeds = SEED_POOL[:n_reservoirs]

        if already_done(existing_df, period, n_layers, n_reservoirs):
            print(f"  [SKIP] period={period} layers={n_layers} "
                  f"reservoirs={n_reservoirs} seeds={seeds}")
            continue

        run_idx += 1
        print(f"  [{run_idx}/{remaining}] period={period}  layers={n_layers}  "
              f"reservoirs={n_reservoirs}  seeds={seeds}")

        t0 = time.time()
        try:
            result  = run_experiment(period, n_layers, n_reservoirs)
            elapsed = time.time() - t0
            print(
                f"          RMSE={result['rmse']:.6f}  "
                f"QLIKE={result['qlike']:.6f}  "
                f"R²={result['mz_r2']:.4f}  "
                f"MZ_a={result['mz_alpha']:.4f}(ideal:0)  "
                f"MZ_b={result['mz_beta']:.4f}(ideal:1)  "
                f"({elapsed:.1f}s)"
            )
            results.append(result)
        except Exception as e:
            print(f"          ERROR: {e}")
            results.append({
                "period":       period,
                "n_qubits":     CFG["n_qubits"],
                "n_layers":     n_layers,
                "n_reservoirs": n_reservoirs,
                "seeds":        str(seeds),
                "error":        str(e),
                "rmse": np.nan, "qlike": np.nan, "mz_r2": np.nan,
            })

        # Checkpoint after every run
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    return pd.DataFrame(results)


# ============================================================
# SUMMARY + HEATMAPS
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH COMPLETE — {CFG['n_qubits']} qubits | LOG VOL target")
    print(f"  MZ in log space  |  RMSE/QLIKE in raw-vol space")
    print(f"{'='*60}\n")

    valid = df.dropna(subset=["rmse"])
    cols  = ["period", "n_layers", "n_reservoirs", "seeds",
             "rmse", "qlike", "mz_r2", "best_alpha"]

    for label, key, ascending in [
        ("RMSE (lower is better)",               "rmse",   True),
        ("QLIKE (lower is better)",              "qlike",  True),
        ("Mincer-Zarnowitz R² (higher is better)", "mz_r2", False),
    ]:
        fn  = valid.nsmallest if ascending else valid.nlargest
        top = fn(5, key)[cols]
        print(f"  Top 5 by {label}:")
        print("  " + "-" * 56)
        print(top.to_string(index=False))
        print()

    for title, idx, col in [
        ("layers × reservoirs (avg across periods)",    "n_layers",     "n_reservoirs"),
        ("layers × data period (avg across reservoirs)","n_layers",     "period"),
        ("reservoirs × data period (avg across layers)","n_reservoirs", "period"),
    ]:
        pivot = valid.pivot_table(
            values="rmse", index=idx, columns=col, aggfunc="mean"
        ).round(6)
        pivot.index.name = idx.replace("n_", "").replace("_", " ") + " \\ " + col.replace("n_", "")
        print(f"  RMSE Heatmap: {title}")
        print("  " + "-" * 40)
        print(pivot.to_string())
        print()

    print(f"  Full results saved to: {RESULTS_FILE}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results_df = run_grid_search()
    print_summary(results_df)
