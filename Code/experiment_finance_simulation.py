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

Protocol
--------
  train      = [0,           val_split)
  validation = [val_split,   train_split)
  test       = [train_split, 1]

  Each target spans the next forecast_horizon returns, so the last
  PURGE_GAP = forecast_horizon - 1 training targets overlap the first
  evaluation targets; they are dropped from every readout fit (and the
  same gap separates the RidgeCV TimeSeriesSplit folds).

Two run modes (set RUN_MODE below):
  "grid"     — grid-search data_period × n_layers × n_reservoirs for the
               n_qubits set in CFG. Fits on train, scores on VALIDATION only;
               test-period features are never computed.
  "champion" — reads every grid file for this ticker (all qubit counts),
               picks the config with the lowest validation QLIKE per period,
               refits on train+validation and evaluates on TEST over
               N_EVAL_SEEDS disjoint reservoir seed sets (mean ± std).
               Per-date forecasts are saved for dm_test.py.

Usage (CLI flags override the defaults in RUN_MODE / CFG below)
-----
  py -3.11 experiment_finance_simulation.py --mode grid     --ticker SPY --n-qubits 4
  py -3.11 experiment_finance_simulation.py --mode champion --ticker SPY

Prices come from the local snapshot in market_data.py ({data_dir}/{ticker}_close.csv).
"""

import argparse
import glob
import itertools
import os
import sys
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

# UTF-8 stdout: progress output uses → / ±, which crash a cp1252 Windows console
# or redirected log; line buffering keeps log files current during long runs.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

warnings.filterwarnings("ignore")


# ============================================================
# RUN MODE  (default; override with --mode)
# ============================================================
RUN_MODE = "grid"   # "grid"  or  "champion"


# ============================================================
# DEFAULT CONFIGURATION — overridable from the command line
# ============================================================
CFG = {
    "ticker":           "SPY",        # default; override with --ticker
    "n_qubits":         10,          # default; override with --n-qubits
    "lookback":         5,
    "hankel_window":    3,
    "forecast_horizon": 5,
    "val_split":        0.6,         # validation starts here
    "train_split":      0.8,         # test starts here
    "seed":             42,
    "data_dir":         DATA_DIR,    # price snapshots (see market_data.py)
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
    "data_period":  ["1y", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "10y"],  # default; override with --periods
    "n_layers":     list(range(1, 11)),
    "n_reservoirs": list(range(1, 11)),
}

# Seeds grow as n_reservoirs increases (accumulating diversity)
SEED_POOL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Champion evaluation: replicate r uses seeds SEED_POOL[0] + SEED_STRIDE*r + [0..n_res-1].
# Replicate 0 reproduces the grid-search seeds; replicates never share a seed
# as long as n_reservoirs < SEED_STRIDE.
N_EVAL_SEEDS = 10
SEED_STRIDE  = 100


def set_output_files() -> None:
    """Derive output file names from CFG (called again after CLI overrides)."""
    global RESULTS_FILE, GRID_FILES_GLOB, CHAMPION_SEEDS_FILE, CHAMPION_SUMMARY_FILE, PREDICTIONS_FILE
    RESULTS_FILE          = f"qrc_grid_{CFG['ticker']}_q{CFG['n_qubits']}_val.csv"
    GRID_FILES_GLOB       = f"qrc_grid_{CFG['ticker']}_q*_val.csv"
    CHAMPION_SEEDS_FILE   = f"qrc_champion_{CFG['ticker']}_seeds.csv"
    CHAMPION_SUMMARY_FILE = f"qrc_champion_{CFG['ticker']}_summary.csv"
    PREDICTIONS_FILE      = f"qrc_predictions_{CFG['ticker']}.csv"


set_output_files()

RIDGE_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
PURGE_GAP    = CFG["forecast_horizon"] - 1   # training samples dropped before each split
METRICS      = ["rmse", "qlike", "mz_alpha", "mz_beta", "mz_r2"]


def replicate_seeds(n_reservoirs: int, replicate: int) -> list:
    base = SEED_POOL[0] + SEED_STRIDE * replicate
    return [base + j for j in range(n_reservoirs)]


# ============================================================
# DATA  (cached so yfinance is not hit repeatedly)
# ============================================================
_data_cache: dict = {}


def get_market_data(period: str):
    """
    Load price data for `period` from the local snapshot and return
    (returns, log_vol_targets, target_dates).

    Returns
    -------
    returns      : np.ndarray, shape (N,)   — daily returns
    targets      : np.ndarray, shape (N,)   — rolling 5-day log-volatility
    target_dates : np.ndarray, shape (N,)   — last trading day of each target's
                                              5-day window (matches benchmark_classical.py)
    """
    if period in _data_cache:
        return _data_cache[period]

    close_s    = load_close(CFG["ticker"], PERIOD_STARTS[period], END_DATE, CFG["data_dir"])
    close      = close_s.values
    returns    = np.diff(close) / close[:-1]

    vol_window = 5
    raw_vol    = np.array(
        [np.std(returns[i - vol_window:i]) for i in range(vol_window, len(returns))]
    )
    targets = np.log(np.clip(raw_vol, 1e-10, None))
    # raw_vol[j] covers returns[j:j+5]; its last return is dated close_s.index[j+5]
    dates   = close_s.index[vol_window:vol_window + len(raw_vol)].strftime("%Y-%m-%d").values
    returns = returns[vol_window:]

    min_len = min(len(returns), len(targets))
    returns, targets, dates = returns[:min_len], targets[:min_len], dates[:min_len]

    mask = ~(np.isnan(returns) | np.isnan(targets))
    result = returns[mask], targets[mask], dates[mask]

    _data_cache[period] = result
    return result


def split_points(period: str):
    """Return (val_start, test_start, n_rows) in aligned-sample coordinates."""
    returns, _, _ = get_market_data(period)
    offset = CFG["lookback"] + CFG["hankel_window"] - 1 + CFG["forecast_horizon"]
    n      = len(returns) - offset          # rows with both features and a target
    return int(n * CFG["val_split"]), int(n * CFG["train_split"]), n


def build_dataset(period: str, n_qubits: int, n_layers: int, seeds: list, n_rows: int):
    """
    Hankel features X, log-vol targets y and target dates for the first
    `n_rows` aligned samples.

    Hankel row m only depends on returns[: m + lookback + hankel_window - 1],
    so truncating the input keeps later (e.g. test-period) data out entirely.
    """
    returns, targets, dates = get_market_data(period)
    L, W   = CFG["lookback"], CFG["hankel_window"]
    offset = L + W - 1 + CFG["forecast_horizon"]

    # ----- Reservoir (from qrc_core) -----
    reservoir = QuantumReservoir(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seeds=seeds,
        global_seed=CFG["seed"],
    )

    # ----- Features (from qrc_core) -----
    X = prepare_hankel_features(
        returns[:n_rows + L + W - 1],
        reservoir,
        lookback=L,
        window=W,
    )
    return X, targets[offset:offset + n_rows], dates[offset:offset + n_rows]


def fit_readout(X_train, y_train, X_eval):
    """Scale + RidgeCV (loose alphas → better MZ beta). Returns (preds, alpha)."""
    scaler = StandardScaler()
    tscv   = TimeSeriesSplit(n_splits=5, gap=PURGE_GAP)
    reg    = RidgeCV(alphas=RIDGE_ALPHAS, cv=tscv)
    reg.fit(scaler.fit_transform(X_train), y_train)
    return reg.predict(scaler.transform(X_eval)), reg.alpha_


def forecast_metrics(y_log, preds) -> dict:
    """RMSE / QLIKE in raw-vol space, Mincer-Zarnowitz in log space."""
    preds_clipped  = np.clip(np.exp(preds),  1e-8, None)
    y_raw          = np.exp(y_log)
    y_clipped      = np.clip(y_raw, 1e-8, None)

    rmse  = np.sqrt(mean_squared_error(y_raw, preds_clipped))
    qlike = np.mean(
        y_clipped / preds_clipped
        - np.log(y_clipped / preds_clipped)
        - 1
    )

    mz_X = np.column_stack([np.ones(len(preds)), preds])
    mz   = sm.OLS(y_log, mz_X).fit()
    return {
        "rmse":     round(rmse,                   6),
        "qlike":    round(qlike,                  6),
        "mz_alpha": round(float(mz.params[0]),    6),
        "mz_beta":  round(float(mz.params[1]),    6),
        "mz_r2":    round(float(mz.rsquared),     4),
    }


# ============================================================
# SINGLE EXPERIMENT RUN  (grid search — validation only)
# ============================================================

def run_experiment(period: str, n_layers: int, n_reservoirs: int) -> dict:
    seeds = SEED_POOL[:n_reservoirs]
    val_start, test_start, _ = split_points(period)

    X, y, _ = build_dataset(period, CFG["n_qubits"], n_layers, seeds, n_rows=test_start)
    fit_end = val_start - PURGE_GAP
    preds, alpha = fit_readout(X[:fit_end], y[:fit_end], X[val_start:])
    metrics = forecast_metrics(y[val_start:], preds)

    return {
        "period":       period,
        "n_qubits":     CFG["n_qubits"],
        "n_layers":     n_layers,
        "n_reservoirs": n_reservoirs,
        "seeds":        str(seeds),
        "n_features":   X.shape[1],
        "best_alpha":   alpha,
        **{f"val_{k}": v for k, v in metrics.items()},
        "train_size":   fit_end,
        "val_size":     test_start - val_start,
    }


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_existing_results(filepath: str = None) -> pd.DataFrame:
    filepath = filepath or RESULTS_FILE
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} existing results from {filepath}")
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
    print(f"  QRC GRID SEARCH — {CFG['ticker']} | {CFG['n_qubits']} qubits (fixed)")
    print(f"  Target: LOG VOLATILITY  |  Scored on VALIDATION set only")
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
                f"          val RMSE={result['val_rmse']:.6f}  "
                f"QLIKE={result['val_qlike']:.6f}  "
                f"R²={result['val_mz_r2']:.4f}  "
                f"MZ_a={result['val_mz_alpha']:.4f}(ideal:0)  "
                f"MZ_b={result['val_mz_beta']:.4f}(ideal:1)  "
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
                "val_rmse": np.nan, "val_qlike": np.nan, "val_mz_r2": np.nan,
            })

        # Checkpoint after every run
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    return pd.DataFrame(results)


# ============================================================
# CHAMPION SELECTION + MULTI-SEED TEST EVALUATION
# ============================================================

def select_champions() -> pd.DataFrame:
    """Lowest validation QLIKE per period, across every qubit count gridded so far."""
    files = sorted(glob.glob(GRID_FILES_GLOB))
    if not files:
        raise FileNotFoundError(f"No grid results match {GRID_FILES_GLOB}; run RUN_MODE='grid' first.")
    print(f"  Selecting champions from: {files}")

    grid = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    grid = grid.dropna(subset=["val_qlike"])
    best = grid.loc[grid.groupby("period")["val_qlike"].idxmin()]
    cols = ["period", "n_qubits", "n_layers", "n_reservoirs",
            "val_rmse", "val_qlike", "val_mz_r2"]
    return best[cols].set_index("period").reindex(
        [p for p in PERIOD_STARTS if p in best["period"].values]
    ).reset_index()


def run_champion_evaluation() -> pd.DataFrame:
    champions = select_champions()
    print(f"\n{'='*60}")
    print(f"  CHAMPIONS (min validation QLIKE) — {CFG['ticker']}")
    print(f"{'='*60}")
    print(champions.to_string(index=False))

    existing  = load_existing_results(CHAMPION_SEEDS_FILE)
    rows      = existing.to_dict("records") if not existing.empty else []
    prev_pred = load_existing_results(PREDICTIONS_FILE)
    pred_rows = prev_pred.to_dict("records") if not prev_pred.empty else []

    print(f"\n  Test evaluation: refit on train+val, {N_EVAL_SEEDS} seed replicates\n")
    for _, c in champions.iterrows():
        period   = c["period"]
        n_qubits, n_layers, n_res = int(c["n_qubits"]), int(c["n_layers"]), int(c["n_reservoirs"])
        _, test_start, n = split_points(period)

        for rep in range(N_EVAL_SEEDS):
            if not existing.empty and (
                (existing["period"] == period) & (existing["replicate"] == rep)
                & (existing["n_qubits"] == n_qubits) & (existing["n_layers"] == n_layers)
                & (existing["n_reservoirs"] == n_res)
            ).any():
                print(f"  [SKIP] period={period} replicate={rep}")
                continue

            seeds = replicate_seeds(n_res, rep)
            print(f"  period={period}  q={n_qubits} L={n_layers} R={n_res}  "
                  f"replicate={rep}  seeds={seeds}")
            t0 = time.time()
            X, y, dates  = build_dataset(period, n_qubits, n_layers, seeds, n_rows=n)
            fit_end      = test_start - PURGE_GAP
            preds, alpha = fit_readout(X[:fit_end], y[:fit_end], X[test_start:])
            metrics      = forecast_metrics(y[test_start:], preds)
            print(f"          test RMSE={metrics['rmse']:.6f}  QLIKE={metrics['qlike']:.6f}  "
                  f"R²={metrics['mz_r2']:.4f}  MZ_b={metrics['mz_beta']:.4f}  "
                  f"({time.time() - t0:.1f}s)")

            rows.append({
                "period": period, "n_qubits": n_qubits, "n_layers": n_layers,
                "n_reservoirs": n_res, "replicate": rep, "seeds": str(seeds),
                "best_alpha": alpha, **metrics,
                "train_size": fit_end, "test_size": n - test_start,
            })
            pred_rows.extend(
                {"ticker": CFG["ticker"], "period": period, "model": "QRC", "seed": rep,
                 "date": d, "y_true_log": yt, "y_pred_log": yp}
                for d, yt, yp in zip(dates[test_start:], y[test_start:], preds)
            )

            # Checkpoint after every replicate
            pd.DataFrame(rows).to_csv(CHAMPION_SEEDS_FILE, index=False)
            pd.DataFrame(pred_rows).to_csv(PREDICTIONS_FILE, index=False)

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY + HEATMAPS
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH COMPLETE — {CFG['n_qubits']} qubits | LOG VOL target")
    print(f"  VALIDATION metrics  |  MZ in log space  |  RMSE/QLIKE in raw-vol space")
    print(f"{'='*60}\n")

    valid = df.dropna(subset=["val_rmse"])
    cols  = ["period", "n_layers", "n_reservoirs", "seeds",
             "val_rmse", "val_qlike", "val_mz_r2", "best_alpha"]

    for label, key, ascending in [
        ("RMSE (lower is better)",               "val_rmse",   True),
        ("QLIKE (lower is better)",              "val_qlike",  True),
        ("Mincer-Zarnowitz R² (higher is better)", "val_mz_r2", False),
    ]:
        fn  = valid.nsmallest if ascending else valid.nlargest
        top = fn(5, key)[cols]
        print(f"  Top 5 by validation {label}:")
        print("  " + "-" * 56)
        print(top.to_string(index=False))
        print()

    for title, idx, col in [
        ("layers × reservoirs (avg across periods)",    "n_layers",     "n_reservoirs"),
        ("layers × data period (avg across reservoirs)","n_layers",     "period"),
        ("reservoirs × data period (avg across layers)","n_reservoirs", "period"),
    ]:
        pivot = valid.pivot_table(
            values="val_qlike", index=idx, columns=col, aggfunc="mean"
        ).round(6)
        pivot.index.name = idx.replace("n_", "").replace("_", " ") + " \\ " + col.replace("n_", "")
        print(f"  Validation QLIKE Heatmap: {title}")
        print("  " + "-" * 40)
        print(pivot.to_string())
        print()

    print(f"  Full results saved to: {RESULTS_FILE}")
    print(f"  Next: run every --n-qubits, then --mode champion.")


def print_champion_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["period", "n_qubits", "n_layers", "n_reservoirs"])[METRICS]
          .agg(["mean", "std", "count"])
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["period"] = pd.Categorical(summary["period"], list(PERIOD_STARTS), ordered=True)
    summary = summary.sort_values("period")
    summary.to_csv(CHAMPION_SUMMARY_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"  CHAMPION TEST RESULTS — {CFG['ticker']} | mean ± std over seed replicates")
    print(f"{'='*60}")
    for _, s in summary.iterrows():
        print(f"  {s['period']:>3}  q={s['n_qubits']} L={s['n_layers']} R={s['n_reservoirs']}  "
              f"n={int(s['rmse_count'])}  "
              f"RMSE={s['rmse_mean']:.6f}±{s['rmse_std']:.6f}  "
              f"QLIKE={s['qlike_mean']:.6f}±{s['qlike_std']:.6f}  "
              f"R²={s['mz_r2_mean']:.4f}±{s['mz_r2_std']:.4f}  "
              f"MZ_b={s['mz_beta_mean']:.4f}±{s['mz_beta_std']:.4f}")
    print(f"\n  Per-replicate results : {CHAMPION_SEEDS_FILE}")
    print(f"  Summary               : {CHAMPION_SUMMARY_FILE}")
    print(f"  Forecasts (DM tests)  : {PREDICTIONS_FILE}")


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="QRC volatility experiments (grid search / champion evaluation).")
    p.add_argument("--mode", choices=["grid", "champion"], default=RUN_MODE)
    p.add_argument("--ticker", default=CFG["ticker"], help="e.g. SPY, QQQ, GLD")
    p.add_argument("--n-qubits", type=int, default=CFG["n_qubits"],
                   help="grid mode only; champion mode reads every qubit count's grid file")
    p.add_argument("--periods", nargs="+", choices=list(PERIOD_STARTS), default=GRID["data_period"],
                   help="grid mode only; subset of data periods to search")
    p.add_argument("--data-dir", default=CFG["data_dir"], help="price snapshot directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RUN_MODE = args.mode
    CFG.update(ticker=args.ticker.upper(), n_qubits=args.n_qubits, data_dir=args.data_dir)
    GRID["data_period"] = args.periods
    set_output_files()

    print(f"\n  RUN_MODE = '{RUN_MODE}'  |  ticker = {CFG['ticker']}"
          + (f"  |  n_qubits = {CFG['n_qubits']}" if RUN_MODE == "grid" else ""))
    if RUN_MODE == "grid":
        results_df = run_grid_search()
        print_summary(results_df)
    elif RUN_MODE == "champion":
        results_df = run_champion_evaluation()
        print_champion_summary(results_df)
    else:
        raise ValueError(f"Unknown RUN_MODE '{RUN_MODE}'. Use 'grid' or 'champion'.")
