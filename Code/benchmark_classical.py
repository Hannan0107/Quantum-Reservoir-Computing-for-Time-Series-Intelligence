# -*- coding: utf-8 -*-

# CLASSICAL TEST CODE:
#
# Protocol (matches experiment_finance_simulation.py):
#   train      = [0,          VAL_SPLIT)
#   validation = [VAL_SPLIT,  TRAIN_SPLIT)  → selects ESN / LSTM lookback (min val QLIKE)
#   test       = [TRAIN_SPLIT, 1]           → selected models refit on train+val,
#                                             evaluated over SEEDS (mean ± std)
# GARCH(1,1) and HAR have no hyperparameters to select and are deterministic,
# so they are evaluated once on test.
#
# Purging: targets[k] spans returns[k:k+5], so the last FORECAST_HORIZON-1
# training targets overlap the first evaluation targets. Those PURGE_GAP
# samples are dropped from every readout / regression fit.
#
# Usage (CLI flags override the defaults below):
#   py -3.11 benchmark_classical.py --ticker SPY
#   py -3.11 benchmark_classical.py --ticker GLD --periods 1y 4y
# Prices come from the local snapshot in market_data.py ({data_dir}/{ticker}_close.csv).

import argparse
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from market_data import DATA_DIR, load_close

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
TICKER           = "SPY"  # default; override with --ticker
FORECAST_HORIZON = 5
VAL_SPLIT        = 0.6    # validation starts here
TRAIN_SPLIT      = 0.8    # test starts here (unchanged from original protocol)
PURGE_GAP        = FORECAST_HORIZON - 1   # training samples dropped before each split

# Seeds for the stochastic baselines (ESN weights, LSTM init/dropout)
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Lookback windows applied to BOTH ESN and LSTM:
#   1  → ESN default (single-step input, current baseline)
#   5  → matches QRC raw lookback
#   7  → matches QRC effective memory (lookback + hankel_window - 1 = 5+3-1)
#   21 → standard ~1-month financial window
LOOKBACKS = [1, 5, 7, 21]

# HAR components (Corsi, 2009): daily, weekly, monthly
HAR_LAGS = (1, 5, 22)

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

# ── CLI overrides ────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Classical volatility baselines (GARCH, HAR, ESN, LSTM).")
_parser.add_argument("--ticker", default=TICKER, help="e.g. SPY, QQQ, GLD")
_parser.add_argument("--periods", nargs="+", choices=list(PERIOD_STARTS), default=list(PERIOD_STARTS),
                     help="subset of data periods to run")
_parser.add_argument("--data-dir", default=DATA_DIR, help="price snapshot directory")
_args = _parser.parse_args()

TICKER   = _args.ticker.upper()
PERIODS  = _args.periods
DATA_DIR = _args.data_dir

RESULTS_FILE     = f"benchmark_logvol_results_{TICKER}.csv"
SUMMARY_FILE     = f"benchmark_logvol_summary_{TICKER}.csv"
VAL_FILE         = f"benchmark_validation_{TICKER}.csv"
PREDICTIONS_FILE = f"classical_predictions_{TICKER}.csv"


# ============================================================
# EVALUATE — expects log-space inputs
# ============================================================
def evaluate(name, y_test, preds):
    # y_test and preds are in log space
    valid = np.isfinite(preds) & np.isfinite(y_test)
    if valid.sum() < 10:
        return {"model": name, "RMSE": np.nan, "QLIKE": np.nan,
                "MZ_alpha": np.nan, "MZ_beta": np.nan, "MZ_R2": np.nan}
    y_log, p_log = y_test[valid], preds[valid]

    # Exp back to raw vol for RMSE and QLIKE — same scale as QRC
    y_raw = np.exp(y_log)
    p_raw = np.clip(np.exp(p_log), 1e-8, None)
    rmse  = np.sqrt(mean_squared_error(y_raw, p_raw))
    qlike = np.mean(y_raw / p_raw - np.log(y_raw / p_raw) - 1)

    # MZ in log space — matches QRC exactly
    mz = sm.OLS(y_log, sm.add_constant(p_log)).fit()
    return {"model": name, "RMSE": rmse, "QLIKE": qlike,
            "MZ_alpha": mz.params[0], "MZ_beta": mz.params[1],
            "MZ_R2": mz.rsquared}


# ============================================================
# DATA — targets = log(raw_vol)
# ============================================================
def get_data(period: str):
    """
    Returns (returns, vol, targets, target_dates).

    target_dates[k] is the last trading day of the 5-day window that
    targets[k] measures — the same convention as the QRC script, so
    forecasts from both pipelines can be matched by date for DM tests.
    """
    close_s = load_close(TICKER, PERIOD_STARTS[period], END_DATE, DATA_DIR)
    close   = close_s.values
    returns = np.diff(close) / close[:-1]

    vol_window = 5
    vol = np.array([np.std(returns[i - vol_window:i])
                    for i in range(vol_window, len(returns))])
    # vol[j] covers returns[j:j+5]; its last return is dated close_s.index[j+5]
    vol_dates = close_s.index[vol_window:vol_window + len(vol)].strftime("%Y-%m-%d").values

    returns = returns[vol_window:]
    min_len = min(len(returns), len(vol))
    returns, vol = returns[:min_len], vol[:min_len]

    # Log-transform targets to match QRC pipeline
    targets      = np.log(np.clip(vol[FORECAST_HORIZON:], 1e-10, None))
    target_dates = vol_dates[FORECAST_HORIZON:min_len]
    vol          = vol[:-FORECAST_HORIZON]
    returns      = returns[:-FORECAST_HORIZON]

    mask = ~(np.isnan(returns) | np.isnan(vol) | np.isnan(targets))
    return returns[mask], vol[mask], targets[mask], target_dates[mask]


# ============================================================
# ESN
# ============================================================
class EchoStateNetwork:
    def __init__(self, n_reservoir=300, spectral_radius=0.9,
                 sparsity=0.1, input_scale=0.5, seed=42):
        rng        = np.random.RandomState(seed)
        self.n_res = n_reservoir
        self.W_in  = rng.uniform(-input_scale, input_scale, n_reservoir)
        W          = rng.randn(n_reservoir, n_reservoir)
        W         *= rng.rand(n_reservoir, n_reservoir) < sparsity
        sr         = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res = W * (spectral_radius / sr) if sr > 0 else W
        self.W_out = None

    def _states(self, inputs):
        x      = np.zeros(self.n_res)
        states = np.zeros((len(inputs), self.n_res))
        for t, v in enumerate(inputs.flatten()):
            x         = np.tanh(self.W_in * v + self.W_res @ x)
            states[t] = x
        return states

    def _augment(self, S):
        """Append a constant-1 bias column so the readout can learn a non-zero mean."""
        return np.hstack([S, np.ones((S.shape[0], 1))])

    def fit(self, X, y, alpha=1.0):
        S               = self._augment(self._states(X))  # (T, n_res+1)
        A               = S.T @ S + alpha * np.eye(S.shape[1])
        self.W_out      = np.linalg.solve(A, S.T @ y.flatten())
        self.last_state = S[-1, :-1].copy()               # store without bias col

    def predict(self, X):
        x, preds = self.last_state.copy(), []
        for v in X.flatten():
            x = np.tanh(self.W_in * v + self.W_res @ x)
            preds.append(float(self.W_out @ np.append(x, 1.0)))  # add bias at predict
        return np.array(preds)


def fit_esn(vol, targets, split, lookback, seed=42):
    """
    Training and predicting ESN with a given lookback window.

    vol     — raw vol input feature
    targets — log vol (from get_data); ESN trains and predicts in log space
    seed    — reservoir weight seed

    Alignment (same proof as LSTM):
      window i = vol[i-lookback+1 : i+1]  (ends at vol[i], info up to returns[i-1])
      yw[j] = targets[j + lookback - 1],  ls = split - lookback + 1
      → yw[ls] = targets[split] = y_test[0]
    """
    # Scale vol input — fit on train only to avoid data leakage
    scaler      = StandardScaler()
    scaler.fit(vol[:split].reshape(-1, 1))
    vol_sc_full = scaler.transform(vol.reshape(-1, 1)).flatten()

    if lookback == 1:
        # Original single-step ESN — feed full sequence, recurrent memory intact.
        # Readout is fit on [0, split - PURGE_GAP); the purged steps still run
        # through the reservoir so the state entering the test window is current.
        fit_end = split - PURGE_GAP
        esn = EchoStateNetwork(seed=seed)
        esn.fit(vol_sc_full[:fit_end].reshape(-1, 1), targets[:fit_end])
        preds = esn.predict(vol_sc_full[fit_end:].reshape(-1, 1))[PURGE_GAP:]
        y_te  = targets[split:]
    else:
        # Windowed ESN — mirrors LSTM windowing for fair comparison
        Xw, yw = [], []
        for i in range(lookback - 1, len(vol_sc_full)):
            Xw.append(vol_sc_full[i - lookback + 1:i + 1])
            yw.append(targets[i])
        Xw, yw = np.array(Xw), np.array(yw)

        ls   = split - lookback + 1
        X_tr = Xw[:ls - PURGE_GAP]
        X_te = Xw[ls:]
        y_tr = yw[:ls - PURGE_GAP]
        y_te = yw[ls:]   # aligned: y_te[0] == targets[split] == y_test[0]

        # Train: run each window through reservoir, collect final states + bias
        esn = EchoStateNetwork(seed=seed)
        train_states = []
        for window in X_tr:
            x = np.zeros(esn.n_res)
            for v in window:
                x = np.tanh(esn.W_in * v + esn.W_res @ x)
            train_states.append(np.append(x, 1.0))   # bias column
        train_states = np.array(train_states)         # (N_tr, n_res+1)

        A         = train_states.T @ train_states + 1.0 * np.eye(esn.n_res + 1)
        esn.W_out = np.linalg.solve(A, train_states.T @ y_tr)

        # Predict: same window-based state extraction + bias
        preds = []
        for window in X_te:
            x = np.zeros(esn.n_res)
            for v in window:
                x = np.tanh(esn.W_in * v + esn.W_res @ x)
            preds.append(float(esn.W_out @ np.append(x, 1.0)))
        preds = np.array(preds)

    return preds, y_te


# ============================================================
# LSTM HELPER
# ============================================================
def fit_lstm(vol, targets, split, lookback, seed=42):
    """
    Training and predicting LSTM with a given lookback window.

    vol     — raw vol input feature
    targets — log vol (from get_data); LSTM trains and predicts in log space
    seed    — seeds Python, NumPy and TensorFlow (weight init, dropout, shuffling)

    Alignment proof:
      window i = vol[i-lookback+1 : i+1]  (ends at vol[i], info up to returns[i-1])
      yl[j] = targets[j + lookback - 1]   (loop starts at range(lookback - 1, ...))
      ls    = split - lookback + 1
      yl[ls]= targets[split] = y_test[0]  offset cancels exactly
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping


    scaler = StandardScaler()
    scaler.fit(vol[:split].reshape(-1, 1))
    vol_sc = scaler.transform(vol.reshape(-1, 1)).flatten()

    Xl, yl = [], []
    for i in range(lookback - 1, len(vol_sc)):
        Xl.append(vol_sc[i - lookback + 1:i + 1])
        yl.append(targets[i])
    Xl, yl = np.array(Xl), np.array(yl)

    ls   = split - lookback + 1
    X_tr = Xl[:ls - PURGE_GAP].reshape(-1, lookback, 1)
    X_te = Xl[ls:].reshape(-1, lookback, 1)
    y_tr = yl[:ls - PURGE_GAP]
    y_te = yl[ls:]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    mdl = Sequential([
        LSTM(64, input_shape=(lookback, 1)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    mdl.compile(optimizer='adam', loss='mse')
    mdl.fit(X_tr, y_tr,
            epochs=100, batch_size=32, validation_split=0.1,
            callbacks=[EarlyStopping(patience=10,
                                     restore_best_weights=True, verbose=0)],
            verbose=0)

    preds = mdl.predict(X_te, verbose=0).flatten()
    return preds, y_te


# ============================================================
# HAR (log-HAR on squared daily returns)
# ============================================================
def har_features(returns):
    """
    Row k = log of the mean squared return over the 1 / 5 / 22 days strictly
    before index k (returns[k-lag:k]).

    targets[k] measures returns[k:k+5] (see get_data), so features and target
    never overlap. Rows with fewer than max(HAR_LAGS) days of history are NaN.
    """
    r2   = returns ** 2
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    X    = np.full((len(returns), len(HAR_LAGS)), np.nan)
    for j, lag in enumerate(HAR_LAGS):
        k       = np.arange(lag, len(returns))
        X[k, j] = (csum[k] - csum[k - lag]) / lag
    return np.log(np.clip(X, 1e-12, None))


def fit_har(returns, targets, split):
    """
    log σ_{t+1:t+5} = b0 + b_d log RV_d + b_w log RV_w + b_m log RV_m,
    fit by OLS on [max(HAR_LAGS), split - PURGE_GAP); predicts [split, end) in log space.
    """
    X     = har_features(returns)
    tr    = np.arange(max(HAR_LAGS), split - PURGE_GAP)
    ols   = sm.OLS(targets[tr], sm.add_constant(X[tr])).fit()
    preds = ols.predict(sm.add_constant(X[split:], has_constant='add'))
    return preds, targets[split:]


# ============================================================
# MAIN LOOP
# ============================================================
from arch import arch_model

NAN_METRICS = {"RMSE": np.nan, "QLIKE": np.nan,
               "MZ_alpha": np.nan, "MZ_beta": np.nan, "MZ_R2": np.nan}
FITTERS     = {"ESN": fit_esn, "LSTM": fit_lstm}


def print_metrics(r):
    print(f"    RMSE={r['RMSE']:.6f}  QLIKE={r['QLIKE']:.6f}  "
          f"MZ_β={r['MZ_beta']:.4f}  R²={r['MZ_R2']:.4f}")


all_results     = []   # test metrics, one row per (period, model, seed)
val_results     = []   # validation metrics, one row per (period, model, lookback)
all_predictions = []   # per-date test forecasts for DM tests

for period in PERIODS:
    print(f"\n{'#'*60}")
    print(f"  PERIOD: {period}  ({PERIOD_STARTS[period]} → {END_DATE})")
    print(f"{'#'*60}")

    try:
        returns, vol, targets, dates = get_data(period)
    except Exception as e:
        print(f"  Data error: {e}")
        continue

    val_split  = int(len(vol) * VAL_SPLIT)
    split      = int(len(vol) * TRAIN_SPLIT)
    y_test     = targets[split:]   # log vol
    test_dates = dates[split:]
    print(f"  Samples: {len(vol)}  |  Train: {val_split}  |  "
          f"Val: {split - val_split}  |  Test: {len(y_test)}")

    def record_test(model, preds, y_te, seed=np.nan, lookback=np.nan):
        """Evaluate one test forecast; store metrics and per-date predictions."""
        n = min(len(preds), len(y_te))
        r = evaluate(model, y_te[:n], preds[:n])
        r.update(period=period, seed=seed, lookback=lookback)
        all_results.append(r)
        all_predictions.extend(
            {"ticker": TICKER, "period": period, "model": model, "seed": seed,
             "date": d, "y_true_log": yt, "y_pred_log": yp}
            for d, yt, yp in zip(test_dates[:n], y_te[:n], preds[:n])
        )
        print_metrics(r)

    # ── GARCH ──────────────────────────────────────────────
    # GARCH always outputs raw vol — log-transform before evaluate().
    # Target is vol over the next FORECAST_HORIZON days, so the matching forecast
    # is sqrt of the mean conditional variance over steps 1..h.
    # GARCH is fit on returns up to the forecast origin only — no target overlap,
    # so no purge needed.
    print(f"  Fitting GARCH(1,1)...")
    full_r      = returns * 100
    garch_preds = []
    for i in range(split, len(returns)):
        try:
            m   = arch_model(full_r[:i], vol='Garch', p=1, q=1, dist='normal')
            res = m.fit(disp='off', show_warning=False)
            fc  = res.forecast(horizon=FORECAST_HORIZON)
            garch_preds.append(np.sqrt(fc.variance.values[-1, :].mean()) / 100)
        except Exception:
            garch_preds.append(np.nan)
    garch_preds_log = np.log(np.clip(np.array(garch_preds), 1e-10, None))
    record_test("GARCH(1,1)", garch_preds_log, y_test)

    # ── HAR ────────────────────────────────────────────────
    print(f"  Fitting HAR...")
    har_preds, y_te = fit_har(returns, targets, split)
    record_test("HAR", har_preds, y_te)

    # ── ESN + LSTM: lookback selection on validation ───────
    # Fit on train only (data truncated at `split`, so the "test" portion
    # returned by the fitters is the validation window).
    for lb in LOOKBACKS:
        for family, fit in FITTERS.items():
            # LSTM skips lb=1 — LSTM needs at least a small sequence
            if family == "LSTM" and lb == 1:
                continue
            print(f"  [val] {family}(lb={lb})...")
            try:
                p, y_v = fit(vol[:split], targets[:split], val_split, lb, seed=SEEDS[0])
                n      = min(len(p), len(y_v))
                r      = evaluate(family, y_v[:n], p[:n])
            except Exception as e:
                print(f"    ERROR: {e}")
                r = {"model": family, **NAN_METRICS}
            r.update(period=period, lookback=lb)
            val_results.append(r)
            print_metrics(r)

    # ── ESN + LSTM: selected lookback, all seeds, on test ──
    val_df = pd.DataFrame([r for r in val_results if r["period"] == period])
    for family, fit in FITTERS.items():
        fam_val = val_df[val_df["model"] == family].dropna(subset=["QLIKE"])
        if fam_val.empty:
            print(f"  No valid {family} lookback on validation — skipping test.")
            continue
        best_lb = int(fam_val.loc[fam_val["QLIKE"].idxmin(), "lookback"])
        print(f"  {family}: selected lb={best_lb} (min val QLIKE); "
              f"testing over {len(SEEDS)} seeds")
        for seed in SEEDS:
            print(f"  [test] {family}(lb={best_lb}) seed={seed}...")
            try:
                p, y_te = fit(vol, targets, split, best_lb, seed=seed)
                record_test(family, p, y_te, seed=seed, lookback=best_lb)
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({"model": family, **NAN_METRICS, "period": period,
                                    "seed": seed, "lookback": best_lb})

# ============================================================
# SUMMARY TABLES
# ============================================================
METRICS = ["RMSE", "QLIKE", "MZ_alpha", "MZ_beta", "MZ_R2"]
ORDER   = [p for p in PERIOD_STARTS if p in PERIODS]

df = pd.DataFrame(all_results)
df = df[["period", "model", "lookback", "seed"] + METRICS]

val_df = pd.DataFrame(val_results)[["period", "model", "lookback"] + METRICS]
print(f"\n\n{'='*80}")
print("  VALIDATION — lookback selection (min QLIKE per model family)")
print(f"{'='*80}")
print(val_df.pivot_table(values="QLIKE", index="period", columns=["model", "lookback"])
      .reindex(ORDER).round(6).to_string())

summary = (df.groupby(["period", "model"])
             .agg(lookback=("lookback", "first"), n_seeds=("RMSE", "count"),
                  **{f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "std")})
             .reset_index())

print(f"\n\n{'='*80}")
print(f"  TEST — mean ± std over {len(SEEDS)} seeds (ESN/LSTM); "
      f"single run (GARCH/HAR, deterministic)")
print("  RMSE/QLIKE in raw vol space | MZ in log vol space")
print(f"{'='*80}")
for metric in ["RMSE", "QLIKE", "MZ_R2"]:
    fmt = summary.assign(
        cell=summary[f"{metric}_mean"].map("{:.6f}".format)
             + summary[f"{metric}_std"].map(lambda s: "" if np.isnan(s) else f" ± {s:.6f}")
    )
    print(f"\n  {metric} by period × model")
    print(fmt.pivot(index="period", columns="model", values="cell").reindex(ORDER).to_string())

df.to_csv(RESULTS_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
pd.DataFrame(all_predictions).to_csv(PREDICTIONS_FILE, index=False)
print(f"\n  Saved: {RESULTS_FILE} (per seed), {SUMMARY_FILE} (mean/std), "
      f"{VAL_FILE} (validation), {PREDICTIONS_FILE} (forecasts for dm_test.py)")
print(f"\n  Lookback note:")
print(f"    lb=1  → single-step input (ESN original, recurrent memory intact)")
print(f"    lb=5  → matches QRC raw lookback")
print(f"    lb=7  → matches QRC effective memory (lookback + hankel_window - 1)")
print(f"    lb=21 → standard ~1-month financial window")
