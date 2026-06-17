# -*- coding: utf-8 -*-

# CLASSICAL TEST CODE:

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
TICKER           = "SPY"  # Change to QQQ, GLD manually
FORECAST_HORIZON = 5
TRAIN_SPLIT      = 0.8

# Lookback windows applied to BOTH ESN and LSTM:
#   1  → ESN default (single-step input, current baseline)
#   5  → matches QRC raw lookback
#   7  → matches QRC effective memory (lookback + hankel_window - 1 = 5+3-1)
#   21 → standard ~1-month financial window
LOOKBACKS = [1, 5, 7, 21]

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
    df = yf.download(
        TICKER,
        start=PERIOD_STARTS[period],
        end=END_DATE,
        interval="1d",
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data for {TICKER} period={period}.")

    close   = df['Close'].values.flatten()
    returns = np.diff(close) / close[:-1]

    vol_window = 5
    vol = np.array([np.std(returns[i - vol_window:i])
                    for i in range(vol_window, len(returns))])

    returns = returns[vol_window:]
    min_len = min(len(returns), len(vol))
    returns, vol = returns[:min_len], vol[:min_len]

    # Log-transform targets to match QRC pipeline
    targets = np.log(np.clip(vol[FORECAST_HORIZON:], 1e-10, None))
    vol     = vol[:-FORECAST_HORIZON]
    returns = returns[:-FORECAST_HORIZON]

    mask = ~(np.isnan(returns) | np.isnan(vol) | np.isnan(targets))
    return returns[mask], vol[mask], targets[mask]


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


def fit_esn(vol, targets, split, lookback):
    """
    Training and predicting ESN with a given lookback window.

    vol     — raw vol input feature
    targets — log vol (from get_data); ESN trains and predicts in log space

    Alignment (same proof as LSTM):
      yl[i] = targets[i + lookback],  ls = split - lookback
      → yl[ls] = targets[split] = y_test[0]
    """
    # Scale vol input — fit on train only to avoid data leakage
    scaler      = StandardScaler()
    scaler.fit(vol[:split].reshape(-1, 1))
    vol_sc_full = scaler.transform(vol.reshape(-1, 1)).flatten()

    if lookback == 1:
        # Original single-step ESN — feed full sequence, recurrent memory intact
        esn = EchoStateNetwork()
        esn.fit(vol_sc_full[:split].reshape(-1, 1), targets[:split])
        preds = esn.predict(vol_sc_full[split:].reshape(-1, 1))
        y_te  = targets[split:]
    else:
        # Windowed ESN — mirrors LSTM windowing for fair comparison
        Xw, yw = [], []
        for i in range(lookback, len(vol_sc_full)):
            Xw.append(vol_sc_full[i - lookback:i])
            yw.append(targets[i])
        Xw, yw = np.array(Xw), np.array(yw)

        ls   = split - lookback
        X_tr = Xw[:ls]
        X_te = Xw[ls:]
        y_tr = yw[:ls]
        y_te = yw[ls:]   # aligned: y_te[0] == targets[split] == y_test[0]

        # Train: run each window through reservoir, collect final states + bias
        esn = EchoStateNetwork()
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
def fit_lstm(vol, targets, split, lookback):
    """
    Training and predicting LSTM with a given lookback window.

    vol     — raw vol input feature
    targets — log vol (from get_data); LSTM trains and predicts in log space

    Alignment proof:
      yl[i] = targets[i + lookback]  (loop starts at range(lookback, ...))
      ls    = split - lookback
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
    for i in range(lookback, len(vol_sc)):
        Xl.append(vol_sc[i - lookback:i])
        yl.append(targets[i])
    Xl, yl = np.array(Xl), np.array(yl)

    ls   = split - lookback
    X_tr = Xl[:ls].reshape(-1, lookback, 1)
    X_te = Xl[ls:].reshape(-1, lookback, 1)
    y_tr = yl[:ls]
    y_te = yl[ls:]   

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

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
# MAIN LOOP
# ============================================================
from arch import arch_model

all_results = []

for period in PERIOD_STARTS:
    print(f"\n{'#'*60}")
    print(f"  PERIOD: {period}  ({PERIOD_STARTS[period]} → {END_DATE})")
    print(f"{'#'*60}")

    try:
        returns, vol, targets = get_data(period)
    except Exception as e:
        print(f"  Data error: {e}")
        continue

    split  = int(len(vol) * TRAIN_SPLIT)
    y_test = targets[split:]   # log vol
    print(f"  Samples: {len(vol)}  |  Train: {split}  |  Test: {len(y_test)}")

    period_results = []

    # ── GARCH ──────────────────────────────────────────────
    # GARCH always outputs raw vol — log-transform before evaluate()
    print(f"  Fitting GARCH(1,1)...")
    full_r      = returns * 100
    garch_preds = []
    for i in range(split, len(returns)):
        try:
            m   = arch_model(full_r[:i], vol='Garch', p=1, q=1, dist='normal')
            res = m.fit(disp='off', show_warning=False)
            fc  = res.forecast(horizon=FORECAST_HORIZON)
            garch_preds.append(np.sqrt(fc.variance.values[-1, -1]) / 100)
        except Exception:
            garch_preds.append(np.nan)
    garch_preds     = np.array(garch_preds)
    garch_preds_log = np.log(np.clip(garch_preds, 1e-10, None))
    min_len         = min(len(garch_preds_log), len(y_test))
    r               = evaluate("GARCH(1,1)", y_test[:min_len], garch_preds_log[:min_len])
    r["period"] = period
    period_results.append(r)
    print(f"    RMSE={r['RMSE']:.6f}  QLIKE={r['QLIKE']:.6f}  "
          f"MZ_β={r['MZ_beta']:.4f}  R²={r['MZ_R2']:.4f}")

    # ── ESN + LSTM (all lookbacks) ─────────────────────────
    # targets are already log vol; predictions come out in log space naturally
    for lb in LOOKBACKS:
        # ESN
        label = f"ESN(lb={lb})"
        print(f"  Fitting {label}...")
        try:
            esn_preds, y_te = fit_esn(vol, targets, split, lb)
            min_len         = min(len(esn_preds), len(y_te))
            r               = evaluate(label, y_te[:min_len], esn_preds[:min_len])
        except Exception as e:
            print(f"    ERROR: {e}")
            r = {"model": label, "RMSE": np.nan, "QLIKE": np.nan,
                 "MZ_alpha": np.nan, "MZ_beta": np.nan, "MZ_R2": np.nan}
        r["period"] = period
        period_results.append(r)
        print(f"    RMSE={r['RMSE']:.6f}  QLIKE={r['QLIKE']:.6f}  "
              f"MZ_β={r['MZ_beta']:.4f}  R²={r['MZ_R2']:.4f}")

        # LSTM (skip lb=1 — LSTM needs at least a small sequence)
        if lb == 1:
            continue
        label = f"LSTM(lb={lb})"
        print(f"  Fitting {label}...")
        try:
            lstm_preds, y_te = fit_lstm(vol, targets, split, lb)
            min_len          = min(len(lstm_preds), len(y_te))
            r                = evaluate(label, y_te[:min_len], lstm_preds[:min_len])
        except Exception as e:
            print(f"    ERROR: {e}")
            r = {"model": label, "RMSE": np.nan, "QLIKE": np.nan,
                 "MZ_alpha": np.nan, "MZ_beta": np.nan, "MZ_R2": np.nan}
        r["period"] = period
        period_results.append(r)
        print(f"    RMSE={r['RMSE']:.6f}  QLIKE={r['QLIKE']:.6f}  "
              f"MZ_β={r['MZ_beta']:.4f}  R²={r['MZ_R2']:.4f}")

    all_results.extend(period_results)

# ============================================================
# SUMMARY TABLES
# ============================================================
df = pd.DataFrame(all_results)
df = df[["period", "model", "RMSE", "QLIKE", "MZ_alpha", "MZ_beta", "MZ_R2"]]

print(f"\n\n{'='*80}")
print("  FULL RESULTS — RMSE/QLIKE in raw vol space | MZ in log vol space")
print(f"{'='*80}")
print(df.to_string(index=False))

for metric in ["RMSE", "QLIKE", "MZ_R2"]:
    print(f"\n\n{'='*80}")
    print(f"  PIVOT: {metric} by period × model")
    print(f"{'='*80}")
    pivot = df.pivot_table(values=metric, index="period", columns="model")
    order = [p for p in PERIOD_STARTS if p in pivot.index]
    print(pivot.loc[order].round(6).to_string())

# Best ESN and best LSTM per period
print(f"\n\n{'='*80}")
print("  BEST ESN per period (min RMSE across lookbacks)")
print(f"{'='*80}")
esn_df = df[df['model'].str.startswith('ESN')]
best_esn = esn_df.loc[esn_df.groupby('period')['RMSE'].idxmin()]
print(best_esn[['period','model','RMSE','QLIKE','MZ_beta','MZ_R2']].to_string(index=False))

print(f"\n\n{'='*80}")
print("  BEST LSTM per period (min RMSE across lookbacks)")
print(f"{'='*80}")
lstm_df = df[df['model'].str.startswith('LSTM')]
best_lstm = lstm_df.loc[lstm_df.groupby('period')['RMSE'].idxmin()]
print(best_lstm[['period','model','RMSE','QLIKE','MZ_beta','MZ_R2']].to_string(index=False))

df.to_csv("benchmark_logvol_results.csv", index=False)
print(f"\n  Saved to: benchmark_logvol_results.csv")
print(f"\n  Lookback note:")
print(f"    lb=1  → single-step input (ESN original, recurrent memory intact)")
print(f"    lb=5  → matches QRC raw lookback")
print(f"    lb=7  → matches QRC effective memory (lookback + hankel_window - 1)")
print(f"    lb=21 → standard ~1-month financial window")

