"""
dm_test.py
==========
Diebold-Mariano tests: QRC champion vs each classical baseline.

Usage:  py -3.11 dm_test.py --ticker SPY

Inputs (same ticker, run from the same directory)
-------------------------------------------------
  qrc_predictions_{ticker}.csv        — experiment_finance_simulation.py, RUN_MODE="champion"
  classical_predictions_{ticker}.csv  — benchmark_classical.py

Method
------
  - Forecasts are matched by target date, so both models are compared on
    exactly the same observations (the two pipelines lose different numbers
    of leading samples, so their 80% test windows differ by a few days).
  - Losses are computed in raw-vol space, matching the reported metrics:
        SE    = (σ - σ̂)²
        QLIKE = σ/σ̂ - log(σ/σ̂) - 1
  - For seeded models the per-date loss is averaged over seeds (expected loss
    over reservoir / initialisation randomness) before testing.
  - d_t = L_QRC,t - L_baseline,t. Long-run variance uses h-1 autocovariances
    (5-day targets overlap, so errors are MA(h-1) under the null), with the
    Harvey-Leybourne-Newbold (1997) small-sample correction and t(T-1)
    critical values. Falls back to Bartlett weights if the rectangular
    estimate is non-positive.
  - As a robustness check, the test is also run per seed pair (QRC replicate i
    vs baseline seed i) and the number of significant pairs is reported.

Interpretation: dm_stat < 0 means QRC has lower loss. p_qrc_better is the
one-sided p-value for H1: E[d] < 0.

Multiple comparisons: Holm (1979) step-down adjustment, controlling the
family-wise error rate. One family per loss function = all periods × baselines
for this ticker. Adjusted p-values are reported as p_two_sided_holm and
p_qrc_better_holm; significance markers in the printed summary use the
Holm-adjusted two-sided p-value. The seed-pair robustness counts are
unadjusted.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================
CFG = {
    "ticker":           "SPY",    # default; override with --ticker
    "forecast_horizon": 5,
    "alpha":            0.05,
}


def set_files() -> None:
    """Derive input / output file names from CFG (called again after CLI overrides)."""
    global QRC_FILE, CLASSICAL_FILE, OUTPUT_FILE
    QRC_FILE       = f"qrc_predictions_{CFG['ticker']}.csv"
    CLASSICAL_FILE = f"classical_predictions_{CFG['ticker']}.csv"
    OUTPUT_FILE    = f"dm_results_{CFG['ticker']}.csv"


set_files()

PERIOD_ORDER = ["1y", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "10y"]


def _se(y, p):
    return (y - p) ** 2


def _qlike(y, p):
    return y / p - np.log(y / p) - 1


LOSSES = {"SE": _se, "QLIKE": _qlike}


# ============================================================
# DIEBOLD-MARIANO (HLN-corrected)
# ============================================================

def diebold_mariano(d, h: int):
    """
    Returns (hln_stat, p_two_sided, p_one_sided_less) for loss differential d.
    p_one_sided_less tests H1: E[d] < 0 (first model more accurate).
    """
    d  = np.asarray(d, dtype=float)
    T  = len(d)
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:T - k]) / T for k in range(h)]

    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        lrv = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    if lrv <= 0:
        return np.nan, np.nan, np.nan

    dm  = d.mean() / np.sqrt(lrv / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return hln, 2 * stats.t.sf(abs(hln), df=T - 1), stats.t.cdf(hln, df=T - 1)


def holm_adjust(p):
    """
    Holm step-down adjusted p-values. NaNs are ignored (excluded from the
    family size m) and returned as NaN.
    """
    p     = np.asarray(p, dtype=float)
    adj   = np.full_like(p, np.nan)
    ok    = np.flatnonzero(~np.isnan(p))
    m     = len(ok)
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i]  = running
    return adj


# ============================================================
# LOSS SERIES
# ============================================================

def add_losses(df: pd.DataFrame) -> pd.DataFrame:
    y = np.clip(np.exp(df["y_true_log"]), 1e-8, None)
    p = np.clip(np.exp(df["y_pred_log"]), 1e-8, None)
    return df.assign(**{name: fn(y, p) for name, fn in LOSSES.items()})


def check_alignment(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    """Both pipelines must have the same realised target on a shared date."""
    ya = a.groupby("date")["y_true_log"].first()
    yb = b.groupby("date")["y_true_log"].first()
    common = ya.index.intersection(yb.index)
    if not np.allclose(ya[common], yb[common], atol=1e-8):
        raise ValueError(f"{label}: y_true differs on shared dates — targets are misaligned.")


# ============================================================
# MAIN
# ============================================================

def run_dm_tests() -> pd.DataFrame:
    qrc = add_losses(pd.read_csv(QRC_FILE))
    cls = add_losses(pd.read_csv(CLASSICAL_FILE))
    h   = CFG["forecast_horizon"]

    rows = []
    periods = [p for p in PERIOD_ORDER if p in set(qrc["period"]) & set(cls["period"])]
    for period in periods:
        q_p = qrc[qrc["period"] == period]
        for baseline in cls.loc[cls["period"] == period, "model"].unique():
            b_p = cls[(cls["period"] == period) & (cls["model"] == baseline)]
            check_alignment(q_p, b_p, f"{period}/{baseline}")

            for loss in LOSSES:
                # Seed-averaged loss per date
                lq     = q_p.groupby("date")[loss].mean()
                lb     = b_p.groupby("date")[loss].mean()
                common = lq.index.intersection(lb.index).sort_values()
                d      = lq[common].values - lb[common].values
                stat, p_two, p_less = diebold_mariano(d, h)

                # Robustness: seed-pair tests
                q_seeds = sorted(q_p["seed"].dropna().unique())
                b_seeds = sorted(b_p["seed"].dropna().unique()) or [None]
                better = worse = n_pairs = 0
                for i, qs in enumerate(q_seeds):
                    bs  = b_seeds[i % len(b_seeds)]
                    lqs = q_p[q_p["seed"] == qs].set_index("date")[loss]
                    lbs = (b_p if bs is None else b_p[b_p["seed"] == bs]).set_index("date")[loss]
                    cs  = lqs.index.intersection(lbs.index).sort_values()
                    s, p2, _ = diebold_mariano(lqs[cs].values - lbs[cs].values, h)
                    n_pairs += 1
                    if p2 < CFG["alpha"]:
                        better += s < 0
                        worse  += s > 0

                rows.append({
                    "period": period, "baseline": baseline, "loss": loss,
                    "n_obs": len(common),
                    "mean_loss_qrc": lq[common].mean(), "mean_loss_baseline": lb[common].mean(),
                    "dm_stat": stat, "p_two_sided": p_two, "p_qrc_better": p_less,
                    "n_seed_pairs": n_pairs,
                    "pairs_qrc_sig_better": better, "pairs_qrc_sig_worse": worse,
                })

    df = pd.DataFrame(rows)
    # Holm within each loss family (all periods × baselines)
    for col in ("p_two_sided", "p_qrc_better"):
        df[f"{col}_holm"] = df.groupby("loss")[col].transform(lambda s: holm_adjust(s.values))
    return df


def print_summary(df: pd.DataFrame) -> None:
    a = CFG["alpha"]
    print(f"\n{'='*80}")
    print(f"  DIEBOLD-MARIANO (HLN) — QRC vs baselines | {CFG['ticker']} | h={CFG['forecast_horizon']}")
    print(f"  dm_stat < 0 → QRC lower loss  |  * Holm-adjusted p<{a} (two-sided), "
          f"† unadjusted p<{a} only")
    print(f"{'='*80}")
    for loss in LOSSES:
        sub  = df[df["loss"] == loss].copy()
        mark = np.where(sub["p_two_sided_holm"] < a, "*",
                        np.where(sub["p_two_sided"] < a, "†", " "))
        print(f"\n  Loss = {loss}: family size m={sub['p_two_sided'].notna().sum()}, "
              f"{(sub['p_two_sided'] < a).sum()} significant unadjusted, "
              f"{(sub['p_two_sided_holm'] < a).sum()} after Holm")
        sub["cell"] = (sub["dm_stat"].map("{:+.2f}".format) + mark
                       + " (" + sub["pairs_qrc_sig_better"].astype(str) + "/"
                       + sub["pairs_qrc_sig_worse"].astype(str) + ")")
        print(f"  [cell: DM stat, (seed pairs QRC sig. better / sig. worse, unadjusted)]")
        print(sub.pivot(index="period", columns="baseline", values="cell")
                 .reindex([p for p in PERIOD_ORDER if p in sub["period"].values]).to_string())
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Diebold-Mariano tests: QRC vs classical baselines.")
    p.add_argument("--ticker", default=CFG["ticker"], help="e.g. SPY, QQQ, GLD")
    p.add_argument("--alpha", type=float, default=CFG["alpha"], help="significance level")
    args = p.parse_args()
    CFG.update(ticker=args.ticker.upper(), alpha=args.alpha)
    set_files()

    results = run_dm_tests()
    print_summary(results)
