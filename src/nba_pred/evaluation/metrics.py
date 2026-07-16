"""Honest evaluation panel: accuracy alone is not enough for a probability model."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (for high-conf accuracy)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def panel(y_true, prob, threshold_high: float = 0.65) -> dict:
    """Full metric panel for calibrated P(home win) vs actual home wins."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        # labels=[0,1] keeps log_loss well-defined even if the eval slice is one class
        "log_loss": float(log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "base_rate_home_win": float(y_true.mean()),
    }
    # high-confidence subset: |prob-0.5| large, i.e. prob>=hi or prob<=1-hi
    hi = threshold_high
    mask = (prob >= hi) | (prob <= 1 - hi)
    n_hc = int(mask.sum())
    if n_hc:
        correct = int((pred[mask] == y_true[mask]).sum())
        lo, up = _wilson(correct, n_hc)
        out["high_conf"] = {
            "threshold": hi, "n": n_hc, "coverage": n_hc / len(y_true),
            "accuracy": correct / n_hc, "wilson_95": [lo, up],
        }
    else:
        out["high_conf"] = {"threshold": hi, "n": 0}
    return out


def calibration_table(y_true, prob, bins=None) -> pd.DataFrame:
    """Reliability table: predicted vs observed win rate per probability bin."""
    if bins is None:
        bins = np.linspace(0.0, 1.0, 11)
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(prob, float)})
    df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    tab = df.groupby("bin", observed=False).agg(
        n=("y", "count"), avg_pred=("p", "mean"), win_rate=("y", "mean"))
    tab["gap"] = tab["win_rate"] - tab["avg_pred"]
    return tab


def expected_calibration_error(y_true, prob, bins=None) -> float:
    tab = calibration_table(y_true, prob, bins).dropna(subset=["avg_pred"])
    if tab["n"].sum() == 0:
        return float("nan")
    return float((tab["n"] * tab["gap"].abs()).sum() / tab["n"].sum())
