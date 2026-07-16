"""Probability calibration on a HELD-OUT slice (fix #4).

The notebooks fit isotonic on in-sample training probabilities (degenerate) and
then never applied it at inference. Here the calibrator is fit strictly on the
calibration slice — disjoint from both train and test — and the bundle carries
it so every inference path applies it before thresholding.

Default method is "sigmoid" (Platt): isotonic overfits small (~hundreds of games)
calibration slices and worsens log-loss, so it is opt-in.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class _PlattCalibrator:
    """Platt scaling: logistic fit of y ~ raw_prob. Exposes .transform() like isotonic."""

    def __init__(self):
        self._lr = LogisticRegression()

    def fit(self, raw, y):
        self._lr.fit(np.asarray(raw).reshape(-1, 1), np.asarray(y))
        return self

    def transform(self, raw):
        return self._lr.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


def make_calibrator(method: str):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    if method == "sigmoid":
        return _PlattCalibrator()
    raise ValueError(f"unknown calibration_method: {method!r}")


class _IdentityCalibrator:
    """No-op fallback (e.g. single-class calibration slice) — passes raw probs through."""

    def fit(self, raw, y):
        return self

    def transform(self, raw):
        return np.asarray(raw, dtype=float)


def fit_calibrator(model, X_calib, y_calib, method: str = "sigmoid"):
    """Fit a calibrator mapping raw model probs -> calibrated probs, on held-out data.

    If the calibration slice has only one class, both Platt (raises) and isotonic
    (silently collapses to a constant) misbehave, so fall back to identity.
    """
    y = np.asarray(y_calib)
    if len(np.unique(y)) < 2:
        return _IdentityCalibrator()
    raw = model.predict_proba(X_calib)[:, 1]
    return make_calibrator(method).fit(raw, y)


def apply_calibrator(model, calibrator, X):
    """Calibrated P(home win) for X."""
    raw = model.predict_proba(X)[:, 1]
    return calibrator.transform(raw)
