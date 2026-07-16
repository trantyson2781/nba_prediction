"""Temporal splits. No shuffling — the holdout is always the most recent games."""
from __future__ import annotations

import pandas as pd


def three_way_temporal(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                       calib_fraction: float, test_fraction: float):
    """Split time-ordered rows into train / calibration / test by GAME_DATE order.

    Expects X/y/meta already sorted ascending by GAME_DATE (matchup output is).
    Returns a dict of (X, y) tuples plus the boundary indices, so callers can
    assert the three index sets are disjoint (calibration must never overlap
    train or test — that is the whole point of honest calibration, fix #4).
    """
    if "GAME_DATE" in meta.columns and not meta["GAME_DATE"].is_monotonic_increasing:
        raise ValueError(
            "three_way_temporal requires rows pre-sorted ascending by GAME_DATE; "
            "an unsorted input silently produces a leaky (non-temporal) split.")

    n = len(X)
    n_test = int(round(n * test_fraction))
    n_calib = int(round(n * calib_fraction))
    n_train = n - n_calib - n_test
    if min(n_train, n_calib, n_test) <= 0:
        raise ValueError(f"degenerate split: n={n} train={n_train} calib={n_calib} test={n_test}")

    sl_train = slice(0, n_train)
    sl_calib = slice(n_train, n_train + n_calib)
    sl_test = slice(n_train + n_calib, n)
    return {
        "train": (X.iloc[sl_train], y.iloc[sl_train]),
        "calib": (X.iloc[sl_calib], y.iloc[sl_calib]),
        "test": (X.iloc[sl_test], y.iloc[sl_test]),
        "meta": {"train": meta.iloc[sl_train], "calib": meta.iloc[sl_calib], "test": meta.iloc[sl_test]},
        "bounds": {"n_train": n_train, "n_calib": n_calib, "n_test": n_test},
    }
