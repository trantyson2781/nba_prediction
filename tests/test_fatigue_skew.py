"""Test #5 — fatigue rate is stable train vs serve (no train/serve skew).

The legacy shift(1) + divergent thresholds produced a wildly different home
FATIGUE_3_IN_4 rate between training (~0.94) and serving (~0.27). The fixed
definition puts the home rate in a plausible ~0.10-0.35 band in BOTH the
training seasons and the 2024-25 serving season.
"""
from __future__ import annotations

TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]
LOW, HIGH = 0.10, 0.35


def _home_rate(full, season):
    sub = full[full["SEASON"] == season]
    return sub["HOME_FATIGUE_3_IN_4"].mean()


def test_fixed_home_fatigue_rate_in_band_each_training_season(fixed_train):
    _, _, _, full = fixed_train
    for s in TRAIN_SEASONS:
        rate = _home_rate(full, s)
        assert LOW <= rate <= HIGH, (
            f"training season {s} home FATIGUE_3_IN_4 rate {rate:.3f} outside "
            f"[{LOW}, {HIGH}]")


def test_fixed_home_fatigue_rate_in_band_serving_season(fixed_test_season):
    _, _, _, full = fixed_test_season
    rate = _home_rate(full, "2024-25")
    assert LOW <= rate <= HIGH, (
        f"serving season 2024-25 home FATIGUE_3_IN_4 rate {rate:.3f} outside "
        f"[{LOW}, {HIGH}]")


def test_no_large_train_serve_fatigue_gap(fixed_train, fixed_test_season):
    _, _, _, full_tr = fixed_train
    _, _, _, full_te = fixed_test_season
    train_rate = full_tr["HOME_FATIGUE_3_IN_4"].mean()
    serve_rate = _home_rate(full_te, "2024-25")
    assert abs(train_rate - serve_rate) < 0.15, (
        f"train ({train_rate:.3f}) vs serve ({serve_rate:.3f}) fatigue gap too "
        f"large -> train/serve skew has returned")
