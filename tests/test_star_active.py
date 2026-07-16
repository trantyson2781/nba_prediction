"""Test #4 — season-aware star availability (fix #3).

Legacy mode keys the star map from ONLY the last season's leaders and pools only
that season's attendance, so DIFF_STAR_ACTIVE is non-zero for very few games
(~0.07). Fixed mode keys the star by (SEASON, TEAM_ID) and pools attendance
across all seasons, so it is non-zero far more often and in every season.
"""
from __future__ import annotations

TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]


def _active_frac(X):
    return (X["DIFF_STAR_ACTIVE"] != 0).mean()


def test_fixed_star_active_much_higher_than_legacy(fixed_train, legacy_train):
    _, Xf, _, _ = fixed_train
    _, Xl, _, _ = legacy_train
    fixed_frac = _active_frac(Xf)
    legacy_frac = _active_frac(Xl)

    # Legacy reproduces the ~0.07 bug.
    assert legacy_frac < 0.15, f"legacy star-active frac unexpectedly high: {legacy_frac:.3f}"
    # Fixed is substantially higher.
    assert fixed_frac > 2 * legacy_frac, (
        f"fixed star-active frac ({fixed_frac:.3f}) should be much higher than "
        f"legacy ({legacy_frac:.3f})")
    assert fixed_frac > 0.15, f"fixed star-active frac too low: {fixed_frac:.3f}"


def test_fixed_star_active_nontrivial_each_season(fixed_train):
    _, _, _, full = fixed_train
    for s in TRAIN_SEASONS:
        sub = full[full["SEASON"] == s]
        frac = (sub["DIFF_STAR_ACTIVE"] != 0).mean()
        assert frac > 0.10, (
            f"season {s} has near-zero star-active fraction ({frac:.3f}); the "
            f"season-aware star map should make it non-trivial every season")
