"""Test #6 — feature matrix schema, order, and NaN behaviour.

X columns must be EXACTLY schema.FEATURES, in order, 15 of them. NaNs are
allowed only in the 7 ROLL-derived DIFF columns and only on season-opener rows
(a team's first game of a season has no rolling history via shift(1)); the other
8 DIFF columns must be completely NaN-free. This pins/justifies the NaN surface.
"""
from __future__ import annotations

import pandas as pd

from nba_pred.features import schema

# DIFF columns derived from ROLL_* (hence NaN on a team's first game of season).
ROLL_DERIVED = [
    "DIFF_AST", "DIFF_REB", "DIFF_TOV", "DIFF_TS%", "DIFF_NET_RAT",
    "DIFF_ADJ_NET_RAT", "DIFF_FATIGUE_ADJ_NET_RAT",
]
NON_ROLL = [c for c in schema.FEATURES if c not in ROLL_DERIVED]


def test_x_columns_exactly_schema_features_in_order(fixed_train):
    _, X, _, _ = fixed_train
    assert list(X.columns) == schema.FEATURES, "X columns drifted from schema.FEATURES"
    assert X.shape[1] == 15, f"expected 15 features, got {X.shape[1]}"


def test_non_rolling_diffs_have_no_nan(fixed_train):
    _, X, _, _ = fixed_train
    n = X[NON_ROLL].isna().sum().sum()
    assert n == 0, f"non-rolling DIFF columns contain {n} unexpected NaNs"


def test_nans_confined_to_rolling_diffs_on_season_openers(fixed_train):
    """The only NaNs are ROLL-derived DIFF columns, and every NaN row involves a
    team playing its season opener (no prior rolling history)."""
    team, X, _, full = fixed_train

    # NaNs exist only in the ROLL-derived columns.
    cols_with_nan = [c for c in schema.FEATURES if X[c].isna().any()]
    assert set(cols_with_nan).issubset(set(ROLL_DERIVED)), (
        f"unexpected NaN columns: {set(cols_with_nan) - set(ROLL_DERIVED)}")

    # Season openers, by (SEASON, TEAM_ID) minimum GAME_DATE.
    t = team.copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])
    opener = t.groupby(["SEASON", "TEAM_ID"])["GAME_DATE"].transform("min")
    opener_gids = set(t.loc[t["GAME_DATE"] == opener, "GAME_ID"])

    nan_mask = X[ROLL_DERIVED].isna().any(axis=1).to_numpy()
    nan_gids = set(full.loc[nan_mask, "GAME_ID"])
    assert nan_gids.issubset(opener_gids), (
        "NaN rows found that are not season openers")

    # And it is a small minority of games.
    assert nan_mask.mean() < 0.05, (
        f"too many NaN rows ({nan_mask.mean():.3f}); expected only season openers")
