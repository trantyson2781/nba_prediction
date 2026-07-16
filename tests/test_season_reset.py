"""Test #3 — season-boundary reset (fix #2).

Fixed mode groups rolling/cumulative features by [SEASON, TEAM_ID], so state
resets each season and GAMES_PLAYED can never exceed one season (~82). Legacy
mode groups by TEAM_ID only and leaks across seasons, so GAMES_PLAYED blows past
82 (this documents the bug the fix removes).

Note: the "first game of each season starts at 0" property is also asserted, but
it currently FAILS for the *chronologically* first game because CUM_WINS /
GAMES_PLAYED are computed on the unsorted (reverse-chronological) cache — the
same leakage bug covered in test_leakage.py. It is xfail(strict) here.
"""
from __future__ import annotations

import pandas as pd
import pytest

MAX_GAMES_PER_SEASON = 82


def test_fixed_games_played_never_exceeds_a_season(fixed_train):
    team, _X, _meta, _full = fixed_train
    mx = int(team["GAMES_PLAYED"].max())
    assert mx <= MAX_GAMES_PER_SEASON, (
        f"fixed mode GAMES_PLAYED reached {mx}; per-season grouping should cap "
        f"it at {MAX_GAMES_PER_SEASON}")


def test_fixed_state_resets_each_season(fixed_train):
    """Every (SEASON, TEAM_ID) group's own GAMES_PLAYED range stays within a
    single season (0-based, so max < 82)."""
    team, _X, _meta, _full = fixed_train
    per_group_max = team.groupby(["SEASON", "TEAM_ID"])["GAMES_PLAYED"].max()
    assert (per_group_max < MAX_GAMES_PER_SEASON).all(), (
        "some (SEASON, TEAM_ID) group exceeded one season of games -> state did "
        "not reset at the season boundary")


def test_legacy_leaks_across_seasons(legacy_train):
    """Legacy TEAM_ID-only grouping accumulates across all 3 seasons."""
    team, _X, _meta, _full = legacy_train
    mx = int(team["GAMES_PLAYED"].max())
    assert mx > MAX_GAMES_PER_SEASON, (
        f"legacy mode GAMES_PLAYED only reached {mx}; expected cross-season "
        f"leakage above {MAX_GAMES_PER_SEASON}")


def test_fixed_first_game_of_season_starts_at_zero(fixed_train):
    # Regression guard for bug #8: after the sort-before-cumulative fix, each
    # season's chronologically-first game must start at GAMES_PLAYED==0.
    team, _X, _meta, _full = fixed_train
    t = team.copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])
    first = t.sort_values("GAME_DATE").groupby(["SEASON", "TEAM_ID"]).first()
    assert (first["GAMES_PLAYED"] == 0).all(), "first game GAMES_PLAYED != 0"
    assert (first["CUM_WINS"] == 0).all(), "first game CUM_WINS != 0"
