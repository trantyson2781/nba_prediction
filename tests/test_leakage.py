"""Test #2 — no-leakage / lag invariance of the per-team features.

Every rolling/cumulative feature is lagged with shift(1), so a game's features
must depend ONLY on chronologically-prior games. Strong form of the property:
perturbing a *later* game must NOT change the features of an *earlier* game.
We exercise a single real season for determinism.

Findings (see summary):
  * The stat-based rolling features (ROLL_AST/REB/TOV/TS%/NET_RAT) ARE causal:
    they are computed after a sort by GAME_DATE, so perturbing a later game's
    box score leaves an earlier game's rolls untouched.  -> passing test.
  * The cumulative win-record path (CUM_WINS -> SEASON_WIN_PCT ->
    ROLL_SEASON_WIN_PCT) LEAKS: add_form_and_rest runs cumcount()/cumsum().shift(1)
    BEFORE sorting, and the games cache is stored reverse-chronologically, so an
    early game's record is built from FUTURE games. Marked xfail(strict).
"""
from __future__ import annotations

import pandas as pd
import pytest

from nba_pred.config import DEFAULT as DEFAULT_CFG
from nba_pred.features import team_features as tf
from nba_pred.data import cache

SEASON = "2023-24"

# Box-score columns bumped to perturb a later game's stat-derived features.
BUMP_COLS = ["PTS", "AST", "REB", "TOV", "FGA", "FGM", "FG3M", "FTA", "PLUS_MINUS"]
STAT_ROLL_COLS = ["ROLL_AST", "ROLL_REB", "ROLL_TOV", "ROLL_TS%", "ROLL_NET_RAT"]


def _bump_stats(games: pd.DataFrame, game_id) -> pd.DataFrame:
    games = games.copy()
    m = games["GAME_ID"] == game_id
    for c in BUMP_COLS:
        games.loc[m, c] = games.loc[m, c] + 20
    return games


def _flip_wl(games: pd.DataFrame, game_id) -> pd.DataFrame:
    games = games.copy()
    m = games["GAME_ID"] == game_id
    games.loc[m, "WL"] = games.loc[m, "WL"].map({"W": "L", "L": "W"})
    return games


@pytest.fixture(scope="module")
def leak_setup():
    """One real season, a fixed team, and stable early/late game ids by date."""
    games = cache.load("games", SEASON)
    tid = games["TEAM_ID"].value_counts().idxmax()   # team with the most games
    sub = games[games["TEAM_ID"] == tid].copy()
    sub["GAME_DATE"] = pd.to_datetime(sub["GAME_DATE"])
    sub = sub.sort_values("GAME_DATE").reset_index(drop=True)
    early_gid = sub["GAME_ID"].iloc[5]
    late_gid = sub["GAME_ID"].iloc[-10]
    return games, tid, early_gid, late_gid


def _val(frame, tid, game_id, col):
    row = frame[(frame["TEAM_ID"] == tid) & (frame["GAME_ID"] == game_id)]
    assert len(row) == 1, f"expected exactly one row for {game_id}"
    return row[col].iloc[0]


def test_stat_roll_features_do_not_leak_future(leak_setup):
    """Perturbing a LATE game's box score must not move an EARLY game's rolls."""
    games, tid, early_gid, late_gid = leak_setup
    base = tf.build_team_features(games, DEFAULT_CFG)
    bumped = tf.build_team_features(_bump_stats(games, late_gid), DEFAULT_CFG)

    for col in STAT_ROLL_COLS:
        b = _val(base, tid, early_gid, col)
        f = _val(bumped, tid, early_gid, col)
        assert b == pytest.approx(f, nan_ok=True), (
            f"{col} for an early game changed when a LATER game's box score was "
            f"perturbed ({b} -> {f}); rolling stats must depend only on prior games")


def test_perturbing_a_game_does_not_change_its_own_roll(leak_setup):
    """shift(1) means a game's own box score is excluded from its own rolls."""
    games, tid, early_gid, late_gid = leak_setup
    base = tf.build_team_features(games, DEFAULT_CFG)
    bumped = tf.build_team_features(_bump_stats(games, early_gid), DEFAULT_CFG)
    for col in STAT_ROLL_COLS:
        b = _val(base, tid, early_gid, col)
        f = _val(bumped, tid, early_gid, col)
        assert b == pytest.approx(f, nan_ok=True), (
            f"{col} for a game changed when that same game's own box score was "
            f"perturbed; the shift(1) lag must exclude the current game")


def test_cumulative_record_does_not_leak_future(leak_setup):
    # Regression guard for bug #8: fixed mode sorts by group_keys+GAME_DATE before
    # the cumulative computations, so an early game's record cannot see future games.
    games, tid, early_gid, late_gid = leak_setup
    base = tf.build_team_features(games, DEFAULT_CFG)
    flipped = tf.build_team_features(_flip_wl(games, late_gid), DEFAULT_CFG)
    for col in ["CUM_WINS", "SEASON_WIN_PCT", "ROLL_SEASON_WIN_PCT"]:
        b = _val(base, tid, early_gid, col)
        f = _val(flipped, tid, early_gid, col)
        assert b == pytest.approx(f, nan_ok=True), (
            f"{col} for an early game changed when a LATER game was flipped "
            f"({b} -> {f}) -> temporal leakage")
