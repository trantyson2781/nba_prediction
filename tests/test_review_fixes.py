"""Regression tests for the three HIGH code-review findings (bugs #1, #2, #3)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_pred.config import DEFAULT
from nba_pred.data import cache
from nba_pred.features import team_features as tf


def _mini(wls):
    """A single-team schedule with the given WL sequence (last may be None = upcoming)."""
    dates = ["2025-11-01", "2025-11-02", "2025-11-04", "2025-11-05", "2025-11-06"][: len(wls)]
    return pd.DataFrame({
        "TEAM_ID": [1] * len(wls), "SEASON": ["2025-26"] * len(wls),
        "GAME_DATE": pd.to_datetime(dates), "WL": wls, "MATCHUP": ["A vs B"] * len(wls)})


def test_add_fatigue_single_team_does_not_crash():
    # Bug #2: the serve path builds features for one team / one game at a time.
    out = tf.add_fatigue(_mini(["W", "L", "W", "W", "W"]), DEFAULT)
    assert list(out["GAMES_IN_LAST_4_DAYS"]) == [1, 2, 3, 3, 3]


def test_fatigue_counts_schedule_not_label():
    # Bug #3: an upcoming game with unknown WL must still be counted, or the window
    # undercounts by 1 at serve time -> train/serve skew.
    known = tf.add_fatigue(_mini(["W", "L", "W", "W", "W"]), DEFAULT)
    serve = tf.add_fatigue(_mini(["W", "L", "W", "W", None]), DEFAULT)
    assert known["GAMES_IN_LAST_4_DAYS"].iloc[-1] == serve["GAMES_IN_LAST_4_DAYS"].iloc[-1]


def test_zero_fta_game_survives_fixed_build():
    # Bug #1: a 0-free-throw game has FT_PCT=NaN; the old blanket dropna() deleted the
    # row, and the matchup merge then silently dropped the whole game. Fixed mode must
    # keep it (both team rows present).
    games = cache.load("games", "2023-24")
    zero_fta = games[games["FTA"] == 0]
    if zero_fta.empty:
        pytest.skip("no 0-FTA game in this cache season")
    gid = zero_fta["GAME_ID"].iloc[0]
    feat = tf.build_team_features(cache.load("games", "2023-24"), DEFAULT)
    assert (feat["GAME_ID"] == gid).sum() == 2, "0-FTA game lost a side in fixed build"
