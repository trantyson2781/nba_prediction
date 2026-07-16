"""Test #1 — add_fatigue correctness on hand-computed synthetic schedules.

Fixed mode counts games in the trailing 4/5-day window INCLUDING the current
game (pandas offset window is left-open, right-closed: (t-window, t]), then
thresholds 3-in-4 (>=3) and 4-in-5 (>=4). Legacy mode shift(1)s the count and
uses divergent thresholds (2/3 by nb01 default).

Two teams with DISTINCT schedules are used on purpose: with a single group (or
groups that share an identical GAME_DATE index) pandas' groupby.apply reshapes
the per-group Series into a wide DataFrame and add_fatigue raises. Distinct
schedules keep the result a stacked Series, matching real multi-team usage.
"""
from __future__ import annotations

import pandas as pd

from nba_pred.config import Config, DEFAULT
from nba_pred.features import team_features as tf
from conftest import synthetic_team

# Team 1 schedule: days 1,2,4,5,6,10  (gaps 1-2, 2, 1-1, 4)
DATES_1 = ["2023-01-01", "2023-01-02", "2023-01-04",
           "2023-01-05", "2023-01-06", "2023-01-10"]
# Team 2 schedule: 1,2,3,4 (four straight) then 8
DATES_2 = ["2023-01-01", "2023-01-02", "2023-01-03",
           "2023-01-04", "2023-01-08"]


def _make_frame():
    return pd.concat(
        [synthetic_team(1, DATES_1), synthetic_team(2, DATES_2)],
        ignore_index=True,
    )


def _series(out, team_id, col):
    sub = out[out["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    return list(sub[col].astype(int))


def test_add_fatigue_fixed_counts_include_current_game():
    out = tf.add_fatigue(_make_frame(), DEFAULT)

    # Hand-computed trailing-window counts INCLUDING the current game.
    assert _series(out, 1, "GAMES_IN_LAST_4_DAYS") == [1, 2, 3, 3, 3, 1]
    assert _series(out, 1, "GAMES_IN_LAST_5_DAYS") == [1, 2, 3, 4, 4, 2]
    assert _series(out, 2, "GAMES_IN_LAST_4_DAYS") == [1, 2, 3, 4, 1]
    assert _series(out, 2, "GAMES_IN_LAST_5_DAYS") == [1, 2, 3, 4, 2]


def test_add_fatigue_fixed_thresholds():
    out = tf.add_fatigue(_make_frame(), DEFAULT)

    # FATIGUE_3_IN_4 = (4-day count >= 3); FATIGUE_4_IN_5 = (5-day count >= 4)
    assert _series(out, 1, "FATIGUE_3_IN_4") == [0, 0, 1, 1, 1, 0]
    assert _series(out, 1, "FATIGUE_4_IN_5") == [0, 0, 0, 1, 1, 0]
    assert _series(out, 2, "FATIGUE_3_IN_4") == [0, 0, 1, 1, 0]
    assert _series(out, 2, "FATIGUE_4_IN_5") == [0, 0, 0, 1, 0]


def test_add_fatigue_legacy_shift_and_thresholds():
    """Legacy mode shift(1)s the window count and uses the nb01 thresholds 2/3."""
    out = tf.add_fatigue(_make_frame(), Config(legacy=True))

    # Legacy counts = fixed counts shifted by one game (fillna 0).
    assert _series(out, 1, "GAMES_IN_LAST_4_DAYS") == [0, 1, 2, 3, 3, 3]
    assert _series(out, 1, "GAMES_IN_LAST_5_DAYS") == [0, 1, 2, 3, 4, 4]
    # FATIGUE_3_IN_4 threshold 2, FATIGUE_4_IN_5 threshold 3.
    assert _series(out, 1, "FATIGUE_3_IN_4") == [0, 0, 1, 1, 1, 1]
    assert _series(out, 1, "FATIGUE_4_IN_5") == [0, 0, 0, 1, 1, 1]
