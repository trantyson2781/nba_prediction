"""Shared fixtures for the feature-pipeline test suite.

The expensive real-cache builds (3 training seasons, fixed + legacy) are built
ONCE per test session and reused. Synthetic tests build their own tiny frames.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from nba_pred.config import Config, DEFAULT
from nba_pred.features import team_features as tf
from nba_pred.features import matchup, stars

TRAIN_SEASONS = list(DEFAULT.train_seasons)          # 2021-22, 2022-23, 2023-24
TEST_SEASON = DEFAULT.test_season                    # 2024-25

# nba_api / pandas emit assorted deprecation & divide-by-zero warnings while the
# real pipeline runs; silence them so test output stays readable.
warnings.filterwarnings("ignore")


def _build(seasons, cfg: Config):
    """Return (team_features, X, meta, full) for `seasons` under `cfg`."""
    team = tf.load_team_features(seasons, cfg)
    star = stars.build_star_data(seasons, cfg)
    X, meta, full = matchup.build_matchup_features(team, star, cfg)
    return team, X, meta, full


@pytest.fixture(scope="session")
def fixed_train():
    """Fixed-mode (correctness-fixes-on) build across the 3 training seasons."""
    return _build(TRAIN_SEASONS, DEFAULT)


@pytest.fixture(scope="session")
def legacy_train():
    """Legacy-mode (bug-for-bug) build across the 3 training seasons."""
    return _build(TRAIN_SEASONS, Config(legacy=True))


@pytest.fixture(scope="session")
def fixed_test_season():
    """Fixed-mode build for the held-out serving season (2024-25)."""
    return _build([TEST_SEASON], DEFAULT)


def synthetic_team(team_id: int, dates, season: str = "2023-24") -> pd.DataFrame:
    """Minimal per-team-game frame with exactly the columns add_fatigue reads
    (TEAM_ID, SEASON, GAME_DATE, WL, MATCHUP)."""
    n = len(dates)
    wl = [("W", "L")[i % 2] for i in range(n)]
    return pd.DataFrame({
        "TEAM_ID": [team_id] * n,
        "SEASON": [season] * n,
        "GAME_DATE": pd.to_datetime(list(dates)),
        "WL": wl,
        "MATCHUP": ["AAA vs. BBB"] * n,
    })
