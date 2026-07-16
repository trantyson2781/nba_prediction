"""Assemble the model dataset: base matchup features + Elo, from the frozen cache.

`build_dataset` is the ONE entry every consumer (train / backtest / inference)
uses, so the feature set is identical everywhere. Elo is computed once over the
full chronological window (`cfg.all_seasons`) so ratings entering any target
season are warm and strictly pre-game.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from nba_pred.config import Config
from nba_pred.data import cache
from nba_pred.features import team_features as tf, stars, matchup, schema
from nba_pred.model import elo


@lru_cache(maxsize=4)
def _elo_table(all_seasons: tuple, params) -> pd.DataFrame:
    games = cache.load_many("games", list(all_seasons))
    return elo.compute_elo_features(games, params).set_index("GAME_ID")


def build_dataset(seasons, cfg: Config):
    """Return (X, y, meta) for `seasons`. X columns == schema.model_features(cfg)."""
    team = tf.load_team_features(list(seasons), cfg)
    star_data = stars.build_star_data(list(seasons), cfg)
    X, meta, full = matchup.build_matchup_features(team, star_data, cfg)
    y = full["HOME_WIN"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    if cfg.use_elo:
        elo_df = _elo_table(cfg.all_seasons, cfg.elo_params)
        e = elo_df.reindex(meta["GAME_ID"].values).reset_index(drop=True)
        for col in schema.ELO_FEATURES:
            X[col] = e[col].values

    return X[schema.model_features(cfg)], y, meta
