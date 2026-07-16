"""Canonical feature ordering. The model bundle stores this list; inference
reindexes to it so column order can never silently drift (fixes nb03's
silent zero-fill)."""
from __future__ import annotations

# The 15 DIFF_* features, in a fixed canonical order.
FEATURES = [
    "DIFF_AST", "DIFF_REB", "DIFF_TOV", "DIFF_TS%", "DIFF_NET_RAT",
    "DIFF_SEASON_WIN_PCT", "DIFF_REST", "DIFF_STAR_ACTIVE", "DIFF_B2B",
    "DIFF_GAMES_4D", "DIFF_GAMES_5D", "DIFF_3_IN_4", "DIFF_4_IN_5",
    "DIFF_ADJ_NET_RAT", "DIFF_FATIGUE_ADJ_NET_RAT",
]

ELO_FEATURES = ["ELO_DIFF", "ELO_WIN_PROB"]

META_COLS = ["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_NAME", "AWAY_TEAM_NAME", "HOME_WIN"]


def model_features(cfg) -> list:
    """The feature columns the model is trained/served on (base + Elo if enabled)."""
    return FEATURES + (ELO_FEATURES if cfg.use_elo else [])
