"""Combine per-team rows into home-vs-away matchups and emit the model matrix.

This module owns the ONE copy of the DIFF_* assembly and the fatigue-penalty
formula, so training and inference cannot diverge. ``build_matchup_features``
returns X reindexed to ``schema.FEATURES``.
"""
from __future__ import annotations

import pandas as pd

from nba_pred.config import Config
from nba_pred.features import schema
from nba_pred.features.stars import StarData


def _stats_cols(team_df: pd.DataFrame) -> list[str]:
    base = ["GAME_ID", "SEASON", "TEAM_NAME", "TEAM_ID", "WL", "SEASON_WIN_PCT",
            "DAYS_REST", "GAMES_PLAYED", "GAME_DATE", "TS%", "NET_RAT", "IS_B2B",
            "FATIGUE_3_IN_4", "FATIGUE_4_IN_5", "GAMES_IN_LAST_4_DAYS",
            "GAMES_IN_LAST_5_DAYS", "ADJ_NET_RAT"]
    return base + [c for c in team_df.columns if "ROLL_" in c]


def build_matchup_features(team_df: pd.DataFrame, star_data: StarData, cfg: Config):
    """Return (X, meta, full_df). X columns are exactly schema.FEATURES, in order."""
    cols = _stats_cols(team_df)
    home = team_df[team_df["MATCHUP"].str.contains("vs")][cols].copy()
    away = team_df[team_df["MATCHUP"].str.contains(" @ ")][cols].copy()
    home = home.add_prefix("HOME_").rename(
        columns={"HOME_GAME_ID": "GAME_ID", "HOME_GAME_DATE": "GAME_DATE"})
    away = away.add_prefix("AWAY_").rename(columns={"AWAY_GAME_ID": "GAME_ID"})

    m = pd.merge(home, away, on="GAME_ID")
    m["HOME_WIN"] = (m["HOME_WL"] == "W").astype(int)
    m["SEASON"] = m["HOME_SEASON"]

    # star identity (season-aware in fixed mode) + availability
    m["HOME_STAR_ID"] = [star_data.star_id(s, t) for s, t in zip(m["HOME_SEASON"], m["HOME_TEAM_ID"])]
    m["AWAY_STAR_ID"] = [star_data.star_id(s, t) for s, t in zip(m["AWAY_SEASON"], m["AWAY_TEAM_ID"])]

    for col in ["AST", "REB", "TOV", "TS%", "NET_RAT"]:
        m[f"DIFF_{col}"] = m[f"HOME_ROLL_{col}"] - m[f"AWAY_ROLL_{col}"]
    m["DIFF_SEASON_WIN_PCT"] = m["HOME_SEASON_WIN_PCT"] - m["AWAY_SEASON_WIN_PCT"]
    m["DIFF_REST"] = m["HOME_DAYS_REST"] - m["AWAY_DAYS_REST"]
    m["DIFF_STAR_ACTIVE"] = [
        star_data.is_active(g, h) - star_data.is_active(g, a)
        for g, h, a in zip(m["GAME_ID"], m["HOME_STAR_ID"], m["AWAY_STAR_ID"])]
    m["DIFF_B2B"] = m["HOME_IS_B2B"] - m["AWAY_IS_B2B"]
    m["DIFF_GAMES_4D"] = m["HOME_GAMES_IN_LAST_4_DAYS"] - m["AWAY_GAMES_IN_LAST_4_DAYS"]
    m["DIFF_GAMES_5D"] = m["HOME_GAMES_IN_LAST_5_DAYS"] - m["AWAY_GAMES_IN_LAST_5_DAYS"]
    m["DIFF_3_IN_4"] = m["HOME_FATIGUE_3_IN_4"] - m["AWAY_FATIGUE_3_IN_4"]
    m["DIFF_4_IN_5"] = m["HOME_FATIGUE_4_IN_5"] - m["AWAY_FATIGUE_4_IN_5"]
    m["DIFF_ADJ_NET_RAT"] = m["HOME_ADJ_NET_RAT"] - m["AWAY_ADJ_NET_RAT"]

    for side in ["HOME", "AWAY"]:
        m[f"{side}_FATIGUE_PENALTY"] = (
            1 - cfg.penalty_b2b * m[f"{side}_IS_B2B"]
            - cfg.penalty_3in4 * m[f"{side}_FATIGUE_3_IN_4"]
            - cfg.penalty_4in5 * m[f"{side}_FATIGUE_4_IN_5"]
            - cfg.penalty_games5d * m[f"{side}_GAMES_IN_LAST_5_DAYS"].clip(lower=2) / 3)
    m["DIFF_FATIGUE_ADJ_NET_RAT"] = (
        m["HOME_ADJ_NET_RAT"] * m["HOME_FATIGUE_PENALTY"]
        - m["AWAY_ADJ_NET_RAT"] * m["AWAY_FATIGUE_PENALTY"])

    m["GAME_DATE"] = pd.to_datetime(m["GAME_DATE"])
    m = m.sort_values("GAME_DATE").reset_index(drop=True)

    X = m[schema.FEATURES].copy()
    meta = m[[c for c in schema.META_COLS if c in m.columns]].copy()
    return X, meta, m
