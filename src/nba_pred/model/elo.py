"""Standard NBA Elo rating model — the honest benchmark for the ML model.

Elo is a strong, leakage-free baseline: a game's rating features use ONLY prior
games (the rating is read before the game, updated after). Many ML NBA models
fail to beat a well-tuned Elo on log-loss, so it is the yardstick Phase 2 must clear.

Design (FiveThirtyEight-style):
  * K-factor, home-court advantage in Elo points
  * margin-of-victory multiplier (dampened by the favorite's rating edge)
  * between-season regression of every team toward the mean
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 100.0
    base: float = 1500.0
    season_regress: float = 0.25   # new = (1-r)*old + r*base at each season start
    use_mov: bool = True


def _expected_home(pre_home: float, pre_away: float, p: EloParams) -> float:
    return 1.0 / (1.0 + 10 ** (-((pre_home + p.home_adv) - pre_away) / 400.0))


def _mov_multiplier(margin: float, winner_elo_edge: float, p: EloParams) -> float:
    """538 MOV multiplier: bigger blowouts move ratings more, dampened when the
    favorite (positive edge) wins (autocorrelation correction)."""
    if not p.use_mov:
        return 1.0
    return math.log(abs(margin) + 1.0) * (2.2 / (winner_elo_edge * 0.001 + 2.2))


def _game_level(games_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two team-rows per game into one home/away row, sorted by date."""
    g = games_df.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    home = g[g["MATCHUP"].str.contains("vs")][["GAME_ID", "SEASON", "GAME_DATE", "TEAM_ID", "WL", "PLUS_MINUS"]]
    away = g[g["MATCHUP"].str.contains(" @ ")][["GAME_ID", "TEAM_ID"]]
    home = home.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "PLUS_MINUS": "HOME_MARGIN"})
    away = away.rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})
    m = pd.merge(home, away, on="GAME_ID")
    m["HOME_WIN"] = (m["WL"] == "W").astype(int)
    # order by date, then GAME_ID for a deterministic tie-break within a day
    return m.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def compute_elo_features(games_df: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Run Elo chronologically across all games; return per-GAME_ID pre-game features.

    Columns: GAME_ID, ELO_PRE_HOME, ELO_PRE_AWAY, ELO_DIFF (home edge incl. home_adv),
    ELO_WIN_PROB (P(home win)). All values are strictly pre-game (no leakage).
    """
    m = _game_level(games_df)
    ratings: dict = {}
    prev_season = None
    rows = []
    for r in m.itertuples(index=False):
        if r.SEASON != prev_season and prev_season is not None:
            # between-season regression toward the mean
            for tid in list(ratings):
                ratings[tid] = (1 - params.season_regress) * ratings[tid] + params.season_regress * params.base
        prev_season = r.SEASON

        pre_home = ratings.get(r.HOME_TEAM_ID, params.base)
        pre_away = ratings.get(r.AWAY_TEAM_ID, params.base)
        exp_home = _expected_home(pre_home, pre_away, params)

        rows.append((r.GAME_ID, pre_home, pre_away,
                     (pre_home + params.home_adv) - pre_away, exp_home))

        # --- update after the game ---
        # winner's pre-game Elo edge (incl. home adv) for the MOV dampening
        if r.HOME_WIN:
            winner_edge = (pre_home + params.home_adv) - pre_away
        else:
            winner_edge = pre_away - (pre_home + params.home_adv)
        mult = _mov_multiplier(r.HOME_MARGIN, winner_edge, params)
        delta = params.k * mult * (r.HOME_WIN - exp_home)
        ratings[r.HOME_TEAM_ID] = pre_home + delta
        ratings[r.AWAY_TEAM_ID] = pre_away - delta

    return pd.DataFrame(rows, columns=[
        "GAME_ID", "ELO_PRE_HOME", "ELO_PRE_AWAY", "ELO_DIFF", "ELO_WIN_PROB"])
