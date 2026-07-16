"""Per-team-game feature engineering — the SINGLE source of truth.

Every consumer (training, backtest, inference) calls ``build_team_features`` so
the fatigue definition, grouping keys, and lag semantics can never drift between
train and serve. Behavior is switched entirely by ``config`` (see config.Config).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pred.config import Config
from nba_pred.data import cache

# Columns the feature pipeline actually consumes. Fixed mode drops a row only if
# one of THESE is missing (fix #1): the notebooks' blanket dropna() also deleted
# rows for NaN in irrelevant columns — e.g. FT_PCT is NaN on a 0-free-throw game,
# which silently removed the whole game from the matchup merge.
_REQUIRED_COLS = ["WL", "GAME_DATE", "MATCHUP", "TEAM_ID",
                  "AST", "REB", "TOV", "TS%", "NET_RAT"]

# Static abbreviation -> team id map (used to resolve the opponent from MATCHUP).
from nba_api.stats.static import teams as _static_teams
TEAM_ABBR_TO_ID = {t["abbreviation"]: t["id"] for t in _static_teams.get_teams()}


def add_advanced_metrics(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["POSS"] = 0.96 * (g["FGA"] + g["TOV"] + 0.4 * g["FTA"] - g["OREB"])
    g["eFG%"] = (g["FGM"] + 0.5 * g["FG3M"]) / g["FGA"]
    g["TS%"] = g["PTS"] / (2 * g["FGA"] + 0.44 * g["FTA"])
    g["NET_RAT"] = 100 * (g["PTS"] / g["POSS"] - (g["PTS"] - g["PLUS_MINUS"]) / g["POSS"])
    # A zero denominator yields inf, which dropna() does NOT catch; normalize to NaN
    # so a degenerate row is handled by the required-column drop, not silently kept.
    g[["eFG%", "TS%", "NET_RAT"]] = g[["eFG%", "TS%", "NET_RAT"]].replace([np.inf, -np.inf], np.nan)
    return g


def add_form_and_rest(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cumulative record, rest, and rolling EWMA form. All lagged with shift(1).

    Grouping is ``cfg.group_keys``: [TEAM_ID] in legacy mode (cross-season leak),
    [SEASON, TEAM_ID] in fixed mode (state resets each season — fix #2).
    """
    g = g.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    keys = cfg.group_keys
    if not cfg.legacy:
        # Fix #8: the cache is stored reverse-chronologically, so cumsum/cumcount
        # must run on date-sorted rows or the record leaks from FUTURE games.
        # Legacy mode intentionally skips this to reproduce the notebooks' bug.
        g = g.sort_values(keys + ["GAME_DATE"]).reset_index(drop=True)
    grp = g.groupby(keys)
    g["CUM_WINS"] = grp["WL"].transform(
        lambda x: (x == "W").astype(int).cumsum().shift(1).fillna(0))
    g["GAMES_PLAYED"] = grp.cumcount().astype(int)
    g["SEASON_WIN_PCT"] = (g["CUM_WINS"] / g["GAMES_PLAYED"]).fillna(0)
    g["DAYS_REST"] = grp["GAME_DATE"].diff().dt.days.clip(upper=cfg.days_rest_clip).fillna(cfg.days_rest_clip)
    g["IS_B2B"] = (g["DAYS_REST"] <= 1).astype(int).fillna(0)

    # Legacy reproduces the notebooks' blanket dropna() (needed for parity); fixed
    # mode only drops rows missing a column the pipeline actually uses (fix #1).
    if cfg.legacy:
        g = g.dropna().sort_values(keys + ["GAME_DATE"])
    else:
        g = g.dropna(subset=_REQUIRED_COLS).sort_values(keys + ["GAME_DATE"])
    for col in cfg.roll_cols:
        g["ROLL_" + col] = g.groupby(keys)[col].transform(
            lambda x: x.ewm(span=cfg.roll_span).mean().shift(1))

    g["OPP_ABBR"] = g["MATCHUP"].apply(lambda x: x.split(" ")[-1])
    g["OPP_ID"] = g["OPP_ABBR"].map(TEAM_ABBR_TO_ID)
    win_pct_map = g.set_index(["TEAM_ID", "GAME_DATE"])["SEASON_WIN_PCT"].to_dict()
    g["OPP_WIN_PCT_AT_GAME"] = g.apply(
        lambda x: win_pct_map.get((x["OPP_ID"], x["GAME_DATE"]), 0.5), axis=1)
    g["ROLL_SOS"] = g.groupby(keys)["OPP_WIN_PCT_AT_GAME"].transform(
        lambda x: x.rolling(cfg.sos_window).mean().shift(1).fillna(0.5))
    g["ADJ_NET_RAT"] = g["ROLL_NET_RAT"] * g["ROLL_SOS"]
    return g


def add_fatigue(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Schedule-density fatigue features.

    Fixed mode (correct): count games in the trailing 4/5-day window INCLUDING the
    current game (no shift) and threshold 3-in-4 / 4-in-5 — uses only the schedule,
    which is known before tip-off, so it is not leakage.
    Legacy mode: reproduce the notebooks' shift(1) window + divergent thresholds.
    """
    g = g.copy()
    keys = cfg.group_keys
    g = g.sort_values(keys + ["GAME_DATE"]).reset_index(drop=True)

    # Fix #3: count the SCHEDULE, not WL. rolling().count() only counts non-null
    # values; at serve time the upcoming game's WL is unknown (NaN) so it would be
    # undercounted by 1, re-introducing train/serve skew. A constant, always-present
    # column counts every scheduled game identically at train and serve time.
    g["_SCHED"] = 1.0
    shift = cfg.legacy  # legacy anchored the window at the PREVIOUS game via shift(1)

    def rolling_count(win):
        # Fix #2: iterate groups explicitly and align by the original index, instead
        # of relying on groupby.apply()'s reshape (which returns a wide DataFrame for
        # a single group / identical schedules and crashes the serve path).
        out = pd.Series(index=g.index, dtype=float)
        for idx in g.groupby(keys, sort=False).groups.values():
            d = g.loc[idx]
            s = pd.Series(d["_SCHED"].to_numpy(), index=pd.DatetimeIndex(d["GAME_DATE"]))
            cnt = s.rolling(win).count().to_numpy()
            if shift:
                cnt = pd.Series(cnt).shift(1).fillna(0).to_numpy()
            out.loc[d.index] = cnt
        return out

    g["GAMES_IN_LAST_4_DAYS"] = rolling_count("4D")
    g["GAMES_IN_LAST_5_DAYS"] = rolling_count("5D")
    if cfg.legacy:
        thr4, thr5 = cfg.legacy_fatigue_3in4_threshold, cfg.legacy_fatigue_4in5_threshold
    else:
        thr4, thr5 = cfg.fatigue_3in4_threshold, cfg.fatigue_4in5_threshold
    g["FATIGUE_3_IN_4"] = (g["GAMES_IN_LAST_4_DAYS"] >= thr4).astype(int)
    g["FATIGUE_4_IN_5"] = (g["GAMES_IN_LAST_5_DAYS"] >= thr5).astype(int)
    return g.drop(columns=["_SCHED"])


def build_team_features(games_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full per-team-game feature frame. One row per team per game."""
    g = add_advanced_metrics(games_df)
    g = add_form_and_rest(g, cfg)
    g = add_fatigue(g, cfg)
    return g


def load_team_features(seasons, cfg: Config) -> pd.DataFrame:
    """Convenience: load games for `seasons` from cache and build features."""
    games = cache.load_many("games", seasons)
    return build_team_features(games, cfg)
