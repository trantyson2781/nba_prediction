"""Fetch all raw nba_api data and freeze it to data/cache/ as parquet.

Reproducibility anchor: with the committed cache the whole pipeline runs offline
and deterministically. Re-run `nba-fetch --refresh` to repull.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats, leaguegamefinder, leaguegamelog)

from nba_pred.config import DEFAULT
from nba_pred.data.cache import CACHE_DIR

SLEEP = 0.7  # be polite to the undocumented stats.nba.com endpoints


def _seasons() -> list[str]:
    return list(DEFAULT.all_seasons)


def fetch_games(season: str) -> pd.DataFrame:
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00",
        season_type_nullable="Regular Season").get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_player_stats(season: str) -> pd.DataFrame:
    df = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_player_logs(season: str) -> pd.DataFrame:
    df = leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="P").get_data_frames()[0]
    df["SEASON"] = season
    return df


ENDPOINTS = {"games": fetch_games, "player_stats": fetch_player_stats,
             "player_logs": fetch_player_logs}


def main(refresh: bool = False) -> None:
    for subdir, fetch in ENDPOINTS.items():
        for season in _seasons():
            path = CACHE_DIR / subdir / f"{season}.parquet"
            if path.exists() and not refresh:
                print(f"skip  {subdir}/{season} (cached)")
                continue
            df = fetch(season)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"wrote {subdir}/{season}  rows={len(df):>6}")
            time.sleep(SLEEP)


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="repull even if cached")
    main(**vars(ap.parse_args()))


if __name__ == "__main__":
    _cli()
