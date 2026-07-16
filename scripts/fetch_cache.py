"""Fetch all raw nba_api data once and freeze it to data/cache/ as parquet.

This is the reproducibility anchor: with the committed cache, the whole
pipeline (feature build, training, tests) runs offline and deterministically,
without ever hitting the flaky live nba_api. Re-run with --refresh to repull.

Endpoints cached, per season:
  games         LeagueGameFinder      (one row per team per game)
  player_stats  LeagueDashPlayerStats (season totals, used to pick each team's star)
  player_logs   LeagueGameLog(P)      (per-player game logs, used for star availability)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguegamefinder,
    leaguegamelog,
)

# Seasons the project uses: 2021-24 train, 2024-25 backtest, 2025-26 inference.
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
SLEEP = 0.7  # be polite to the undocumented stats.nba.com endpoints


def _write(df: pd.DataFrame, subdir: str, key: str) -> Path:
    out_dir = CACHE_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}.parquet"
    df.to_parquet(path, index=False)
    return path


def fetch_games(season: str) -> pd.DataFrame:
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    ).get_data_frames()[0]
    df["SEASON"] = season  # stamp SEASON on every fetch (the leakage fix depends on it)
    return df


def fetch_player_stats(season: str) -> pd.DataFrame:
    df = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_player_logs(season: str) -> pd.DataFrame:
    df = leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="P"
    ).get_data_frames()[0]
    df["SEASON"] = season
    return df


ENDPOINTS = {
    "games": fetch_games,
    "player_stats": fetch_player_stats,
    "player_logs": fetch_player_logs,
}


def main(refresh: bool = False) -> None:
    for subdir, fetch in ENDPOINTS.items():
        for season in SEASONS:
            path = CACHE_DIR / subdir / f"{season}.parquet"
            if path.exists() and not refresh:
                print(f"skip  {subdir}/{season} (cached)")
                continue
            df = fetch(season)
            p = _write(df, subdir, season)
            print(f"wrote {subdir}/{season}  rows={len(df):>6}  -> {p.relative_to(CACHE_DIR.parent.parent)}")
            time.sleep(SLEEP)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="repull even if cached")
    main(**vars(ap.parse_args()))
