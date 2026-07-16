"""Star-player identity and availability.

Each team's "star" is its top scorer (season PTS leader). ``DIFF_STAR_ACTIVE``
is +1/0/-1 for whether the home vs away star appeared in the game log.

Fixed mode keys the star by (SEASON, TEAM_ID) and pools availability across all
requested seasons. Legacy mode reproduces the notebook bug: the star map is built
from only the LAST season in the loop (fix #3).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nba_pred.config import Config
from nba_pred.data import cache


@dataclass
class StarData:
    star_map: dict          # legacy: {TEAM_ID: PLAYER_ID}; fixed: {(SEASON, TEAM_ID): PLAYER_ID}
    active_pairs: set       # {(GAME_ID, PLAYER_ID)}
    legacy: bool

    def star_id(self, season, team_id):
        return self.star_map.get(team_id if self.legacy else (season, team_id))

    def is_active(self, game_id, player_id) -> int:
        return 1 if (game_id, player_id) in self.active_pairs else 0


def build_star_data(seasons, cfg: Config) -> StarData:
    per_season_stars, per_season_att = [], []
    for s in seasons:
        ps = cache.load("player_stats", s)
        stars = ps.sort_values("PTS", ascending=False).drop_duplicates("TEAM_ID")
        sl = stars[["TEAM_ID", "PLAYER_ID", "PLAYER_NAME"]].copy()
        sl["SEASON"] = s
        per_season_stars.append(sl)

        pl = cache.load("player_logs", s)
        att = pl[pl["PLAYER_ID"].isin(sl["PLAYER_ID"])][["GAME_ID", "PLAYER_ID"]].copy()
        per_season_att.append(att)

    if cfg.legacy:
        # BUG reproduction: only the last season's star list / attendance survive.
        sl = per_season_stars[-1]
        att = per_season_att[-1]
        star_map = dict(zip(sl["TEAM_ID"], sl["PLAYER_ID"]))
        active_pairs = set(zip(att["GAME_ID"], att["PLAYER_ID"]))
    else:
        sl = pd.concat(per_season_stars, ignore_index=True)
        att = pd.concat(per_season_att, ignore_index=True)
        star_map = dict(zip(zip(sl["SEASON"], sl["TEAM_ID"]), sl["PLAYER_ID"]))
        active_pairs = set(zip(att["GAME_ID"], att["PLAYER_ID"]))

    return StarData(star_map=star_map, active_pairs=active_pairs, legacy=cfg.legacy)
