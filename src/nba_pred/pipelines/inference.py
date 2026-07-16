"""Inference pipeline: calibrated predictions for a season's games. Replaces nb03's
per-season scoring. (Future-matchup 'as-of' prediction is a Phase-2 follow-up; it
must reuse build_dataset's feature path — never a hand-copied one — to stay skew-free.)
"""
from __future__ import annotations

import pandas as pd

from nba_pred.config import DEFAULT, Config
from nba_pred.model.bundle import ModelBundle
from nba_pred.pipelines.dataset import build_dataset


def predict_season(season: str | None = None, cfg: Config = DEFAULT,
                   bundle: ModelBundle | None = None) -> pd.DataFrame:
    """Return one row per game with calibrated P(home win), prediction, and outcome."""
    season = season or cfg.infer_season
    bundle = bundle or ModelBundle.load()
    X, y, meta = build_dataset([season], cfg)
    out = meta.copy()
    out["HOME_WIN_PROB"] = bundle.predict_proba(X)
    out["PRED_HOME_WIN"] = (out["HOME_WIN_PROB"] >= 0.5).astype(int)
    out["CORRECT"] = (out["PRED_HOME_WIN"] == out["HOME_WIN"]).astype(int)
    return out


def main():
    df = predict_season()
    print(f"{DEFAULT.infer_season}: {len(df)} games, "
          f"accuracy={df['CORRECT'].mean():.4f}")
    print(df[["GAME_DATE", "HOME_TEAM_NAME", "AWAY_TEAM_NAME", "HOME_WIN_PROB", "HOME_WIN"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
