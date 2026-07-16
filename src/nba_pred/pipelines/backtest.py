"""Backtest a saved bundle on a season, plus the Elo-alone benchmark. Replaces nb02."""
from __future__ import annotations

from nba_pred.config import DEFAULT, Config
from nba_pred.evaluation import metrics as M
from nba_pred.model.bundle import ModelBundle
from nba_pred.pipelines.dataset import build_dataset, _elo_table


def run_backtest(season: str | None = None, cfg: Config = DEFAULT, bundle: ModelBundle | None = None) -> dict:
    season = season or cfg.test_season
    bundle = bundle or ModelBundle.load()
    X, y, meta = build_dataset([season], cfg)

    prob = bundle.predict_proba(X)
    result = {"season": season, "n": int(len(y)), "model": M.panel(y, prob),
              "ece": M.expected_calibration_error(y, prob)}

    # Elo-alone benchmark on the same games
    elo_df = _elo_table(cfg.all_seasons, cfg.elo_params)
    elo_prob = elo_df.reindex(meta["GAME_ID"].values)["ELO_WIN_PROB"].values
    result["elo_benchmark"] = M.panel(y, elo_prob)
    return result


def main():
    r = run_backtest()
    m, e = r["model"], r["elo_benchmark"]
    print(f"Backtest {r['season']} (n={r['n']})")
    print(f"  model: acc={m['accuracy']:.4f} brier={m['brier']:.4f} "
          f"logloss={m['log_loss']:.4f} auc={m['auc']:.4f}")
    print(f"  elo:   acc={e['accuracy']:.4f} brier={e['brier']:.4f} "
          f"logloss={e['log_loss']:.4f} auc={e['auc']:.4f}")


if __name__ == "__main__":
    main()
