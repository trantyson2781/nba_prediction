# NBA Game Prediction

Predicts whether the **home team wins** an NBA game, using an XGBoost model over
team-form, rest/fatigue, strength-of-schedule, star availability, and **Elo rating**
features — with an Elo model as the honest benchmark.

> **This project was rebuilt from three copy-pasted notebooks into a tested `src/nba_pred/`
> package.** The rebuild fixed several silent bugs (train/serve skew, cross-season
> leakage, broken calibration) that had inflated the previously-reported accuracy. The
> numbers below are the honest, leakage-free re-baseline. See `docs/IMPROVEMENT_PLAN.md`
> for the full story.

## Performance (out-of-sample: 2024-25 regular season, 1,225 games)

| Model | Accuracy | Brier | Log-loss | AUC | High-conf acc |
|-------|:---:|:---:|:---:|:---:|:---:|
| Elo alone (benchmark) | 0.665 | 0.218 | 0.629 | 0.719 | 0.70 |
| XGBoost, 15 features | 0.655 | 0.218 | 0.626 | 0.711 | 0.76 |
| **XGBoost + Elo (shipped)** | **0.66** | **0.216** | **0.622** | **0.715** | **0.76** |

Home base rate is 0.544, so the model adds ~11 points over always-picking-home.
**A plain Elo rating model is a very strong baseline** — the engineered features add
only a small edge on probability quality (Brier / log-loss). High-confidence games
(model prob ≥ 0.65) are predicted at ~76% accuracy.

*Metrics are reported with Brier, log-loss, AUC, calibration error, and Wilson
confidence intervals — not accuracy alone — because this is a probability model.*

## Architecture

```
src/nba_pred/
  config.py            # single source of truth (seasons, thresholds, hyperparams, Elo, betting)
  data/                # cache.py (frozen parquet), fetch.py (nba_api -> cache)
  features/            # team_features.py (ONE shared pipeline), stars.py, matchup.py, schema.py
  model/               # split.py, train.py (TimeSeriesSplit), calibrate.py, elo.py, bundle.py
  evaluation/          # metrics.py (honest panel), plots.py
  betting/             # kelly.py (fractional-Kelly sizing)
  pipelines/           # dataset.py, train.py, backtest.py, inference.py
data/cache/            # frozen raw data (committed — reproducible offline)
artifacts/             # model_bundle.pkl (model + calibrator + feature order + provenance)
notebooks/             # thin callers of the package (originals kept in notebooks/legacy/)
tests/                 # pytest suite (feature correctness, no-leakage, parity, betting)
```

The design guarantee: **training and inference call the identical feature code path**,
so train/serve skew cannot silently return.

## Usage

```bash
pip install -e ".[dev]"          # install package + dev deps
# if that errors on an older pip (<23), use the already-installed setuptools:
#   pip install --user -e ".[dev]" --no-build-isolation
python -m pytest                 # run the test suite

nba-fetch                        # (re)build the raw data cache from nba_api (optional; cache is committed)
nba-train                        # train + calibrate -> artifacts/model_bundle.pkl
nba-backtest                     # evaluate on 2024-25 vs the Elo benchmark
```

Or from Python / the notebooks:

```python
from nba_pred.pipelines.train import run_training
from nba_pred.pipelines.backtest import run_backtest
from nba_pred.pipelines.inference import predict_season

bundle, report = run_training()          # notebooks/01
result = run_backtest("2024-25")         # notebooks/02
preds  = predict_season("2025-26")       # notebooks/03
```

## What was fixed in the rebuild

1. **Train/serve skew** — fatigue flags used different thresholds in training vs scoring.
2. **Cross-season leakage** — rolling/cumulative features never reset at season boundaries.
3. **Cumulative-record leakage** — records were computed on reverse-chronological data.
4. **Star feature** — availability was built from one season's data, not all three.
5. **Broken calibration** — isotonic was fit in-sample and never applied at inference.
6. **CV leakage** — stratified K-fold on time-series data → `TimeSeriesSplit`.
7. **Reproducibility** — added `random_state`, pinned dependencies, frozen data cache.

## Known limitations & next steps

- **Elo is hard to beat.** Further accuracy likely requires signal Elo can't capture —
  pre-game injury/availability reports, matchup/travel effects — i.e. new data sources.
- **Betting backtest is a placeholder.** Honest ROI/closing-line-value evaluation needs
  real historical odds data (not yet integrated); only the Kelly sizer is implemented.
- **Star = full-season scoring leader** (mild intra-season look-ahead); season-opener
  games have NaN form features handled natively by XGBoost. Both tracked for follow-up.
```
