"""Train the XGBoost model with leakage-free temporal CV (fix #5)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from nba_pred.config import Config


def tune_xgboost(X_train, y_train, cfg: Config):
    """GridSearchCV over PARAM_GRID using TimeSeriesSplit + log-loss scoring.

    TimeSeriesSplit guarantees every fold trains on games strictly before its
    validation games (the notebooks used StratifiedKFold on time data → leakage).
    Scoring on neg_log_loss optimizes probability quality, not just accuracy.
    """
    base = XGBClassifier(eval_metric="logloss", random_state=cfg.random_state)
    grid = GridSearchCV(
        base, cfg.param_grid,
        cv=TimeSeriesSplit(n_splits=cfg.cv_folds),
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def train_final(X_train, y_train, params: dict, cfg: Config) -> XGBClassifier:
    """Fit the final booster on the training slice with the tuned params."""
    model = XGBClassifier(eval_metric="logloss", random_state=cfg.random_state, **params)
    model.fit(X_train, y_train)
    return model
