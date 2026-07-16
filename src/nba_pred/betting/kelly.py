"""Fractional-Kelly stake sizing. Fixes from the notebook version:
  * KELLY_CAP corrected (config: 0.05, was 0.20 with a comment claiming 3%)
  * always feed a CALIBRATED probability (caller's responsibility)
Honest ROI/CLV backtesting against real closing odds is deferred (no odds data yet).
"""
from __future__ import annotations

from nba_pred.config import Config, DEFAULT


def suggest_bet_amount(model_prob: float, decimal_odds: float, cfg: Config = DEFAULT) -> float:
    """Fractional-Kelly stake given a (calibrated) model probability and decimal odds."""
    if not (0 < model_prob < 1):
        raise ValueError("model_prob must be in (0, 1).")
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be > 1.")
    if model_prob <= cfg.conf_threshold:
        return 0.0
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    f = ((b * p) - q) / b * cfg.kelly_multiplier
    f = max(0.0, min(f, cfg.kelly_cap))
    return cfg.bankroll * f


def edge(model_prob: float, decimal_odds: float) -> float:
    """Model probability minus the odds-implied probability."""
    return model_prob - 1.0 / decimal_odds
