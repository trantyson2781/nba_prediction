"""Honest betting backtest against prediction-market prices (Polymarket/Kalshi).

Platform-agnostic: it consumes a NORMALIZED frame where each row is one game with
the model's P(home win), the market's implied P(home win) at entry, the actual
outcome, and (optionally) the closing implied P(home win) for closing-line value.

Bet rule: back the side where the model's probability exceeds the market's by at
least `min_edge`; size with fractional Kelly on the calibrated model probability
and the fee-adjusted market odds. Reports ROI, win rate, and CLV with bootstrap CIs.

Prediction-market prices ARE probabilities (0-1), so decimal odds = 1 / price and
there is no bookmaker vig to strip — only the platform fee and the bid/ask spread,
passed in via `fee_fn`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from nba_pred.config import DEFAULT, Config
from nba_pred.betting import kelly

# Required columns in the normalized input frame.
REQUIRED = ["game_date", "home_win", "model_prob", "market_home_prob"]


def _no_fee(price: float, contracts: float) -> float:
    return 0.0


@dataclass
class Bet:
    game_date: object
    side: str            # "HOME" or "AWAY"
    model_prob: float    # model prob for the side we backed
    market_prob: float   # market implied prob for that side (entry)
    edge: float
    stake: float
    won: bool
    pnl: float
    clv: float           # closing-line value: close_prob_for_side - entry_prob_for_side (>0 = beat close)


def _decide(row, cfg) -> tuple[str, float, float] | None:
    """Return (side, model_prob_side, market_prob_side) if there is an edge, else None."""
    mp_home = row["model_prob"]
    mk_home = row["market_home_prob"]
    home_edge = mp_home - mk_home
    away_edge = (1 - mp_home) - (1 - mk_home)  # = mk_home - mp_home
    if home_edge >= cfg.min_edge and home_edge >= away_edge:
        return "HOME", mp_home, mk_home
    if away_edge >= cfg.min_edge:
        return "AWAY", 1 - mp_home, 1 - mk_home
    return None


def run_backtest(df: pd.DataFrame, cfg: Config = DEFAULT,
                 fee_fn: Callable[[float, float], float] = _no_fee) -> dict:
    """Simulate flat-bankroll fractional-Kelly betting over the normalized frame."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"normalized odds frame missing columns: {missing}")
    df = df.sort_values("game_date").reset_index(drop=True)
    has_close = "close_home_prob" in df.columns

    bets: list[Bet] = []
    for _, row in df.iterrows():
        decision = _decide(row, cfg)
        if decision is None:
            continue
        side, mp_side, mk_side = decision
        if not (0 < mk_side < 1):
            continue
        decimal_odds = 1.0 / mk_side
        stake = kelly.suggest_bet_amount(mp_side, decimal_odds, cfg)
        if stake <= 0:
            continue

        side_won = bool(row["home_win"]) if side == "HOME" else not bool(row["home_win"])
        contracts = stake / mk_side  # 1 contract pays $1 if it hits; cost = price
        fee = fee_fn(mk_side, contracts)
        pnl = (contracts - stake - fee) if side_won else (-stake - fee)

        clv = np.nan
        if has_close and pd.notna(row.get("close_home_prob")):
            close_side = row["close_home_prob"] if side == "HOME" else 1 - row["close_home_prob"]
            clv = close_side - mk_side  # positive = we got a better price than the close

        bets.append(Bet(row["game_date"], side, mp_side, mk_side,
                        mp_side - mk_side, stake, side_won, pnl, clv))

    return _summarize(bets, cfg)


def _bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 42):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = stat(values[idx], axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _summarize(bets: list[Bet], cfg: Config) -> dict:
    if not bets:
        return {"n_bets": 0, "note": "no bets cleared the edge threshold"}
    pnl = np.array([b.pnl for b in bets])
    stake = np.array([b.stake for b in bets])
    won = np.array([b.won for b in bets])
    clv = np.array([b.clv for b in bets], dtype=float)
    total_staked = float(stake.sum())
    # ROI per bet (pnl / stake) for a stake-independent bootstrap
    roi_per_bet = pnl / stake
    log = pd.DataFrame([b.__dict__ for b in bets])
    return {
        "n_bets": len(bets),
        "win_rate": float(won.mean()),
        "total_staked": total_staked,
        "total_pnl": float(pnl.sum()),
        "roi": float(pnl.sum() / total_staked) if total_staked else float("nan"),
        "roi_per_bet_ci95": _bootstrap_ci(roi_per_bet),
        "avg_edge": float(np.mean([b.edge for b in bets])),
        "avg_clv": float(np.nanmean(clv)) if np.isfinite(clv).any() else float("nan"),
        "clv_positive_rate": float(np.nanmean(clv > 0)) if np.isfinite(clv).any() else float("nan"),
        "bet_log": log,
    }
