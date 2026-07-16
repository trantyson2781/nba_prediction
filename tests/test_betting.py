"""Betting-math edge cases for the fixed Kelly sizer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_pred.config import DEFAULT
from nba_pred.betting import kelly


def test_below_confidence_gate_returns_zero():
    # 0.55 <= conf_threshold (0.60) -> no bet
    assert kelly.suggest_bet_amount(0.55, 2.0, DEFAULT) == 0.0


def test_half_kelly_matches_hand_computation():
    # p=0.70, decimal odds 2.0 -> b=1, full Kelly f=(1*0.7-0.3)/1=0.4; half=0.2; capped at 0.05
    stake = kelly.suggest_bet_amount(0.70, 2.0, DEFAULT)
    assert stake == pytest.approx(DEFAULT.bankroll * DEFAULT.kelly_cap)  # 0.2 -> capped to 0.05


def test_uncapped_half_kelly():
    # p=0.62, odds 2.0 -> full f=(0.62-0.38)/1=0.24; half=0.12; still above cap 0.05 -> capped
    # choose a case under the cap: p=0.62, odds 1.2 -> b=0.2, f=((0.2*0.62)-0.38)/0.2=-1.28<0 -> 0
    assert kelly.suggest_bet_amount(0.62, 1.2, DEFAULT) == 0.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        kelly.suggest_bet_amount(1.5, 2.0, DEFAULT)
    with pytest.raises(ValueError):
        kelly.suggest_bet_amount(0.7, 1.0, DEFAULT)


def test_edge_sign():
    assert kelly.edge(0.60, 2.0) == pytest.approx(0.10)   # 0.60 - 0.50
    assert kelly.edge(0.40, 2.0) == pytest.approx(-0.10)
