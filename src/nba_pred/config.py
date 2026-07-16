"""Single source of truth for every constant in the pipeline.

The ``legacy`` flag is the mechanism that lets us prove a *behavior-preserving*
refactor: with ``Config(legacy=True)`` the package reproduces the original
notebooks bug-for-bug (verified against the golden snapshot); flipping to
``Config()`` (legacy=False) turns on the correctness fixes, one concern at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Config:
    # --- seasons -------------------------------------------------------------
    train_seasons: tuple[str, ...] = ("2021-22", "2022-23", "2023-24")
    test_season: str = "2024-25"
    infer_season: str = "2025-26"

    # --- rolling / form ------------------------------------------------------
    roll_cols: tuple[str, ...] = ("AST", "REB", "TOV", "TS%", "NET_RAT", "SEASON_WIN_PCT")
    roll_span: int = 10
    sos_window: int = 10
    days_rest_clip: int = 4

    # --- fatigue -------------------------------------------------------------
    # Correct basketball definition: count games in the trailing window INCLUDING
    # the current game (no shift), threshold 3-in-4 / 4-in-5. The legacy notebooks
    # used shift(1) + divergent thresholds (>=2/>=3 in nb01, >=3/>=4 in nb02/03).
    fatigue_3in4_threshold: int = 3
    fatigue_4in5_threshold: int = 4

    # fatigue-penalty coefficients (kept identical to the notebooks)
    penalty_b2b: float = 0.12
    penalty_3in4: float = 0.08
    penalty_4in5: float = 0.05
    penalty_games5d: float = 0.05

    # --- modeling ------------------------------------------------------------
    param_grid: dict = field(default_factory=lambda: {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
    })
    cv_folds: int = 5
    test_fraction: float = 0.2
    calib_fraction: float = 0.15   # held-out slice for calibration (fix #4)
    # "sigmoid" (Platt) is preferred for small calibration slices; isotonic
    # overfits ~hundreds of games and hurts log-loss. See nba-honest-baseline.
    calibration_method: str = "sigmoid"
    random_state: int = 42

    # --- elo (Phase 2: the ratings benchmark, also fed to XGBoost as features) -
    use_elo: bool = True           # append ELO_DIFF + ELO_WIN_PROB to the model features
    elo_k: float = 20.0
    elo_home_adv: float = 100.0
    elo_base: float = 1500.0
    elo_season_regress: float = 0.25
    elo_use_mov: bool = True

    # --- betting -------------------------------------------------------------
    min_edge: float = 0.03
    kelly_cap: float = 0.05        # fixed: was 0.20 with a comment claiming 3%
    kelly_multiplier: float = 0.5  # half-Kelly
    bankroll: float = 500.0
    conf_threshold: float = 0.60

    # --- refactor / correctness switch --------------------------------------
    legacy: bool = False
    # In legacy mode ONLY, these reproduce the exact (buggy) notebook behavior:
    #   * group rolling/cumulative features by TEAM_ID only (cross-season leakage)
    #   * shift(1) the fatigue window count and use the given thresholds
    #   * build the star map from a single leftover season
    legacy_fatigue_3in4_threshold: int = 2   # nb01 value
    legacy_fatigue_4in5_threshold: int = 3   # nb01 value

    @property
    def all_seasons(self) -> tuple[str, ...]:
        """Every season in chronological order — the continuous window Elo runs over
        so ratings entering the test/inference season are warm (not reset to base)."""
        return tuple(self.train_seasons) + (self.test_season, self.infer_season)

    @property
    def elo_params(self):
        from nba_pred.model.elo import EloParams
        return EloParams(k=self.elo_k, home_adv=self.elo_home_adv, base=self.elo_base,
                         season_regress=self.elo_season_regress, use_mov=self.elo_use_mov)

    @property
    def group_keys(self) -> list[str]:
        """Grouping for all per-team rolling/cumulative features.

        The leakage fix (#2) is entirely expressed here: fixed mode resets state
        at each season boundary; legacy mode leaks across seasons like the notebooks.
        """
        return ["TEAM_ID"] if self.legacy else ["SEASON", "TEAM_ID"]

    def legacy_variant(self, fatigue_3in4: int, fatigue_4in5: int) -> "Config":
        """A legacy config with specific fatigue thresholds (nb01 vs nb02 differ)."""
        return replace(self, legacy=True,
                       legacy_fatigue_3in4_threshold=fatigue_3in4,
                       legacy_fatigue_4in5_threshold=fatigue_4in5)


DEFAULT = Config()
