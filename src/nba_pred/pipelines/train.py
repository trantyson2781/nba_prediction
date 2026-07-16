"""Training pipeline: build dataset -> temporal split -> tune -> calibrate -> bundle.

Produces ONE versioned ModelBundle (model + calibrator + feature order + provenance)
saved to artifacts/. Replaces notebook 01.
"""
from __future__ import annotations

from nba_pred.config import DEFAULT, Config
from nba_pred.features import schema
from nba_pred.model import split as splitmod, train as trainmod, calibrate as cal
from nba_pred.model.bundle import ModelBundle
from nba_pred.pipelines.dataset import build_dataset


def run_training(cfg: Config = DEFAULT, save: bool = True) -> tuple[ModelBundle, dict]:
    X, y, meta = build_dataset(cfg.train_seasons, cfg)
    parts = splitmod.three_way_temporal(X, y, meta, cfg.calib_fraction, cfg.test_fraction)
    (Xtr, ytr) = parts["train"]
    (Xca, yca) = parts["calib"]

    grid = trainmod.tune_xgboost(Xtr, ytr, cfg)
    model = trainmod.train_final(Xtr, ytr, grid.best_params_, cfg)
    calibrator = cal.fit_calibrator(model, Xca, yca, method=cfg.calibration_method)

    bundle = ModelBundle(
        model=model, calibrator=calibrator,
        features=schema.model_features(cfg),
        train_seasons=tuple(cfg.train_seasons),
        best_params=grid.best_params_,
        metadata={"use_elo": cfg.use_elo, "calibration_method": cfg.calibration_method,
                  "random_state": cfg.random_state, "cv_best_score": float(grid.best_score_),
                  "split_bounds": parts["bounds"]},
    )
    report = {"best_params": grid.best_params_, "cv_best_score": float(grid.best_score_),
              "n_features": len(bundle.features), "split_bounds": parts["bounds"]}
    if save:
        report["artifact"] = str(bundle.save())
    return bundle, report


def main():
    bundle, report = run_training()
    print("Trained bundle:", report)


if __name__ == "__main__":
    main()
