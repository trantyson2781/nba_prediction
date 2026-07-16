"""One versioned artifact instead of three mismatched pkls.

Bundles the model, the calibrator, the exact feature order, and provenance, so
model / calibrator / feature-list can never again be loaded from inconsistent
files (the root of the notebooks' "three model objects" calibration bug).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts"


@dataclass
class ModelBundle:
    model: Any
    calibrator: Any
    features: list
    train_seasons: tuple
    best_params: dict
    metadata: dict

    def predict_proba(self, X):
        """Calibrated P(home win). Always reindexes X to the stored feature order."""
        Xo = X[self.features]
        raw = self.model.predict_proba(Xo)[:, 1]
        return self.calibrator.transform(raw)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "model_bundle.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path | None = None) -> "ModelBundle":
        path = Path(path) if path else ARTIFACT_DIR / "model_bundle.pkl"
        return joblib.load(path)
