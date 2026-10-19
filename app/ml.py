"""Model loading and inference helpers for the mushroom API."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mushroom.joblib"
REQUIRED_BUNDLE_KEYS = {
    "model",
    "algorithm",
    "features",
    "threshold",
    "metrics",
    "trained_at",
    "sklearn_version",
}


def configured_model_path() -> Path:
    """Return the configured model path, resolving it at call time for tests."""
    return Path(os.getenv("MUSHROOM_MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()


def load_bundle(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a trusted joblib model bundle."""
    model_path = Path(path).resolve() if path else configured_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {model_path}. Run `python mushroom_train.py`."
        )

    # joblib can execute code while loading. Only load artifacts created by this project.
    bundle = joblib.load(model_path)
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")

    trained_version = bundle.get("sklearn_version")
    if trained_version != sklearn.__version__:
        warnings.warn(
            f"The model was trained with scikit-learn {trained_version}, but the API "
            f"uses {sklearn.__version__}. Retrain the model if inference fails.",
            stacklevel=2,
        )
    return bundle


def predict_mushrooms(
    bundle: dict[str, Any], mushrooms: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Predict poison probability and threshold-based class for one or more rows."""
    if not mushrooms:
        raise ValueError("At least one mushroom is required.")

    features = list(bundle["features"])
    frame = pd.DataFrame(mushrooms)
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    frame = frame.loc[:, features]

    model = bundle["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain poisonous class 1.")

    positive_index = classes.index(1)
    probabilities = model.predict_proba(frame)[:, positive_index]
    threshold = float(bundle["threshold"])

    return [
        {
            "prediction": "poisonous" if probability >= threshold else "edible",
            "is_poisonous": bool(probability >= threshold),
            "poisonous_probability": round(float(probability), 6),
            "threshold": threshold,
        }
        for probability in probabilities
    ]
