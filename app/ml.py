"""Load the trained models and make predictions with them."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

# Where train_churn.py and train_sentiment.py saved the models (override for tests).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))


def load(name: str) -> dict | None:
    """The saved bundle (model + metadata), or None if that model was not trained yet."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)  # only load model files you trust: joblib can run code
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed: retrain it if predictions look wrong.",
            stacklevel=2,
        )
    return bundle


def risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def predict_churn(bundle: dict, customers: list[dict]) -> list[dict]:
    frame = pd.DataFrame(customers, columns=bundle["features"])
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    threshold = bundle["threshold"]
    return [
        {
            "churn_probability": round(float(p), 4),
            "will_churn": bool(p >= threshold),
            "risk": risk_level(float(p), threshold),
            "threshold": threshold,
        }
        for p in probabilities
    ]


def predict_sentiment(bundle: dict, texts: list[str]) -> list[dict]:
    model = bundle["model"]
    labels = list(model.classes_)
    out = []
    for row in model.predict_proba(texts):
        probabilities = {label: round(float(p), 4) for label, p in zip(labels, row, strict=True)}
        label = max(probabilities, key=probabilities.get)
        out.append(
            {"label": label, "confidence": probabilities[label], "probabilities": probabilities}
        )
    return out


def predict_mushroom(bundle: dict, mushrooms: list[dict]) -> list[dict]:
    frame = pd.DataFrame(mushrooms, columns=bundle["features"])
    model = bundle["model"]
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(frame)[:, positive_index]
    return [
        {
            "label": "poisonous" if p >= 0.5 else "edible",
            "poisonous_probability": round(float(p), 4),
            "confidence": round(float(max(p, 1 - p)), 4),
        }
        for p in probabilities
    ]
