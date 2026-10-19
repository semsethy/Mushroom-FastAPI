
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

MODELS = Path(__file__).resolve().parent.parent / "models"
MUSHROOM_PATH = MODELS / "mushroom.joblib"


class ModelNotTrained(FileNotFoundError):
    """The .joblib file does not exist yet: run the training script first."""


def load(path: Path = MUSHROOM_PATH) -> dict:
    """Read a model bundle. Only load files you trust: joblib can run code."""
    if not path.exists():
        raise ModelNotTrained(f"{path} not found. Run: python train_mushroom.py")
    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"model trained with scikit-learn {bundle['sklearn_version']}, "
            f"running {sklearn.__version__}: retrain if predictions look wrong",
            stacklevel=2,
        )
    return bundle


def risk_level(probability: float, threshold: float) -> str:
    """For people reading the result. Every line comes from the trained threshold:
    below it = edible = low; above it = poisonous, split in half: medium, then high."""
    if probability < threshold:  # edible
        return "low"
    high_cut = round((threshold + 1) / 2, 4)  # middle point between threshold and 1.0
    if probability >= high_cut:  # poisonous, model very sure
        return "high"
    return "medium"  # poisonous, model less sure


def to_csv_names(mushroom: dict[str, str]) -> dict[str, str]:
    """Python/JSON-friendly names (cap_shape) -> the CSV names the model was trained on."""
    return {key.replace("_", "-"): value for key, value in mushroom.items()}


def predict_mushroom(bundle: dict, mushrooms: list[dict[str, str]]) -> list[dict]:
    """A list of mushrooms (letter codes) -> a list of answers, one model call for all."""
    # Columns in exactly the training order; a missing field becomes NaN -> one-hot ignores it,
    # so the API schema must make every field required.
    rows = [to_csv_names(m) for m in mushrooms]
    df = pd.DataFrame(rows, columns=bundle["features"])
    probabilities = bundle["model"].predict_proba(df)[:, 1]
    threshold = bundle["threshold"]
    return [
        {
            "poisonous_probability": round(float(p), 4),
            "label": bundle["labels"]["p" if p >= threshold else "e"],
            "risk": risk_level(float(p), threshold),
            "threshold": threshold,
        }
        for p in probabilities
    ]


def model_info(bundle: dict) -> dict:
    """Everything about the model except the model object (for GET /models)."""
    return {
        key: value for key, value in bundle.items() if key not in ("model", "codes")
    }