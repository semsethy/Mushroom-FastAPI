"""Train, evaluate, and save the mushroom classification model.

This script is the production-friendly version of Notebook/mushroom_model.ipynb.
It selects a model and threshold using only training data, evaluates the locked
test set once, and writes the complete bundle consumed by the FastAPI app.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "mushrooms.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mushroom.joblib"
TARGET = "class"
POSITIVE_CLASS = "p"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def build_pipeline(estimator: Any, *, scale: bool = False) -> Pipeline:
    """Combine categorical encoding, optional scaling, and a classifier."""
    steps: list[tuple[str, Any]] = [
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def candidate_models() -> dict[str, Pipeline]:
    """Return the five model pipelines compared in the notebook."""
    return {
        "logistic regression": build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000), scale=True
        ),
        "decision tree": build_pipeline(
            DecisionTreeClassifier(
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "random forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "extra trees": build_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "support vector machine": build_pipeline(
            CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced", C=1.0),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
            scale=True,
        ),
    }


def compare_models(
    candidates: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    folds: StratifiedKFold,
) -> pd.DataFrame:
    """Compare candidates using out-of-sample training-fold metrics."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows: list[dict[str, float | str]] = []
    for name, candidate in candidates.items():
        scores = cross_validate(
            candidate,
            x_train,
            y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=-1,
        )
        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values(["roc_auc", "recall", "f1"], ascending=False)
    )


def select_threshold(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    folds: StratifiedKFold,
) -> tuple[float, float]:
    """Choose the F1-maximizing threshold from out-of-fold probabilities."""
    probabilities = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=folds,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    f1_values = np.array(
        [f1_score(y_train, probabilities >= value) for value in thresholds]
    )
    best_index = int(np.argmax(f1_values))
    return float(np.round(thresholds[best_index], 2)), float(f1_values[best_index])


def evaluate(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Calculate the final locked-test metrics."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
    }


def train(data_path: Path, output_path: Path) -> dict[str, Any]:
    """Run model selection, final evaluation, and artifact serialization."""
    frame = pd.read_csv(data_path)
    if TARGET not in frame:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")

    features = [column for column in frame.columns if column != TARGET]
    x = frame.loc[:, features]
    y = frame[TARGET].eq(POSITIVE_CLASS).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    candidates = candidate_models()
    comparison = compare_models(candidates, x_train, y_train, folds)
    best_name = str(comparison.index[0])
    model = candidates[best_name]
    threshold, cross_validated_f1 = select_threshold(
        model, x_train, y_train, folds
    )

    model.fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test, threshold)
    bundle: dict[str, Any] = {
        "model": model,
        "algorithm": best_name,
        "features": features,
        "target": TARGET,
        "class_labels": {0: "edible", 1: "poisonous"},
        "threshold": threshold,
        "metrics": metrics,
        "cross_validated_f1": round(cross_validated_f1, 4),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    print("\nCross-validation results:")
    print(comparison.round(4).to_string())
    print(f"\nSelected model: {best_name}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Test metrics: {metrics}")
    print(f"Saved bundle: {output_path.resolve()}")
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.data, arguments.output)
