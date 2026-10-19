from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
DATA = HERE / "data" / "mushroom" / "mushrooms.csv"
MODELS = HERE / "models"

TARGET = "class"  # e = edible, p = poisonous
ALGORITHMS = ["logistic regression", "decision tree", "random forest"]


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def candidates() -> dict:
    """The three models from the notebook. One-hot encoding sits inside each pipeline, so the
    encoder is fitted on training folds only and the saved model takes the raw letter codes."""
    return {
        "logistic regression": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "decision tree": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            DecisionTreeClassifier(random_state=42),
        ),
        "random forest": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
        ),
    }


def train(
    data_path: Path = DATA,
    models_dir: Path = MODELS,
    algorithm: str | None = None,
    verbose: bool = True,
) -> dict:
    say = print if verbose else (lambda *args, **kwargs: None)
    show = step if verbose else (lambda title: None)

    show("1. Load and check the data")
    df = pd.read_csv(data_path)
    features = df.columns.drop(TARGET).tolist()
    say(f"{len(df)} mushrooms, {len(features)} features (all one-letter category codes)")
    say(f"duplicate rows: {df.duplicated().sum()}, empty cells: {int(df.isna().sum().sum())}")
    say(f"'?' (unknown) in stalk-root: {df['stalk-root'].eq('?').sum()} -> kept as its own category")
    say(f"poisonous: {df[TARGET].eq('p').mean():.1%} -> the classes are balanced")

    show("2. Split: train on 80%, keep 20% aside for the final test")
    X, y = df[features], df[TARGET].eq("p").astype(int)  # 1 = poisonous
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    say(f"train: {len(X_train)} rows, test: {len(X_test)} rows")

    show("3. A baseline: always predict the majority class")
    majority = int(y_train.mode()[0])
    baseline = accuracy_score(y_test, np.full(len(y_test), majority))
    say(f"accuracy {baseline:.4f} ('{'poisonous' if majority else 'edible'}' for every mushroom)")

    show("4. Compare models with 5-fold cross-validation (ROC AUC)")
    models = candidates()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, candidate in models.items():
        auc = cross_val_score(candidate, X_train, y_train, cv=folds, scoring="roc_auc")
        scores[name] = float(auc.mean())
        say(f"{name:<20} ROC AUC {auc.mean():.6f} (+/- {auc.std():.6f})")
    if algorithm is None:
        best_name = max(scores, key=scores.get)
        say(f"-> best: {best_name}")
    else:
        best_name = algorithm
        say(f"-> using {best_name} (chosen with --algorithm)")
    model = models[best_name]

    show("5. Choose the decision threshold (on training data, never on the test set)")
    # Probabilities each training row gets from a model that did not see it.
    oof = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    thresholds = np.round(np.arange(0.1, 0.91, 0.01), 2)
    f1s = np.array([f1_score(y_train, oof >= t) for t in thresholds])
    tied = thresholds[np.isclose(f1s, f1s.max(), rtol=0, atol=1e-12)]
    threshold = float(tied[np.argmin(abs(tied - 0.5))])  # on ties, the one nearest 0.5
    say(f"threshold {threshold} maximises poisonous F1 ({f1s.max():.4f})")

    show("6. Train on all training data, evaluate once on the test set")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predicted = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted, zero_division=0), 4),
        "recall": round(recall_score(y_test, predicted, zero_division=0), 4),
        "f1": round(f1_score(y_test, predicted, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    for name, value in metrics.items():
        say(f"{name:<10} {value:.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    say(f"poisonous caught: {tp} of {tp + fn}  <- missed poisonous: {fn} (the dangerous mistake)")
    say(f"edible wrongly flagged poisonous: {fp} of {fp + tn}")

    show("7. Which features matter (permutation importance on the test set)")
    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1
    )
    ranked = sorted(zip(features, importance.importances_mean, strict=True), key=lambda x: -x[1])
    for feature, value in ranked[:5]:
        say(f"{feature:<26} {value:+.4f}")

    show("8. Save the pipeline and metadata for FastAPI")
    bundle = {
        "model": model,
        "algorithm": best_name,
        "features": features,
        "threshold": threshold,
        "metrics": metrics,
        "cv_scores": scores,
        "baseline_accuracy": float(baseline),
        "labels": {"e": "edible", "p": "poisonous"},
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "mushroom.joblib"
    joblib.dump(bundle, path)
    say(f"saved to {path}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the mushroom model.")
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        help="train this model instead of picking the best by cross-validated ROC AUC",
    )
    args = parser.parse_args()
    train(algorithm=args.algorithm)
