"""Train a Random Forest that classifies mushrooms as edible or poisonous.

EDUCATIONAL PROTOTYPE ONLY. This is not a certified food-safety system.
A low predicted probability of toxicity never guarantees a mushroom is safe.

Usage:  python train_mushroom.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- Configuration (all tunables in one place) ------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Prefer data/mushrooms.csv; fall back to the copy already in the repo.
DATA_PATH = next(
    (
        p
        for p in [
            BASE_DIR / "data" / "mushrooms.csv",
            BASE_DIR / "data" / "mushroom" / "mushrooms.csv",
        ]
        if p.is_file()
    ),
    None,
)

MODELS_DIR = BASE_DIR / "models"
TARGET = "class"

TEST_SIZE = 0.20
RANDOM_SEED = 42
N_ESTIMATORS = 150

# Business rule: reject as poisonous unless P(poisonous) is below this value.
# This is a risk setting, not a safety certification.
POISONOUS_THRESHOLD = 0.05


# --- Data -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and validate the mushroom CSV."""
    if DATA_PATH is None:
        raise FileNotFoundError(
            "Dataset not found. Expected data/mushrooms.csv or "
            "data/mushroom/mushrooms.csv next to this script."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.relative_to(BASE_DIR)}: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found. Columns: {list(df.columns)}")

    labels = set(df[TARGET].unique())
    if labels != {"e", "p"}:
        raise ValueError(f"Expected labels 'e' and 'p', found: {sorted(labels)}")

    # '?' marks unknown values (in stalk-root); treat it as missing.
    if df.eq("?").any().any():
        counts = df.eq("?").sum()
        print(f"Missing-value marker '?': {counts[counts > 0].to_dict()}")
        df = df.replace("?", np.nan)

    print(f"Class balance: {df[TARGET].value_counts().to_dict()}")
    return df


# --- Model ------------------------------------------------------------------

def build_pipeline(feature_columns: list[str]) -> Pipeline:
    """One-hot encode the categorical features, then train a Random Forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            # handle_unknown="ignore" lets inference accept unseen codes.
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_columns)
        ]
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def predict_poisonous(pipeline: Pipeline, X: pd.DataFrame):
    """Return (probabilities, 'e'/'p' decisions) using the business rule."""
    # Find the poisonous column by checking the fitted class order.
    classes = list(pipeline.named_steps["model"].classes_)
    proba = pipeline.predict_proba(X)[:, classes.index(1)]
    decisions = np.where(proba < POISONOUS_THRESHOLD, "e", "p")
    return proba, decisions


# --- Evaluation -------------------------------------------------------------

def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate on the held-out test set using business error terms."""
    _, decisions = predict_poisonous(pipeline, X_test)
    y_pred = (decisions == "p").astype(int)  # poisonous = positive class

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        # Business false positive: poisonous predicted edible (dangerous).
        "dangerous_edible_approvals": int(fn),
        # Business false negative: edible predicted poisonous (unnecessary).
        "unnecessary_rejections": int(fp),
        "approved_edible": int(tn + fn),
        "rejected_poisonous": int(tp + fp),
        "total": len(y_test),
    }

    print(f"\nThreshold: reject when P(poisonous) >= {POISONOUS_THRESHOLD}")
    print(f"Accuracy:               {results['accuracy']:.6f}")
    print(f"Precision (poisonous):  {results['precision']:.6f}")
    print(f"Recall (poisonous):     {results['recall']:.6f}")
    print(f"F1-score (poisonous):   {results['f1']:.6f}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=[0, 1]),
            index=["actual edible", "actual poisonous"],
            columns=["pred edible", "pred poisonous"],
        ).to_string()
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["edible", "poisonous"], digits=4
        )
    )

    total = results["total"]
    print("Business error summary")
    print(
        f"  Dangerous edible approvals (poisonous -> edible): "
        f"{results['dangerous_edible_approvals']}"
    )
    print(
        f"  Unnecessary rejections (edible -> poisonous):     "
        f"{results['unnecessary_rejections']}"
    )
    print(
        f"  Approved as edible:   {results['approved_edible']}/{total} "
        f"({results['approved_edible'] / total * 100:.2f}%)"
    )
    print(
        f"  Rejected as poisonous: {results['rejected_poisonous']}/{total} "
        f"({results['rejected_poisonous'] / total * 100:.2f}%)"
    )
    return results


# --- Artifacts --------------------------------------------------------------

def save_artifacts(pipeline: Pipeline) -> None:
    """Save the fitted pipeline and a standalone copy of its encoder."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "random_forest_model.joblib")
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    joblib.dump(encoder, MODELS_DIR / "categorical_encoder.joblib")
    print(f"\nSaved model and encoder to {MODELS_DIR.relative_to(BASE_DIR)}/")


def check_artifacts(pipeline: Pipeline, X_test) -> None:
    """Reload the saved model and confirm it still predicts correctly."""
    reloaded = joblib.load(MODELS_DIR / "random_forest_model.joblib")
    _, original = predict_poisonous(pipeline, X_test.head(5))
    _, restored = predict_poisonous(reloaded, X_test.head(5))
    if not (original == restored).all():
        raise RuntimeError("Reloaded model does not match the trained model.")
    print("Artifact reload check passed.")


# --- Main -------------------------------------------------------------------

def main() -> None:
    print(
        "SAFETY NOTICE: educational prototype only. Predictions are not a "
        "certified food-safety assessment.\n"
    )

    df = load_data()
    feature_columns = [c for c in df.columns if c != TARGET]

    X = df[feature_columns]
    y = df[TARGET].map({"e": 0, "p": 1})

    # Stratified split keeps the edible/poisonous ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    pipeline = build_pipeline(feature_columns)
    # The encoder is fit only on training data, so the test set cannot leak in.
    pipeline.fit(X_train, y_train)

    results = evaluate_model(pipeline, X_test, y_test)
    save_artifacts(pipeline)
    check_artifacts(pipeline, X_test)

    print("\n=== Summary ===")
    print(f"Recall (poisonous): {results['recall']:.6f}")
    print(f"Dangerous edible approvals: {results['dangerous_edible_approvals']}")
    print(f"Unnecessary rejections:      {results['unnecessary_rejections']}")
    print(
        "Remember: good test scores on this dataset do not prove the model "
        "is safe for real-world use."
    )


if __name__ == "__main__":
    main()
