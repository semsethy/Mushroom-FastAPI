"""Notebook workflow in one script: python train_mushroom.py."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data/mushroom/mushrooms.csv'
MODEL_PATH = PROJECT_ROOT / 'models'
TARGET = 'class'


def step(title):
    print(f'\n=== {title} ===')


def candidates():
    # Encoding is fitted inside each CV fold, preventing preprocessing leakage.
    return {
        'logistic regression': make_pipeline(OneHotEncoder(handle_unknown='ignore'), LogisticRegression(max_iter=2000, random_state=42)),
        'decision tree': make_pipeline(OneHotEncoder(handle_unknown='ignore'), DecisionTreeClassifier(random_state=42)),
        'random forest': make_pipeline(OneHotEncoder(handle_unknown='ignore'), RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1)),
    }


def evaluate(y, probabilities, threshold):
    predicted = probabilities >= threshold
    return {name: float(value) for name, value in {
        'accuracy': accuracy_score(y, predicted),
        'precision': precision_score(y, predicted, zero_division=0),
        'recall': recall_score(y, predicted, zero_division=0),
        'f1': f1_score(y, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y, probabilities),
    }.items()}


def make_bundle(model, algorithm, features, threshold, metrics, cv_scores, baseline, data_path):
    return dict(model=model, algorithm=algorithm, features=list(features), threshold=float(threshold),
                metrics=metrics, cv_scores=cv_scores, baseline_accuracy=float(baseline),
                labels={'e': 'edible', 'p': 'poisonous'},
                trained_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                sklearn_version=sklearn.__version__, data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest())


def train(data_path=DATA_PATH, model_path=MODEL_PATH, verbose=True):
    say = print if verbose else lambda *args: None
    show = step if verbose else lambda *args: None
    show('1. Load and check the categorical data')
    df = pd.read_csv(data_path)
    if set(df[TARGET]) != {'e', 'p'} or df.isna().any().any():
        raise ValueError('Expected e/p labels and no empty cells; ? is retained as an explicit unknown category.')
    features = df.columns.drop(TARGET).tolist()
    say(f'{len(df)} rows, {len(features)} categorical features; {(df == "?").sum().sum()} unknown values')
    X, y = df[features], (df[TARGET] == 'p').astype(int)
    show('2. Stratified 80/20 split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    show('3. Majority-class baseline')
    majority = y_train.mode()[0]
    baseline = accuracy_score(y_test, np.full(len(y_test), majority))
    say(f'Baseline accuracy: {baseline:.4f}')
    show('4. Compare models with five-fold training-only ROC AUC')
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = candidates()
    scores = {}
    for name, candidate in models.items():
        values = cross_val_score(candidate, X_train, y_train, cv=folds, scoring='roc_auc')
        scores[name] = float(values.mean())
        say(f'{name}: {values.mean():.4f} (+/- {values.std():.4f})')
    best = max(scores, key=scores.get)
    model = models[best]
    show('5. Choose threshold using out-of-fold training predictions')
    oof = cross_val_predict(model, X_train, y_train, cv=folds, method='predict_proba')[:, 1]
    thresholds = np.round(np.arange(.1, .91, .01), 2)
    f1s = np.array([f1_score(y_train, oof >= t) for t in thresholds])
    # Prefer the threshold nearest 0.5 when several thresholds tie.
    tied = thresholds[np.isclose(f1s, f1s.max(), rtol=0, atol=1e-12)]
    threshold = float(tied[np.argmin(abs(tied - .5))])
    show('6. Fit and evaluate once on held-out data')
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    metrics = evaluate(y_test, probabilities, threshold)
    say(best, 'threshold:', threshold, metrics)
    say('Confusion matrix [edible, poisonous]:\n', confusion_matrix(y_test, probabilities >= threshold))
    show('7. Save the fitted pipeline and metadata')
    bundle = make_bundle(model, best, features, threshold, metrics, scores, baseline, data_path)
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path / 'mushroom.joblib')
    say(f'Saved {model_path / "mushroom.joblib"}')
    return bundle


if __name__ == '__main__':
    train()
