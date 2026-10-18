# Mushroom Safety Classification

Educational machine-learning prototype that trains a Random Forest to estimate
the probability that a mushroom sample is poisonous, then applies a
conservative business decision rule for a conceptual Food Safety and Quality
Assurance application.

> **Safety warning:** This is an educational prototype, **not** a certified
> mushroom identification system. A low predicted probability of toxicity must
> never be presented as a guarantee that a mushroom is safe to eat. Real-world
> food-safety deployment requires independent validation and expert mycologist
> oversight.

## Dataset

- Source: [Kaggle - Mushroom Classification (UCI)](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- 8,124 instances, 22 categorical physical features, no numerical measurements
- Target column `class`: `e` = Edible (4,208), `p` = Poisonous (3,916)
- `stalk-root` uses `?` for unknown values (2,480 rows); pandas does not treat
  `?` as null, so the training script converts it to a missing value that the
  one-hot encoder keeps as its own category
- The repository currently stores the CSV at `data/mushroom/mushrooms.csv`.
  `train_mushroom.py` first looks for `data/mushrooms.csv` and falls back to
  the existing path, so no duplicate download is needed.

## Folder structure

```text
Mushroom-FastAPI/
│
├── app/                        # Reserved for a future API (untouched in this task)
├── data/
│   └── mushroom/
│       └── mushrooms.csv       # Original Kaggle dataset
├── models/
│   ├── random_forest_model.joblib    # Fitted pipeline (preprocessor + model)
│   └── categorical_encoder.joblib   # Fitted OneHotEncoder (standalone)
├── notebooks/
│   ├── exploration.ipynb       # Data exploration and plots (this task)
│   └── mushroom_model.ipynb    # Earlier exploratory notebook (preserved)
├── train_mushroom.py           # Standalone training entry point
├── requirements.txt
└── README.md
```

## Environment setup

Requires Python 3.10+.

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

pip install -r requirements.txt
```

## How to run the training script

```bash
python train_mushroom.py
```

The script resolves every path relative to its own location, so it also works
from another working directory. It:

1. Loads and validates `data/mushrooms.csv` (clear errors for missing files,
   unexpected columns, or invalid labels)
2. Converts the `?` marker to a missing value
3. Performs a stratified 80/20 train/test split (`random_state=42`)
4. Fits a `Pipeline` of `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")`
   + `RandomForestClassifier(n_estimators=150, random_state=42)` — the encoder
   is fit only on training data, preventing leakage
5. Evaluates on the held-out test set using the 0.05 business decision rule
6. Saves `models/random_forest_model.joblib` and
   `models/categorical_encoder.joblib`, then reloads both to verify them

## The 0.05 decision threshold

The model outputs `predict_proba()`; the poisonous column is located by
checking the fitted model's `classes_` (never assumed by position). The
decision rule is configured in **one place** in `train_mushroom.py`
(`POISONOUS_DECISION_THRESHOLD = 0.05`):

```python
if poisonous_probability < 0.05:
    prediction = "e"   # approved as edible
else:
    prediction = "p"   # rejected as poisonous
```

This is a deliberately conservative **business risk setting**: a sample must
look almost certainly non-toxic before it is approved. A probability below
0.05 is a model estimate, **not** proof of safety. Raising the threshold
rejects more edible mushrooms; lowering it admits more dangerous approvals.

## Business error definitions

The business terminology is intentionally opposite to standard binary
classification when Poisonous is the positive class:

| Business term | What happened | Standard classification term |
| --- | --- | --- |
| **Business false positive** (dangerous edible approval) | Poisonous predicted as **Edible** | False negative of the poisonous class |
| **Business false negative** (unnecessary rejection) | Edible predicted as **Poisonous** | False positive of the poisonous class |

Standard metrics in the script output (accuracy, precision, recall, F1,
confusion matrix) follow the standard convention with `poisonous` as the
positive class. The primary business objective is maximizing poisonous-class
recall and driving dangerous edible approvals to zero - not maximizing
accuracy.

## Actual evaluation results

Produced by `python train_mushroom.py` on the held-out test set
(1,625 samples: 842 edible, 783 poisonous; never used for fitting):

| Metric (poisonous = positive class) | Value |
| --- | --- |
| Accuracy | 1.000000 |
| Precision (poisonous) | 1.000000 |
| Recall (poisonous) | 1.000000 |
| F1-score (poisonous) | 1.000000 |

Confusion matrix (rows = actual, columns = predicted):

| | pred edible | pred poisonous |
| --- | --- | --- |
| **actual edible** | 842 | 0 |
| **actual poisonous** | 0 | 783 |

Business summary at the 0.05 threshold:

- Dangerous edible approvals (poisonous predicted edible): **0**
  (0.00% of actual poisonous) - safety target **achieved on this test set**
- Unnecessary rejections (edible predicted poisonous): **0** (0.00% of
  actual edible)
- Approved as edible: 842/1,625 (51.82% of test samples)
- Rejected as poisonous: 783/1,625 (48.18% of test samples)

These perfect scores are real outputs of the script on this dataset, but they
are a property of the data: the UCI mushroom dataset is near-deterministically
separable on features such as `odor`. They do **not** measure reliability on
real wild samples.

## Loading the saved artifacts

Only load joblib files you trust (joblib uses pickle internally).

```python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

models_dir = Path(__file__).resolve().parent / "models"  # adjust as needed
pipeline = joblib.load(models_dir / "random_forest_model.joblib")
encoder = joblib.load(models_dir / "categorical_encoder.joblib")  # standalone copy

# One row per sample, same 22 feature names and order as the training data.
# Use None/NaN for unknown stalk-root values.
sample = pd.DataFrame([{
    "cap-shape": "x", "cap-surface": "s", "cap-color": "n", "bruises": "t",
    "odor": "p", "gill-attachment": "f", "gill-spacing": "c", "gill-size": "n",
    "gill-color": "k", "stalk-shape": "e", "stalk-root": None,
    "stalk-surface-above-ring": "s", "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w", "stalk-color-below-ring": "w",
    "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p",
    "spore-print-color": "k", "population": "s", "habitat": "u",
}])

# Locate the poisonous column via the model's actual class order.
classes = list(pipeline.named_steps["model"].classes_)
poisonous_idx = classes.index(1)

probability = pipeline.predict_proba(sample)[0, poisonous_idx]
prediction = "e" if probability < 0.05 else "p"
print(prediction, probability)
```

## Exploration notebook

`notebooks/exploration.ipynb` covers dataset shape, missing values, class
balance, feature distributions, and poisonous-rate plots. It does not
duplicate the production training workflow; `train_mushroom.py` is the single
training entry point.

## Known limitations and safety warnings

- **Educational prototype only.** Never present model output as proof that a
  mushroom is safe to eat.
- Trained on a curated laboratory dataset; wild mushrooms vary by region,
  season, subspecies, and observer. Field descriptions may not match the
  training distribution.
- `predict_proba` output is uncalibrated; 0.05 is a risk-policy threshold, not
  a measured 5% toxicity probability.
- Zero dangerous approvals was achieved on this particular test set. That is
  not a statistical guarantee for unseen data; even one dangerous approval in
  deployment could be fatal.
- `veil-type` is constant and carries no signal but is kept to preserve the
  22-feature input contract.
- Always consult qualified experts and authoritative identification resources;
  never consume a wild mushroom based on this model.
