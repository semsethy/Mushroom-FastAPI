# Mushroom classification with FastAPI

A simple classroom project following the structure of the teacher’s **Plato** project:
explore data in a notebook, compare scikit-learn models, save a joblib bundle, and serve
predictions with FastAPI. No database, Docker, or external services.

## Project files

```text
notebooks/mushroom_model.ipynb   Step-by-step exploration, plots, training and saving
train_mushroom.py               The same training workflow in one script
data/mushroom/mushrooms.csv     8,124 rows; 22 categorical features and class label
models/mushroom.joblib          Generated fitted pipeline and metadata
app/ml.py                      Load the model and predict
app/schemas.py                 Input codes and response validation
app/main.py                    API startup and routes
requests.http                  Example requests, including an invalid batch
tests/                         Training quality and API tests
```

## 1. Setup

Use Python 3.12 (the version used to verify this project).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code, open `notebooks/mushroom_model.ipynb`, select `.venv` as the Python kernel,
and run all cells. The notebook includes saved outputs and plots.

## 2. Understand and train the model

The target `class` is `e` for edible or `p` for poisonous. All predictors are category
codes. For example, `odor=n` means no odor; `odor=p` means pungent. Codes depend on the
column: do not interpret them as numbers or reuse a code’s meaning across columns.
The notebook shows category counts for each column; `/docs` lists accepted codes.

There are no duplicate rows or ordinary nulls in the supplied file. `stalk-root` has
2,480 `?` values, retained as an explicit unknown category. `veil-type` is constant.

The workflow matches the teacher’s churn example:

1. Inspect the dataset, missing values and class balance.
2. Reserve a stratified 20% test set with `random_state=42`.
3. Measure a majority-class baseline.
4. Compare logistic regression, decision tree and random forest with five-fold ROC AUC.
5. Choose a threshold by poisonous-class F1 using out-of-fold training probabilities.
6. Fit the winner on training rows and evaluate on the held-out rows.
7. Explain predictions with permutation importance in the notebook.
8. Save the encoder and classifier together with features, threshold and metadata.

One-hot encoding is inside each pipeline, so it learns only from the training fold.
The API rejects category codes outside the supplied dataset’s vocabulary. Only `stalk-root`
accepts `?`. All 22 features are required, and unknown fields are rejected.

To train without opening the notebook:

```bash
python train_mushroom.py
```

The verified run selected **random forest**, with threshold **0.5**. Held-out accuracy,
precision, poisonous recall, F1 and ROC AUC were all **1.0** on 1,625 test rows (842 edible,
783 poisonous), versus **0.5182** majority baseline accuracy. These are random hold-out
results on this dataset, not evidence of generalization to unseen species or field observations.
The model’s probability estimates are not calibrated safety guarantees. This project is
for learning and must not be used to decide whether to eat wild mushrooms.

## 3. Run the API

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try requests. The generated OpenAPI contract is at
`/openapi.json`. Use `requests.http` with the VS Code REST Client extension if preferred.

| Method | Path | Result |
|---|---|---|
| GET | `/health` | Status and model availability |
| GET | `/models` | Algorithm, training date, metrics, threshold and features |
| POST | `/mushroom/predict` | One mushroom prediction |
| POST | `/mushroom/predict/batch` | 1–1,000 predictions in input order |

Input JSON uses the CSV’s hyphenated feature names. Copy the complete example from `/docs`
or `requests.http`. The response contains `label`, `poisonous_probability`,
`will_be_poisonous` and `threshold`. The label is poisonous when its probability is at
least the saved threshold. Invalid requests return `422`.

Models load once at startup. If the bundle is missing, health reports `degraded`, metadata
reports `loaded: false`, and valid prediction requests return `503` with the training command.
Restart the API after retraining. Only load trusted joblib files, and use the same
scikit-learn version for training and serving. Model binaries are ignored by Git; recreate
one with the notebook or training script.

## 4. Test

```bash
python -m pytest -q
```

Tests train into a temporary directory, leaving your saved model alone. They check held-out
quality against the baseline, serialization, health, metadata, documentation, predictions,
batch order and bounds, invalid inputs, and missing-model behavior.
