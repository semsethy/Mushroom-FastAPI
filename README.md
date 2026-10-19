# Mushroom Classification API

A FastAPI service that loads the model trained in the mushroom notebook and predicts whether a mushroom is edible or poisonous from 22 categorical characteristics.

> This project is educational. Never use a model prediction to decide whether a wild mushroom is safe to eat.

## Project structure

```text
app/
  __init__.py       Package metadata
  main.py           FastAPI application and endpoints
  ml.py             Model loading and inference
  schemas.py        Pydantic request/response validation
data/
  mushrooms.csv     Training dataset
models/
  mushroom.joblib   Saved model bundle
tests/
  test_api.py       API integration tests
mushroom_train.py   Reproducible training program
requirements.txt    Python dependencies
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the model

```powershell
python mushroom_train.py
```

The script compares logistic regression, decision tree, random forest, extra trees, and a calibrated support-vector machine using five-fold cross-validation. It selects a decision threshold from out-of-fold training probabilities, evaluates the locked test set once, and writes `models/mushroom.joblib`.

## Start the API

```powershell
python -m uvicorn app.main:app --reload
```

Open the interactive documentation at <http://127.0.0.1:8000/docs>.

## Endpoints

- `GET /health` — service and model status
- `GET /models` — algorithm, feature order, threshold, metrics, and model version
- `POST /predict` — classify one mushroom
- `POST /predict/batch` — classify 1–1,000 mushrooms
- `POST /mushroom/predict` — compatibility alias for single prediction

Example request bodies are available in Swagger UI at `/docs`.

## Run tests

```powershell
python -m pytest -q
```

To load a bundle from another location, set `MUSHROOM_MODEL_PATH` before starting the API.
