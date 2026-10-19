"""Integration tests for the FastAPI endpoints and real saved model."""

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EXAMPLE_MUSHROOM

EDIBLE_MUSHROOM = {
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "y",
    "bruises": "t",
    "odor": "a",
    "gill-attachment": "f",
    "gill-spacing": "c",
    "gill-size": "b",
    "gill-color": "k",
    "stalk-shape": "e",
    "stalk-root": "c",
    "stalk-surface-above-ring": "s",
    "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w",
    "stalk-color-below-ring": "w",
    "veil-type": "p",
    "veil-color": "w",
    "ring-number": "o",
    "ring-type": "p",
    "spore-print-color": "n",
    "population": "n",
    "habitat": "g",
}


def test_health_and_model_metadata() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}

        response = client.get("/models")
        assert response.status_code == 200
        payload = response.json()
        assert payload["name"] == "mushroom"
        assert len(payload["features"]) == 22
        assert 0 <= payload["threshold"] <= 1
        assert "accuracy" in {key.lower() for key in payload["metrics"]}


def test_predicts_poisonous_mushroom() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=EXAMPLE_MUSHROOM)

    assert response.status_code == 200
    prediction = response.json()
    assert prediction["prediction"] == "poisonous"
    assert prediction["is_poisonous"] is True
    assert prediction["poisonous_probability"] >= prediction["threshold"]


def test_predicts_edible_mushroom_through_compatibility_route() -> None:
    with TestClient(app) as client:
        response = client.post("/mushroom/predict", json=EDIBLE_MUSHROOM)

    assert response.status_code == 200
    assert response.json()["prediction"] == "edible"


def test_batch_prediction() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch", json=[EXAMPLE_MUSHROOM, EDIBLE_MUSHROOM]
        )

    assert response.status_code == 200
    assert [item["prediction"] for item in response.json()] == [
        "poisonous",
        "edible",
    ]


def test_rejects_invalid_category() -> None:
    invalid = {**EXAMPLE_MUSHROOM, "odor": "invalid"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)

    assert response.status_code == 422


def test_rejects_missing_feature() -> None:
    incomplete = EXAMPLE_MUSHROOM.copy()
    incomplete.pop("habitat")
    with TestClient(app) as client:
        response = client.post("/predict", json=incomplete)

    assert response.status_code == 422
