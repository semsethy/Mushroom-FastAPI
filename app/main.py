from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app import ml
from app.schemas import Mushroom, MushroomPrediction
from typing import Annotated
from fastapi import Body
from app.schemas import ChurnPrediction, Customer, Feedback, ModelInfo, SentimentPrediction

MODELS: dict[str, dict | None] = {}
HOW_TO_TRAIN = {"churn": "python train_churn.py", "sentiment": "python train_sentiment.py", "mushroom": "python train_mushroom.py"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in HOW_TO_TRAIN:
        MODELS[name] = ml.load(name)
    yield
    MODELS.clear()


app = FastAPI(
    title="Mushroom, churn & sentiment API",
    description="Classroom ML predictions, including edible/poisonous mushroom classification. Educational use only; do not use to decide whether to eat a mushroom.",
    version="1.0.0",
    lifespan=lifespan,
)


def model(name: str) -> dict:
    """The loaded model, or a clear error if it has not been trained yet."""
    bundle = MODELS.get(name)
    if bundle is None:
        hint = f"The {name} model is not trained yet: run `{HOW_TO_TRAIN[name]}`."
        raise HTTPException(status_code=503, detail=hint)
    return bundle


@app.get("/", include_in_schema=False)
def home() -> dict:
    return {"message": "Churn & sentiment API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "models": {name: bundle is not None for name, bundle in MODELS.items()}}


@app.get("/models", response_model=list[ModelInfo], tags=["service"])
def models() -> list[dict]:
    """Which models are loaded, how they were trained and how well they scored."""
    return [
        {"name": name, "loaded": False}
        if bundle is None
        else {
            "name": name,
            "loaded": True,
            "algorithm": bundle["algorithm"],
            "trained_at": bundle["trained_at"],
            "metrics": bundle["metrics"],
        }
        for name, bundle in MODELS.items()
    ]


@app.post("/churn/predict", response_model=ChurnPrediction, tags=["churn"])
def predict_churn(customer: Customer) -> dict:
    """Will this customer leave?"""
    return ml.predict_churn(model("churn"), [customer.model_dump()])[0]



@app.post("/sentiment/predict", response_model=SentimentPrediction, tags=["sentiment"])
def predict_sentiment(feedback: Feedback) -> dict:
    """Is this feedback negative, neutral or positive?"""
    return ml.predict_sentiment(model("sentiment"), [feedback.text])[0]



@app.post("/mushroom/predict", response_model=MushroomPrediction, tags=["mushroom"])
def predict_mushroom(mushroom: Mushroom) -> dict:
    """Classify category codes from the Kaggle dataset. Educational use only."""
    return ml.predict_mushroom(model("mushroom"), [mushroom.model_dump(by_alias=True)])[0]


@app.post("/mushroom/predict/batch", response_model=list[MushroomPrediction], tags=["mushroom"])
def predict_mushroom_batch(
    mushrooms: Annotated[list[Mushroom], Body(min_length=1, max_length=1000)],
) -> list[dict]:
    return ml.predict_mushroom(
        model("mushroom"), [item.model_dump(by_alias=True) for item in mushrooms]
    )
