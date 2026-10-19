"""FastAPI application exposing the trained mushroom classifier."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Request

from app import __version__, ml
from app.schemas import (
    HealthResponse,
    ModelInfo,
    MushroomFeatures,
    MushroomPrediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    try:
        app.state.bundle = ml.load_bundle()
        app.state.model_error = None
    except (FileNotFoundError, ValueError) as exc:
        app.state.bundle = None
        app.state.model_error = str(exc)
    yield
    app.state.bundle = None


app = FastAPI(
    title="Mushroom Classification API",
    description=(
        "Predict whether a mushroom is edible or poisonous from 22 categorical "
        "characteristics. Educational use only—never use this API to decide whether "
        "a wild mushroom is safe to eat."
    ),
    version=__version__,
    lifespan=lifespan,
)


def get_bundle(request: Request) -> dict[str, Any]:
    """Return the loaded bundle or a useful service-unavailable response."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        detail = getattr(request.app.state, "model_error", None) or (
            "The model is unavailable. Run `python mushroom_train.py`."
        )
        raise HTTPException(status_code=503, detail=detail)
    return bundle


@app.get("/", include_in_schema=False)
def home() -> dict[str, str]:
    return {
        "name": "Mushroom Classification API",
        "docs": "/docs",
        "health": "/health",
        "model": "/models",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> dict[str, Any]:
    loaded = getattr(request.app.state, "bundle", None) is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}


@app.get("/models", response_model=ModelInfo, tags=["service"])
def model_info(request: Request) -> dict[str, Any]:
    bundle = get_bundle(request)
    return {
        "name": "mushroom",
        "algorithm": bundle["algorithm"],
        "features": bundle["features"],
        "threshold": bundle["threshold"],
        "metrics": bundle["metrics"],
        "trained_at": bundle["trained_at"],
        "sklearn_version": bundle["sklearn_version"],
    }


@app.post(
    "/predict",
    response_model=MushroomPrediction,
    tags=["prediction"],
    summary="Classify one mushroom",
)
@app.post(
    "/mushroom/predict",
    response_model=MushroomPrediction,
    tags=["prediction"],
    include_in_schema=False,
)
def predict_mushroom(
    mushroom: MushroomFeatures, request: Request
) -> dict[str, Any]:
    record = mushroom.model_dump(by_alias=True)
    return ml.predict_mushrooms(get_bundle(request), [record])[0]


@app.post(
    "/predict/batch",
    response_model=list[MushroomPrediction],
    tags=["prediction"],
    summary="Classify up to 1,000 mushrooms",
)
def predict_mushroom_batch(
    mushrooms: Annotated[list[MushroomFeatures], Body(min_length=1, max_length=1000)],
    request: Request,
) -> list[dict[str, Any]]:
    records = [item.model_dump(by_alias=True) for item in mushrooms]
    return ml.predict_mushrooms(get_bundle(request), records)
