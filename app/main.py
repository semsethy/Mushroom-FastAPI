"""Run: fastapi dev app/main.py"""
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, HTTPException, Request, Body
from app import ml
from app.schemas import Mushroom, MushroomPrediction, ModelInfo


@asynccontextmanager
async def lifespan(app):
    app.state.models = {'mushroom': ml.load()}
    yield
    app.state.models.clear()


app = FastAPI(title='Mushroom Prediction API', version='1.0.0', lifespan=lifespan,
              description='Classroom classifier using categorical mushroom codes. Not a guide to eating wild mushrooms.')
UNAVAILABLE = {503: {'description': 'Model missing. Train and restart the API.'}}


def require_model(request):
    bundle = request.app.state.models['mushroom']
    if bundle is None:
        raise HTTPException(503, 'Model not loaded. Run python train_mushroom.py and restart the API.')
    return bundle


@app.get('/health')
def health(request: Request):
    loaded = request.app.state.models['mushroom'] is not None
    return {'status': 'ok' if loaded else 'degraded', 'models': {'mushroom': loaded}}


@app.get('/models', response_model=list[ModelInfo])
def models(request: Request):
    bundle = request.app.state.models['mushroom']
    return [dict(name='mushroom', loaded=bundle is not None,
                 **({key: bundle[key] for key in ['algorithm', 'trained_at', 'metrics', 'threshold', 'features', 'sklearn_version']} if bundle else {}))]


@app.post('/mushroom/predict', response_model=MushroomPrediction, responses=UNAVAILABLE)
def predict(item: Mushroom, request: Request):
    return ml.predict_mushroom(require_model(request), [item])[0]


@app.post('/mushroom/predict/batch', response_model=list[MushroomPrediction], responses=UNAVAILABLE)
def predict_batch(items: Annotated[list[Mushroom], Body(min_length=1, max_length=1000)], request: Request):
    return ml.predict_mushroom(require_model(request), items)
