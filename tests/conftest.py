import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import ml
from app.main import app
from train_mushroom import train, DATA_PATH


@pytest.fixture(scope='session')
def bundle(tmp_path_factory):
    return train(model_path=tmp_path_factory.mktemp('models'), verbose=False)


@pytest.fixture
def client(bundle, monkeypatch):
    monkeypatch.setattr(ml, 'load', lambda: bundle)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def examples():
    df = pd.read_csv(DATA_PATH)
    return [df[df['class'] == label].drop(columns='class').iloc[0].to_dict() for label in ['e', 'p']]
