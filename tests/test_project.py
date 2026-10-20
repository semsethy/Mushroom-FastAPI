import joblib
import pytest
from fastapi.testclient import TestClient
from app import ml
from app.main import app
from app.schemas import Mushroom


def test_training_quality_and_round_trip(bundle, examples, tmp_path):
    assert bundle['metrics']['accuracy'] > bundle['baseline_accuracy'] + .35
    assert bundle['metrics']['recall'] > .95
    assert bundle['metrics']['roc_auc'] > .98
    path = tmp_path / 'model.joblib'
    joblib.dump(bundle, path)
    assert ml.predict_mushroom(bundle, [Mushroom(**x) for x in examples]) == ml.predict_mushroom(joblib.load(path), [Mushroom(**x) for x in examples])


def test_health_metadata_and_contract(client):
    assert client.get('/health').json() == {'status': 'ok', 'models': {'mushroom': True}}
    info = client.get('/models').json()[0]
    assert info['loaded'] and len(info['features']) == 22
    spec = client.get('/openapi.json').json()
    assert '503' in spec['paths']['/mushroom/predict']['post']['responses']
    assert client.get('/docs').status_code == 200


def test_single_batch_and_order(client, examples):
    single = [client.post('/mushroom/predict', json=x).json() for x in examples]
    response = client.post('/mushroom/predict/batch', json=examples)
    assert response.status_code == 200
    assert response.json() == single
    assert [x['label'] for x in single] == ['edible', 'poisonous']
    for prediction in single:
        assert prediction['will_be_poisonous'] == (prediction['poisonous_probability'] >= prediction['threshold'])


@pytest.mark.parametrize('change', ['missing', 'extra', 'category', 'null', 'numeric'])
def test_invalid_inputs(client, examples, change):
    row = examples[0].copy()
    if change == 'missing': row.pop('odor')
    if change == 'extra': row['class'] = 'e'
    if change == 'category': row['odor'] = 'invalid'
    if change == 'null': row['odor'] = None
    if change == 'numeric': row['odor'] = 1
    assert client.post('/mushroom/predict', json=row).status_code == 422


def test_unknown_marker_and_batch_bounds(client, examples):
    row = examples[0] | {'stalk-root': '?'}
    assert client.post('/mushroom/predict', json=row).status_code == 200
    for rows in [[], [row] * 1001, [row, row | {'odor': 'invalid'}]]:
        assert client.post('/mushroom/predict/batch', json=rows).status_code == 422


def test_missing_model(monkeypatch, examples):
    monkeypatch.setattr(ml, 'load', lambda: None)
    with TestClient(app) as client:
        assert client.get('/health').json()['status'] == 'degraded'
        assert client.get('/models').json()[0]['loaded'] is False
        assert client.post('/mushroom/predict', json=examples[0]).status_code == 503
        assert client.post('/mushroom/predict/batch', json=examples).status_code == 503
