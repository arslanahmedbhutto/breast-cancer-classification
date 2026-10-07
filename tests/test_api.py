import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EXAMPLE
from src.data import FEATURES, load_data


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["features"] == FEATURES
    assert body["metrics"]["test_roc_auc"] > 0.95


def test_predict_malignant_example(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["diagnosis"] == "malignant"
    assert 0.5 <= body["probability_malignant"] <= 1


def test_predict_batch_matches_labels(client):
    X, y = load_data()
    rows = X.iloc[:20]
    r = client.post("/predict/batch", json={"samples": rows.to_dict(orient="records")})
    assert r.status_code == 200
    preds = [p["diagnosis"] == "malignant" for p in r.json()["predictions"]]
    assert len(preds) == 20
    assert sum(p == bool(t) for p, t in zip(preds, y.iloc[:20])) >= 18


def test_missing_feature_rejected(client):
    payload = {k: v for k, v in EXAMPLE.items() if k != "radius_mean"}
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_value_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "area_mean": -1}).status_code == 422


def test_unknown_field_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "age": 50}).status_code == 422
