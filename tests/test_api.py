import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from pulseguard.config import MODEL_INPUT_FIELDS
from pulseguard.features.builder import build_pipeline
from pulseguard.serving.app import create_app


def _write_tiny_model(tmp_path, frame):
    pipeline = build_pipeline(LogisticRegression(max_iter=400, random_state=5299))
    pipeline.fit(frame.loc[:, MODEL_INPUT_FIELDS], frame["heart_disease"])
    path = tmp_path / "tiny.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "card": {
                "model_name": "tiny_logreg",
                "model_version": "test",
                "decision_threshold": 0.5,
                "high_risk_threshold": 0.7,
                "disclaimer": "test only",
            },
        },
        path,
    )
    return path


def test_predict_contract(tmp_path, tiny_training_frame, clinical_row, monkeypatch):
    model = _write_tiny_model(tmp_path, tiny_training_frame)
    monkeypatch.setenv("PULSEGUARD_MODEL_PATH", str(model))
    client = TestClient(create_app())
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        response = client.post("/predict", json=clinical_row)
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in {0, 1}
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["risk_band"] in {"low", "moderate", "high"}
        assert "request_id" in body
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert b"pulseguard_predictions_total" in metrics.content


def test_rejects_impossible_cholesterol(tmp_path, tiny_training_frame, clinical_row, monkeypatch):
    model = _write_tiny_model(tmp_path, tiny_training_frame)
    monkeypatch.setenv("PULSEGUARD_MODEL_PATH", str(model))
    client = TestClient(create_app())
    with client:
        payload = dict(clinical_row)
        payload["chol"] = 0
        response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_health_is_degraded_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSEGUARD_MODEL_PATH", str(tmp_path / "missing.joblib"))
    client = TestClient(create_app())
    with client:
        response = client.get("/health")
    assert response.status_code == 503
