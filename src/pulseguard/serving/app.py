"""PulseGuard scoring API.

Educational service only. Logs a payload fingerprint rather than raw clinical values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pulseguard.config import DISCLAIMER, HIGH_RISK_THRESHOLD, MODELS_DIR
from pulseguard.monitoring.metrics import LATENCY, PREDICTIONS, PROBABILITY, REQUESTS
from pulseguard.serving.schemas import PatientRecord, PredictionResponse

LOGGER = logging.getLogger("pulseguard.api")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.getenv("PULSEGUARD_LOG_LEVEL", "INFO"))


def model_path() -> str:
    return os.getenv("PULSEGUARD_MODEL_PATH", str(MODELS_DIR / "pulseguard_pipeline.joblib"))


def load_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if "pipeline" not in bundle or "card" not in bundle:
        raise RuntimeError("Model bundle is missing pipeline or card")
    return bundle


def _attach_model(app: FastAPI) -> None:
    path = model_path()
    try:
        bundle = load_bundle(path)
        app.state.pipeline = bundle["pipeline"]
        app.state.card = bundle["card"]
        app.state.ready = True
        LOGGER.info("model loaded", extra={"fields": {"model_path": path}})
    except Exception as exc:  # noqa: BLE001 - surface a clear degraded state
        app.state.pipeline = None
        app.state.card = {}
        app.state.ready = False
        LOGGER.error("model load failed", extra={"fields": {"error": str(exc), "model_path": path}})


def create_app() -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _attach_model(app)
        yield

    app = FastAPI(
        title="PulseGuard",
        version="1.0.0",
        description=DISCLAIMER,
        lifespan=lifespan,
    )
    _attach_model(app)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        endpoint = request.url.path
        REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        return response

    @app.get("/health")
    def health():
        status_code = 200 if app.state.ready else 503
        body = {
            "status": "ok" if app.state.ready else "degraded",
            "model_version": app.state.card.get("model_version"),
            "model_name": app.state.card.get("model_name"),
            "disclaimer": DISCLAIMER,
        }
        return JSONResponse(body, status_code=status_code)

    @app.get("/model-info")
    def model_info():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        card = dict(app.state.card)
        card.pop("candidates", None)
        return card

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(record: PatientRecord):
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        request_id = uuid.uuid4().hex[:12]
        frame = pd.DataFrame([record.model_dump()])
        probability = float(app.state.pipeline.predict_proba(frame)[0, 1])
        threshold = float(app.state.card.get("decision_threshold", 0.5))
        high_threshold = float(app.state.card.get("high_risk_threshold", HIGH_RISK_THRESHOLD))
        prediction = int(probability >= threshold)
        if probability >= high_threshold:
            risk_band = "high"
        elif probability >= threshold:
            risk_band = "moderate"
        else:
            risk_band = "low"
        label = "elevated_educational_risk" if prediction == 1 else "lower_educational_risk"
        confidence = probability if prediction == 1 else 1.0 - probability
        fingerprint = hashlib.sha256(
            json.dumps(record.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:12]
        LOGGER.info(
            "prediction served",
            extra={
                "fields": {
                    "request_id": request_id,
                    "payload_fingerprint": fingerprint,
                    "prediction": prediction,
                    "risk_band": risk_band,
                    "probability": round(probability, 4),
                    "source_site": record.source_site,
                }
            },
        )
        PREDICTIONS.labels(label, risk_band).inc()
        PROBABILITY.observe(probability)
        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability_heart_disease=round(probability, 6),
            confidence=round(confidence, 6),
            risk_band=risk_band,
            model_name=str(app.state.card.get("model_name", "unknown")),
            model_version=str(app.state.card.get("model_version", "unknown")),
            decision_threshold=threshold,
            request_id=request_id,
            disclaimer=DISCLAIMER,
        )

    return app


app = create_app()
