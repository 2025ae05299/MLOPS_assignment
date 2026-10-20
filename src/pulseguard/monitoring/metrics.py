"""Process-local Prometheus metrics. Run one uvicorn worker so counts stay coherent."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "pulseguard_http_requests_total",
    "HTTP requests handled by PulseGuard.",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    "pulseguard_request_latency_seconds",
    "Request latency in seconds.",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
PREDICTIONS = Counter(
    "pulseguard_predictions_total",
    "Scored educational risk labels.",
    ["label", "risk_band"],
)
PROBABILITY = Histogram(
    "pulseguard_positive_probability",
    "Predicted probability of the positive educational label.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
