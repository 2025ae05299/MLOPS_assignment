"""Score one JSON patient record with the saved pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulseguard.config import HIGH_RISK_THRESHOLD, MODELS_DIR  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/infer.py examples/patient_elevated.json")
    payload = json.loads(Path(sys.argv[1]).read_text())
    bundle = joblib.load(MODELS_DIR / "pulseguard_pipeline.joblib")
    probability = float(bundle["pipeline"].predict_proba(pd.DataFrame([payload]))[0, 1])
    threshold = float(bundle["card"]["decision_threshold"])
    prediction = int(probability >= threshold)
    if probability >= float(bundle["card"].get("high_risk_threshold", HIGH_RISK_THRESHOLD)):
        band = "high"
    elif probability >= threshold:
        band = "moderate"
    else:
        band = "low"
    print(
        json.dumps(
            {
                "prediction": prediction,
                "probability_heart_disease": round(probability, 6),
                "confidence": round(probability if prediction else 1 - probability, 6),
                "risk_band": band,
                "model_name": bundle["card"]["model_name"],
                "disclaimer": bundle["card"]["disclaimer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
