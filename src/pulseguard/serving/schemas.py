"""Request and response contracts for the scoring API."""

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    """One UCI-style clinical row. Values are validated, not interpreted clinically."""

    age: int = Field(ge=18, le=100)
    sex: int = Field(ge=0, le=1, description="0 = female, 1 = male, using the UCI coding.")
    cp: int = Field(ge=1, le=4, description="Chest-pain code 1-4 as published by UCI.")
    trestbps: float = Field(ge=70, le=250)
    chol: float = Field(ge=80, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=60, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=1, le=7, description="UCI thal codes are typically 3, 6, or 7.")
    source_site: str = Field(default="unspecified", min_length=3, max_length=32)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 4,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
                "source_site": "cleveland",
            }
        }
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    label: str
    probability_heart_disease: float
    confidence: float
    risk_band: str
    model_name: str
    model_version: str
    decision_threshold: float
    request_id: str
    disclaimer: str
