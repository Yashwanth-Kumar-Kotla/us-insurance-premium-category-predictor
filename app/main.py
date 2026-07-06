"""
FastAPI service for the US insurance premium-category predictor.

Run locally:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs

This mirrors the structure of a typical dockerized FastAPI ML service so you
can lift the schema + endpoints straight into your existing project.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

app = FastAPI(
    title="US Insurance Premium Category Predictor",
    description="Predicts a premium band (Low / Medium / High / Very High) "
    "from US insurance applicant features.",
    version="1.0.0",
)

# Loaded once at startup.
_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


# ---------------------------------------------------------------------------
# Request / response schemas  (US feature set)
# ---------------------------------------------------------------------------
class ApplicantFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    sex: Literal["male", "female"] = Field(..., example="male")
    bmi: float = Field(..., gt=0, lt=80, example=28.5)
    children: int = Field(..., ge=0, le=15, example=2)
    smoker: Literal["yes", "no"] = Field(..., example="no")
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., example="northeast"
    )


class PredictionResponse(BaseModel):
    premium_category: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/categories")
def categories():
    """Explain what each premium category means in USD."""
    art = get_artifact()
    edges = art["bin_edges_usd"]
    labels = art["categories"]
    return {
        labels[i]: {
            "usd_charges_from": round(edges[i], 2),
            "usd_charges_to": round(edges[i + 1], 2),
        }
        for i in range(len(labels))
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ApplicantFeatures):
    art = get_artifact()
    pipe = art["pipeline"]

    row = pd.DataFrame([features.model_dump()])[art["features"]]
    pred = pipe.predict(row)[0]

    proba = pipe.predict_proba(row)[0]
    classes = pipe.named_steps["model"].classes_
    probabilities = {
        str(cls): round(float(p), 4) for cls, p in zip(classes, proba)
    }

    return PredictionResponse(
        premium_category=str(pred), probabilities=probabilities
    )
