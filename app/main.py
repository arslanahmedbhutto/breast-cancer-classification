"""FastAPI service for breast cancer (benign vs malignant) prediction.

Run:
    uvicorn app.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import BatchRequest, BatchResponse, ModelInfo, Prediction, TumorFeatures
from src.data import LABELS, ROOT

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python -m src.train` first.")
    artifact.update(joblib.load(MODEL_PATH))
    yield
    artifact.clear()


app = FastAPI(
    title="Breast Cancer Classifier",
    description="Predicts whether a breast mass is benign or malignant from 30 FNA "
    "cell-nucleus features (Wisconsin Diagnostic Breast Cancer dataset). "
    "For educational use only, not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


def _predict(samples: list[TumorFeatures]) -> list[Prediction]:
    if not artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([s.model_dump() for s in samples])[artifact["features"]]
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    return [
        Prediction(diagnosis=LABELS[int(p >= 0.5)], probability_malignant=round(float(p), 4))
        for p in proba
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(artifact)}


@app.get("/model", response_model=ModelInfo)
def model_info():
    return ModelInfo(**{k: v for k, v in artifact.items() if k != "pipeline"})


@app.post("/predict", response_model=Prediction)
def predict(sample: TumorFeatures):
    return _predict([sample])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    return BatchResponse(predictions=_predict(request.samples))
