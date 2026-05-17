from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python -m src.churn.train_model` first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, metadata, metrics


def default_form_values(metadata: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "SeniorCitizen": "0",
        "tenure": "12",
        "MonthlyCharges": "70.00",
        "TotalCharges": "840.00",
    }
    for column, options in metadata.get("categorical_options", {}).items():
        defaults[column] = options[0] if options else ""
    return defaults


def predict_churn(form_values: dict[str, Any], metadata: dict[str, Any], model: Any) -> dict[str, Any]:
    row_values = {}
    numeric_features = set(metadata["numeric_features"])

    for column in metadata["feature_columns"]:
        value = form_values.get(column)
        if column in numeric_features:
            row_values[column] = float(value or 0)
        else:
            row_values[column] = str(value or "")

    row = pd.DataFrame([row_values], columns=metadata["feature_columns"])
    probability = float(model.predict_proba(row)[0, 1])
    label, tone = risk_label(probability)
    return {
        "probability": probability,
        "percentage": round(probability * 100, 1),
        "label": label,
        "tone": tone,
    }


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High risk", "high"
    if probability >= 0.35:
        return "Medium risk", "medium"
    return "Low risk", "low"

