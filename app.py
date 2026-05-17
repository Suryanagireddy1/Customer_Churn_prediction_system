from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

app = Flask(__name__)


def load_artifacts():
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


def default_form_values(metadata: dict) -> dict[str, str]:
    defaults = {
        "tenure": "12",
        "MonthlyCharges": "70.00",
        "TotalCharges": "840.00",
        "SeniorCitizen": "0",
    }
    for column, options in metadata.get("categorical_options", {}).items():
        defaults[column] = options[0] if options else ""
    return defaults


def build_customer_row(form: dict, metadata: dict) -> pd.DataFrame:
    values = {}
    for column in metadata["feature_columns"]:
        if column in metadata["numeric_features"]:
            values[column] = float(form.get(column, 0) or 0)
        else:
            values[column] = form.get(column, "")
    return pd.DataFrame([values], columns=metadata["feature_columns"])


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High risk", "high"
    if probability >= 0.35:
        return "Medium risk", "medium"
    return "Low risk", "low"


@app.route("/", methods=["GET", "POST"])
def index():
    model, metadata, metrics = load_artifacts()
    form_values = default_form_values(metadata)
    prediction = None
    error = None

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            row = build_customer_row(form_values, metadata)
            probability = float(model.predict_proba(row)[0, 1])
            label, tone = risk_label(probability)
            prediction = {
                "probability": probability,
                "percentage": round(probability * 100, 1),
                "label": label,
                "tone": tone,
            }
        except ValueError:
            error = "Please enter valid numeric values for tenure, monthly charges, and total charges."

    best_model = metrics.get("best_model")
    best_metrics = metrics.get("models", {}).get(best_model, {}) if best_model else {}
    return render_template(
        "index.html",
        metadata=metadata,
        best_model=best_model,
        best_metrics=best_metrics,
        form_values=form_values,
        prediction=prediction,
        error=error,
    )


@app.route("/reports/<path:filename>")
def reports(filename: str):
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)

