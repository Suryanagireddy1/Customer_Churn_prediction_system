from __future__ import annotations

from pathlib import Path

from django.http import FileResponse, Http404, HttpRequest, HttpResponse
from django.shortcuts import render

from .forms import ChurnPredictionForm
from .services import default_form_values, load_artifacts, predict_churn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

ACCOUNT_FIELDS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]
SERVICE_FIELDS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
BILLING_FIELDS = [
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def index(request: HttpRequest) -> HttpResponse:
    model, metadata, metrics = load_artifacts()
    prediction = None

    if request.method == "POST":
        form = ChurnPredictionForm(metadata, request.POST)
        if form.is_valid():
            prediction = predict_churn(form.cleaned_data, metadata, model)
    else:
        form = ChurnPredictionForm(metadata, initial=default_form_values(metadata))

    best_model = metrics.get("best_model")
    best_metrics = metrics.get("models", {}).get(best_model, {}) if best_model else {}

    context = {
        "form": form,
        "account_fields": [form[name] for name in ACCOUNT_FIELDS],
        "service_fields": [form[name] for name in SERVICE_FIELDS],
        "billing_fields": [form[name] for name in BILLING_FIELDS],
        "best_model": best_model,
        "best_model_label": best_model.replace("_", " ").title() if best_model else "Model ready",
        "best_metrics": best_metrics,
        "prediction": prediction,
    }
    return render(request, "predictor/index.html", context)


def report_file(request: HttpRequest, filename: str) -> FileResponse:
    allowed_reports = {"confusion_matrix.png", "roc_curve.png", "top_features.png"}
    if filename not in allowed_reports:
        raise Http404("Report not found.")

    report_path = REPORTS_DIR / filename
    if not report_path.exists():
        raise Http404("Report not found.")

    return FileResponse(report_path.open("rb"))

