from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return features, target


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def model_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        min_samples_leaf=3,
                        n_estimators=300,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def score_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_plots(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No churn", "Churn"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2563eb", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], color="#8b95a5", linestyle="--", label="Baseline")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=160)
    plt.close()

    model_step = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model_step, "coef_"):
        importances = abs(model_step.coef_[0])
    elif hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    else:
        return

    top_features = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    top_features["feature"] = top_features["feature"].str.replace(
        r"^(numeric|categorical)__", "", regex=True
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_features, y="feature", x="importance", color="#2563eb")
    plt.title("Top Model Signals")
    plt.xlabel("Relative importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(reports_dir / "top_features.png", dpi=160)
    plt.close()


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    df = load_dataset(data_path)
    features, target = split_features_target(df)
    preprocessor, numeric_features, categorical_features = build_preprocessor(features)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in model_candidates(preprocessor).items():
        model.fit(x_train, y_train)
        results[name] = score_model(model, x_test, y_test)
        fitted_models[name] = model

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = fitted_models[best_name]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)

    categorical_options = {
        column: sorted(str(value) for value in df[column].dropna().unique())
        for column in categorical_features
    }

    metadata = {
        "model_name": best_name,
        "target_column": TARGET_COLUMN,
        "feature_columns": list(features.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_options": categorical_options,
        "churn_rate": float(target.mean()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }
    metrics = {
        "best_model": best_name,
        "models": results,
    }

    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(make_json_safe(metrics), indent=2), encoding="utf-8")
    save_plots(best_model, x_test, y_test, metrics_path.parent)

    return {"metadata": metadata, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the customer churn model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    output = train(args.data, args.model, args.metadata, args.metrics)
    best = output["metrics"]["best_model"]
    best_metrics = output["metrics"]["models"][best]
    print(f"Saved model: {args.model}")
    print(f"Best model: {best}")
    print(f"ROC AUC: {best_metrics['roc_auc']:.3f}")
    print(f"Recall: {best_metrics['recall']:.3f}")
    print(f"Precision: {best_metrics['precision']:.3f}")


if __name__ == "__main__":
    main()
