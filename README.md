# Customer Churn Prediction System

This project trains a churn classifier on the Telco Customer Churn dataset and serves it through a Django web app.

## Run

```powershell
python -m pip install -r requirements.txt
python -m src.churn.train_model
python manage.py runserver
```

Open the local Django URL shown in the terminal, usually `http://127.0.0.1:8000`.

## What is included

- `src/churn/train_model.py` trains preprocessing and classification pipelines, evaluates candidate models, and saves the best model.
- `models/churn_model.joblib` is created after training.
- `reports/metrics.json` and chart images are created after training.
- `predictor/` contains the Django form, views, URLs, and prediction service.
- `churn_project/` contains the Django project settings and root routing.
