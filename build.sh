#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.churn.train_model
python manage.py collectstatic --no-input
