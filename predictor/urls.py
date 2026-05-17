from __future__ import annotations

from django.urls import path

from . import views


app_name = "predictor"

urlpatterns = [
    path("", views.index, name="index"),
    path("reports/<str:filename>/", views.report_file, name="report_file"),
]

