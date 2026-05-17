from __future__ import annotations

from django import forms


FIELD_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
}


class ChurnPredictionForm(forms.Form):
    def __init__(self, metadata: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        categorical_options = metadata.get("categorical_options", {})
        numeric_features = set(metadata.get("numeric_features", []))

        for column in metadata["feature_columns"]:
            label = FIELD_LABELS.get(column, column)
            if column == "SeniorCitizen":
                self.fields[column] = forms.ChoiceField(
                    label=label,
                    choices=[("0", "No"), ("1", "Yes")],
                    widget=forms.Select(attrs={"class": "input-control"}),
                )
            elif column in numeric_features:
                self.fields[column] = forms.DecimalField(
                    label=label,
                    min_value=0,
                    required=True,
                    widget=forms.NumberInput(
                        attrs={
                            "class": "input-control",
                            "step": "1" if column == "tenure" else "0.01",
                        }
                    ),
                )
            else:
                options = [(option, option) for option in categorical_options.get(column, [])]
                self.fields[column] = forms.ChoiceField(
                    label=label,
                    choices=options,
                    widget=forms.Select(attrs={"class": "input-control"}),
                )

