# 📉 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat)
![Render](https://img.shields.io/badge/Render-Live-brightgreen?style=flat&logo=render)
![Status](https://img.shields.io/badge/Status-Live-success?style=flat)

> An end-to-end machine learning system that predicts whether a telecom customer will churn, with full model explainability using SHAP — deployed live on Render.

🔗 **Live App:** [customer-churn-prediction-system-t4b3.onrender.com](https://customer-churn-prediction-system-t4b3.onrender.com)  
📁 **GitHub:** [github.com/Suryanagireddy1/Customer_Churn_prediction_system](https://github.com/Suryanagireddy1/Customer_Churn_prediction_system)

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the telecom industry — losing a customer costs 5x more than retaining one. This system predicts which customers are likely to leave, enabling businesses to take proactive retention action.

This project covers the complete ML workflow:
- Exploratory Data Analysis (EDA)
- Feature engineering (11 new features created)
- Handling class imbalance with SMOTE
- Model training, evaluation & comparison
- SHAP explainability for business insights
- Web deployment with Flask on Render

---

## 🎯 Problem Statement

Given customer data (contract type, tenure, monthly charges, services used), predict:
- `0` — Customer will **Stay**
- `1` — Customer will **Churn**

**Dataset:** IBM Telco Customer Churn Dataset — 7,043 customers, 21 features

---

## 🗂️ Project Structure

```
Customer_Churn_prediction_system/
│
├── app.py                    # Flask web application
├── model.py                  # Model training script
├── churn_model.pkl           # Trained XGBoost model
├── scaler.pkl                # Feature scaler
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_prediction.ipynb   # Full EDA & modelling notebook
│
├── templates/
│   └── index.html            # Web UI
│
├── static/
│   └── style.css             # Styling
│
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Library | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Deployment | Render |

---

## ⚙️ ML Pipeline

### 1. Exploratory Data Analysis
- Analyzed churn rate distribution — **73% non-churn vs 27% churn**
- Visualized churn by contract type, tenure, monthly charges
- Correlation heatmap to identify feature relationships
- Identified `TotalCharges` had blank strings — cleaned and converted to float

### 2. Feature Engineering
Created **11 new features** to improve model performance:

| New Feature | Description |
|---|---|
| `contract_tenure_ratio` | Contract length relative to tenure |
| `avg_monthly_spend` | Total charges divided by tenure |
| `service_bundling_score` | Count of active services |
| `tenure_group` | Binned tenure into Early / Mid / Long |
| `high_value_customer` | Flag for customers above avg spend |
| + 6 more interaction features | |

### 3. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique)
- Combined with **cost-sensitive XGBoost** (`scale_pos_weight`)
- Result: balanced training set without losing original data

### 4. Model Training & Evaluation

| Model | Accuracy | AUC-ROC | F1 (Churn) |
|---|---|---|---|
| Logistic Regression | 80.1% | 0.84 | 0.61 |
| Random Forest | 85.3% | 0.88 | 0.74 |
| XGBoost (baseline) | 87.2% | 0.79 | 0.71 |
| **XGBoost + Features + SMOTE** | **88.6%** | **0.91** | **0.88** |

> AUC-ROC improved from **0.79 → 0.91** through feature engineering alone.

### 5. SHAP Explainability

Applied SHAP (SHapley Additive exPlanations) to interpret model decisions:

**Top churn drivers identified:**
1. 🔴 **Contract Type** — Month-to-month customers churn most
2. 🔴 **Tenure** — Customers with < 12 months are highest risk
3. 🟡 **Monthly Charges** — Higher charges increase churn probability
4. 🟡 **Internet Service Type** — Fiber optic users churn more
5. 🟢 **Tech Support** — Having tech support reduces churn risk

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Suryanagireddy1/Customer_Churn_prediction_system.git
cd Customer_Churn_prediction_system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🌐 Live Demo

Deployed on **Render** and accessible at:  
👉 [https://customer-churn-prediction-system-t4b3.onrender.com](https://customer-churn-prediction-system-t4b3.onrender.com)

Enter customer details (contract type, tenure, monthly charges, services) and get an instant churn prediction with probability score.

---

## 📊 Key Results

- ✅ **AUC-ROC improved from 0.79 → 0.91** through feature engineering
- ✅ **F1-Score: 0.88** on minority (churn) class
- ✅ **88.6% overall accuracy** on test data
- ✅ SHAP identified top 5 actionable business retention drivers
- ✅ 11 new engineered features boosted model performance significantly

---

## 💡 Business Insights

Based on SHAP analysis, the following retention strategies are recommended:

| Insight | Action |
|---|---|
| Month-to-month contracts churn most | Offer discounts to switch to annual contracts |
| New customers (< 12 months) are high risk | Onboarding support program in first 3 months |
| High monthly charges increase churn | Introduce loyalty pricing for long-term customers |
| No tech support = higher churn | Bundle tech support in base plans |

---

## 📈 Future Improvements

- [ ] Add SHAP waterfall chart in the web UI for per-prediction explanation
- [ ] Build a customer segmentation dashboard (RFM analysis)
- [ ] Retrain model monthly with fresh data pipeline
- [ ] Add A/B testing framework for retention strategies
- [ ] Integrate with CRM systems via REST API

---

## 👨‍💻 Author

**Surya Nagireddy**  
MSc Computer Science — Dravidian University  
📧 suryanagireddy7564@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/surya-nagireddy-568728245)  
🐙 [GitHub](https://github.com/Suryanagireddy1)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this project helpful, please give it a star!**
