# AI-Based Credit Risk Prediction & Underwriting System

An end-to-end Machine Learning credit scoring and loan default risk prediction platform built in Python. The system provides automated underwriting decisions, FICO-like credit score scaling ($300 - 850$), risk tiering, Explainable AI (XAI) risk drivers, and an interactive Streamlit web dashboard.

---

## 🌟 Key Features

1. **Synthetic & Custom Dataset Support**: Includes realistic financial dataset generator simulating bank credit applications (`data/generate_dataset.py`).
2. **Feature Engineering**: Automated domain features (`loan_to_income_ratio`, `repayment_to_income_ratio`, `cred_hist_to_age_ratio`).
3. **Multi-Model Suite & Benchmarking**:
   - **Logistic Regression** (Scorecard baseline)
   - **Random Forest Classifier** (Non-linear ensemble)
   - **XGBoost Classifier** (High performance gradient boosting)
4. **Credit Scorecard Engine**: Scales default probabilities to standard credit scores ($300 - 850$) using Points to Double Odds (PDO) log-odds formulas.
5. **Risk Tiering & Automated Recommendations**:
   - **Low Risk (Score >= 720)**: `APPROVE`
   - **Medium Risk (Score 620 - 719)**: `MANUAL REVIEW`
   - **High Risk (Score < 620)**: `DECLINE`
6. **Explainable AI (XAI)**: Global feature importance rankings and localized applicant risk drivers & mitigating factors.
7. **Interactive Streamlit Web App**:
   - **Single Applicant Assessment**: Live loan evaluation with interactive Plotly gauge chart.
   - **Batch Portfolio Processing**: Upload CSV of applicants and download scored underwriting report.
   - **Model Analytics**: ROC-AUC curves, Gini index, KS-statistic, and feature importance visualizer.

---

## 🛠️ Project Structure

```
credit_risk_prediction/
├── data/
│   └── generate_dataset.py       # Synthetic dataset generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading, cleaning, & splitting
│   ├── preprocessing.py         # Sklearn feature engineering & pipeline transformers
│   ├── models.py                # Model training & metrics (ROC-AUC, Gini, KS Stat)
│   ├── scorecard.py             # Log-odds credit scorecard engine (300-850)
│   └── explainability.py        # Global feature importance & local risk explanations
├── app.py                       # Streamlit web application
├── train_model.py               # Main CLI script to train models and save artifacts
├── test_credit_risk.py          # Pytest automated test suite
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```powershell
py -m pip install -r requirements.txt
```

### 2. Train Models & Build Scorecard
Run the training script to generate dataset, evaluate all models (Logistic Regression, Random Forest, XGBoost), select the best performer, and save `models/credit_risk_model.joblib`:

```powershell
py train_model.py
```

### 3. Run Automated Tests
Run unit tests with `pytest`:

```powershell
py -m pytest test_credit_risk.py -v
```

### 4. Launch Interactive Web Dashboard
Run the Streamlit web application:

```powershell
py -m streamlit run app.py
```

---

## 📊 Evaluation Metrics

The system calculates standard credit risk quantitative metrics:
- **ROC-AUC**: Receiver Operating Characteristic Area Under Curve
- **Gini Coefficient**: $2 \times \text{ROC-AUC} - 1$
- **KS-Statistic**: Kolmogorov-Smirnov maximum separation distance between cumulative default and non-default distributions.
- **Precision, Recall, & F1 Score**

---

## 🔒 Credit Scorecard Scaling Formula

Default Probability $P(\text{Default})$ is mapped to credit scores using:

$$\text{Score} = \text{Offset} + \text{Factor} \times \ln\left(\frac{1 - P(\text{Default})}{P(\text{Default})}\right)$$

Where:
- $\text{Offset} = 600$
- $\text{Factor} = \frac{50}{\ln(2)} \approx 72.13$
