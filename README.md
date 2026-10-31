# Telecom Customer Churn Prediction

A business intelligence solution that predicts which telecom customers are likely to leave, built with **Python, scikit-learn (Random Forest) and Streamlit**.

> University project · COMP8811 Data Analytics & Intelligence · Rehaan

<!-- Live demo: add your Streamlit link here once deployed, e.g.
**[▶ Try the live dashboard](https://your-app.streamlit.app)** -->

## The problem

Churn is when a customer stops using a service. Spotting at-risk customers early lets a business focus its retention efforts where they matter. This project trains a model on the IBM Telco Customer Churn dataset and wraps it in an interactive dashboard that business users can explore.

## What the dashboard does

The Streamlit app has three pages:

| Page | What you see |
|---|---|
| **Overview & Charts** | Headline KPIs (customers, churn count, churn rate, average monthly charge), churn distribution and churn rate by contract type |
| **Model Analysis** | Baseline vs improved model metrics, confusion matrices and feature importance |
| **Customer Prediction** | A 19-field form for one customer that returns a churn-risk score and suggested retention actions |

## Results

Both models were evaluated on the same 1,407 held-out test customers (20% split).

| Metric | Baseline Random Forest | Improved Random Forest |
|---|---|---|
| Accuracy | 78.54% | 72.00% |
| Precision | 62.41% | 48.34% |
| **Recall** | 48.40% | **78.07%** |
| F1-score | 54.52% | 59.71% |
| ROC-AUC | 81.44% | 83.34% |

The improved model was tuned for **recall**, because for retention, missing a customer who is about to leave costs more than a false alarm. It catches **292 of 374 churners** (up from 181), cutting missed churners from 193 to 82, at the cost of more false alerts.

## Approach

1. **Explore** the raw data: 7,043 customers, 21 columns (`scripts/explore_data.py`)
2. **Clean** it: convert `TotalCharges` to numeric, drop the 11 rows with blanks, drop `customerID`, encode `Churn` as 0/1, leaving 7,032 rows (`scripts/prepare_data.py`)
3. **Baseline model**: one-hot encoding + Random Forest (200 trees, balanced class weights), 80/20 stratified split (`scripts/train_model.py`)
4. **Improved model**: `RandomizedSearchCV` with 5-fold cross-validation optimising recall (`scripts/improve_model.py`)
5. **Dashboard**: Streamlit app loading both saved pipelines (`app.py`)

## Project structure

```
├── app.py                  # Streamlit dashboard
├── data/
│   ├── WA_FnUseC_TelcoCustomerChurn.csv   # raw IBM Telco dataset
│   └── telco_churn_clean.csv              # cleaned dataset
├── model/
│   ├── churn_model.pkl                    # baseline pipeline
│   └── improved_churn_model.pkl           # tuned pipeline
├── scripts/                # data prep and training scripts
├── docs/
│   ├── technical-report.docx
│   └── presentation.pptx
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/000RS/telecom-churn-prediction.git
cd telecom-churn-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The saved models were trained with scikit-learn 1.6.1, so keep that version pinned. To retrain from scratch, run the scripts in order from the project root:

```bash
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/improve_model.py
```

## Tech stack

Python · pandas · scikit-learn · Streamlit · Matplotlib · joblib

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), a public sample dataset of 7,043 customers.

## Documentation

- [Technical report](docs/technical-report.docx)
- [Presentation slides](docs/presentation.pptx)
