# Employee Attrition Prediction

Predicts whether an employee is likely to leave a company, using the IBM HR Analytics dataset. Built as an end-to-end ML project — from raw data to a deployed prediction app.

## Problem
Employee attrition is costly for businesses (recruitment, training, lost productivity). This project builds a model to flag employees at risk of leaving, so HR can intervene early.

## Approach
1. **EDA & Cleaning** — explored attrition patterns across department, overtime, marital status, and job satisfaction; dropped constant/irrelevant columns.
2. **Encoding** — label encoding for ordinal features (BusinessTravel), one-hot encoding for nominal features (Department, JobRole, etc.)
3. **Class Imbalance Handling** — the dataset is ~84% "No" / 16% "Yes", so accuracy alone is misleading. Tested two approaches:
   - `class_weight='balanced'` (algorithm-level reweighting)
   - SMOTE (synthetic oversampling of the minority class)
4. **Model Comparison** — trained and evaluated Logistic Regression, Random Forest, and XGBoost, each with both imbalance strategies.
5. **Threshold Tuning** — adjusted the default 0.5 classification threshold to optimize for F1-score / recall trade-offs relevant to a real HR use case.
6. **Deployment** — built a Streamlit app for real-time predictions.

## Results

| Model | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (class_weight) | 0.37 | 0.66 | 0.47 | 0.80 |
| XGBoost (SMOTE) | 0.67 | 0.30 | 0.41 | 0.80 |
| **XGBoost (SMOTE, threshold=0.3)** | **0.65** | **0.43** | **0.51** | **0.80** |

**Final model: XGBoost + SMOTE, threshold=0.3** — chosen for the best F1-score, balancing recall (catching actual leavers) against precision (keeping predictions trustworthy enough for HR to act on).

## Key Learnings
- Accuracy is a misleading metric on imbalanced data — a model predicting "No" for everyone still scores ~84% accuracy.
- More complex models (XGBoost, Random Forest) don't automatically outperform simpler ones (Logistic Regression) — especially on smaller datasets like this one (~1,200 rows).
- Threshold tuning can meaningfully improve model usefulness without retraining — two models with similar ROC-AUC can have very different real-world performance depending on the decision threshold chosen.

## Tech Stack
Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), Streamlit

## Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```