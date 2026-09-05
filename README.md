# Telco Customer Churn: Predictive Modeling

## Overview
Customer churn is a major revenue drain for telecom providers, and identifying at-risk customers early enables targeted, cost-effective retention campaigns. 

This project builds and evaluates a churn prediction model, then translates its output into a business decision: which customers to target, and at what probability threshold, to maximize net retention value.

## Approach/Methods
- Data cleaning and feature encoding (binary + one-hot categorical encoding)
- Stratified train/validation/test split (60/20/20) to separate hyperparameter tuning, threshold tuning, and final evaluation
- Hyperparameter optimization via `GridSearchCV` (5-fold CV, ROC-AUC scoring) for Logistic Regression, Random Forest, and XGBoost
- Model comparison on the validation set using ROC-AUC, precision, recall, and F1, with a tie-breaking rule favoring simpler models when performance is statistically close
- Business-driven threshold optimization: customer lifetime value (LTV) and retention campaign cost estimated from the data, then used to compute net business value across decision thresholds
- Final evaluation on a held-out test set, plus 5-fold cross-validation stability check
- Model interpretation via logistic regression odds ratios and SHAP (TreeSHAP) for the tree-based models, including aggregation of one-hot encoded features back to their original variables

## Key Results
- **Selected model:** Logistic Regression (chosen over Random Forest and XGBoost, which performed comparably on ROC-AUC, on grounds of interpretability)
- **Test ROC-AUC:** 0.831 (CV mean: 0.843 ± 0.020, test performance is consistent with cross-validation, indicating a stable model)
- **Optimized decision threshold:** 0.10 (chosen to maximize net business value rather than the default 0.5)
- **Test set recall (Churn class):** 0.94, precision: 0.40, F1: 0.56 — the low threshold intentionally trades precision for catching the large majority of churners
- **Estimated net business value:** around $470K per deployment cycle on the validation set, based on estimated customer LTV (around $1,464) and retention campaign cost (around $59)
- **Top churn drivers:** contract type (month-to-month customers are highest risk), fiber optic internet service, absence of tech support/online security, and payment method (electronic check); tenure was a comparatively low-importance feature once contract type was accounted for

## Data Note
Uses the publicly available IBM/Kaggle Telco Customer Churn dataset (7,043 customer records, 21 features). 

## Tools Used
- Python
- pandas, numpy
- scikit-learn (Logistic Regression, Random Forest, GridSearchCV, pipelines, preprocessing, metrics)
- XGBoost
- SHAP
- matplotlib, seaborn

## Repository Contents
- `notebook/telco_churn_modeling.ipynb`: full pipeline - data prep, model tuning/comparison, threshold optimization, evaluation, and interpretation
- `data/Telco-Customer-Churn.csv`: raw dataset
