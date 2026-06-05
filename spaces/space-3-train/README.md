---
title: AI Pipeline Train Evaluate
emoji: 🏋️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Space 3 — Train & Evaluate

Part of the **AI Dataset Intelligence Pipeline**.

## What it does
- Trains Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Runs cross-validation on each model
- Evaluates on a held-out test set
- Returns ranked model metrics (accuracy, F1, ROC-AUC, RMSE, R²)
- Returns feature importances from the best model

## Endpoint
`POST /run` — send cleaned X + y, get back ranked model results
