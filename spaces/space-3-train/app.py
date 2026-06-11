"""
Space 3 — Train & Evaluate
Hugging Face Space: your-username/ai-pipeline-train
Responsibilities:
  - Receive cleaned X and y from Render backend
  - Train multiple ML models (LogReg, RF, XGBoost, LightGBM, CatBoost)
  - Cross-validate each model
  - Evaluate on held-out test set
  - Return ranked model results + best model predictions
"""


from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)


app = FastAPI(title="AI Pipeline — Space 3: Train & Evaluate")


@app.middleware("http")
async def add_cors(request, call_next):
    from fastapi.responses import JSONResponse
    if request.method == "OPTIONS":
        return JSONResponse(content={}, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        })
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# Request model
# ═══════════════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    X: list[dict[str, Any]]
    y: list[Any]
    feature_names: list[str]
    problem_type: str
    target_classes: list[str] = []
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize(obj: Any) -> Any:
    """Recursively replace nan/inf with None so json.dumps never chokes."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "space": "train-evaluate"}


# ═══════════════════════════════════════════════════════════════════════════════
# Main endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/run")
def run_train(req: TrainRequest):
    """
    Train multiple models, evaluate, and return ranked results
    plus the test set predictions from the best model.
    """
    X = pd.DataFrame(req.X, columns=req.feature_names)
    y = np.array(req.y)

    # Replace any inf/nan that slipped through
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    is_clf = req.problem_type in ("binary_classification", "multiclass_classification")

    # FIX 1: Re-encode class labels to a compact 0..N-1 range.
    # XGBoost requires labels to be exactly [0, 1, ..., n_classes-1].
    # A stratified split may drop a rare class from a fold, producing gaps
    # like [0, 1, 2, 4], which XGBoost rejects.  LabelEncoder guarantees
    # contiguous integers regardless of what values arrive.
    le: Optional[LabelEncoder] = None
    if is_clf:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=req.test_size,
            random_state=req.random_state,
            stratify=y if is_clf else None,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state
        )

    # Get models for this task
    models = _get_models(req.problem_type, req.random_state)
    all_metrics: list[dict] = []
    best_model_obj = None
    best_score = -np.inf

    for name, model in models.items():
        metrics = _train_evaluate(
            name, model, X_train, X_test, y_train, y_test,
            req.problem_type, req.cv_folds, req.random_state
        )
        all_metrics.append(metrics)
        score = _primary_score(metrics, req.problem_type)
        if score > best_score:
            best_score = score
            best_model_obj = model

    # Sort by primary score descending
    all_metrics.sort(
        key=lambda m: _primary_score(m, req.problem_type), reverse=True
    )
    best_name = all_metrics[0]["model_name"]

    # Get predictions + feature importances from best model
    y_pred = best_model_obj.predict(X_test).tolist()

    # Decode labels back to original values for the caller
    if le is not None:
        y_pred   = le.inverse_transform(np.array(y_pred, dtype=int)).tolist()
        y_test   = le.inverse_transform(y_test.astype(int)).tolist()
    else:
        y_test = y_test.tolist()

    feature_importances = _get_importances(best_model_obj, req.feature_names)

    result = {
        "best_model": best_name,
        "best_score": round(best_score, 4),
        "models": all_metrics,
        "feature_importances": feature_importances,
        "test_predictions": y_pred,
        "test_actuals": y_test,
        "problem_type": req.problem_type,
        "target_classes": req.target_classes,
    }

    # FIX 2: Scrub any nan / inf that survived into the response dict.
    # This prevents the "Out of range float values are not JSON compliant: nan"
    # crash that occurs when a CV fold fails and sklearn returns nan for that
    # fold's score, which then propagates into cv_mean / cv_std.
    return JSONResponse(content=_sanitize(result))


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

def _get_models(problem_type: str, random_state: int) -> dict:
    models = {}

    if problem_type in ("binary_classification", "multiclass_classification"):
        models["logistic_regression"] = LogisticRegression(
            max_iter=500, random_state=random_state, n_jobs=-1
        )
        models["random_forest"] = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        try:
            from xgboost import XGBClassifier
            models["xgboost"] = XGBClassifier(
                n_estimators=100, learning_rate=0.1,
                random_state=random_state, n_jobs=-1,
                eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            pass
        try:
            from lightgbm import LGBMClassifier
            models["lightgbm"] = LGBMClassifier(
                n_estimators=100, learning_rate=0.1,
                random_state=random_state, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            pass
        try:
            from catboost import CatBoostClassifier
            models["catboost"] = CatBoostClassifier(
                iterations=100, learning_rate=0.1,
                random_seed=random_state, verbose=0,
            )
        except ImportError:
            pass

    elif problem_type == "regression":
        models["linear_regression"] = LinearRegression(n_jobs=-1)
        models["random_forest"] = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        try:
            from xgboost import XGBRegressor
            models["xgboost"] = XGBRegressor(
                n_estimators=100, learning_rate=0.1,
                random_state=random_state, n_jobs=-1, verbosity=0,
            )
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor
            models["lightgbm"] = LGBMRegressor(
                n_estimators=100, learning_rate=0.1,
                random_state=random_state, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            pass
        try:
            from catboost import CatBoostRegressor
            models["catboost"] = CatBoostRegressor(
                iterations=100, learning_rate=0.1,
                random_seed=random_state, verbose=0,
            )
        except ImportError:
            pass

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# Training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _train_evaluate(
    name, model, X_train, X_test, y_train, y_test,
    problem_type, cv_folds, random_state
) -> dict:
    start = time.time()
    is_clf = problem_type in ("binary_classification", "multiclass_classification")

    cv_metric = (
        "roc_auc"     if problem_type == "binary_classification"
        else "f1_weighted" if problem_type == "multiclass_classification"
        else "r2"
    )

    try:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds,
            scoring=cv_metric, n_jobs=-1,
        )
        # FIX 2 (partial): drop any nan folds produced by failed fits before
        # computing mean/std so we never store nan in the metrics dict.
        valid_scores = cv_scores[~np.isnan(cv_scores)]
        cv_mean = float(np.mean(valid_scores)) if len(valid_scores) else 0.0
        cv_std  = float(np.std(valid_scores))  if len(valid_scores) else 0.0
    except Exception:
        cv_mean, cv_std = 0.0, 0.0

    # Final fit on full training split
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return {
            "model_name": name,
            "error": str(e),
            "cv_score_mean": cv_mean,
            "cv_score_std": cv_std,
            "training_time_seconds": round(time.time() - start, 2),
        }

    elapsed = round(time.time() - start, 2)
    y_pred  = model.predict(X_test)

    metrics: dict[str, Any] = {
        "model_name": name,
        "training_time_seconds": elapsed,
        "cv_score_mean": round(cv_mean, 4),
        "cv_score_std":  round(cv_std, 4),
    }

    if is_clf:
        avg = "binary" if problem_type == "binary_classification" else "weighted"
        metrics["accuracy"]  = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["precision"] = round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)
        metrics["recall"]    = round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4)
        metrics["f1"]        = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if problem_type == "binary_classification":
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(
                        roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                    ), 4)
        except Exception:
            pass
    else:
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["mae"]  = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["r2"]   = round(float(r2_score(y_test, y_pred)), 4)

    return metrics


def _primary_score(metrics: dict, problem_type: str) -> float:
    if problem_type in ("binary_classification", "multiclass_classification"):
        return metrics.get("roc_auc") or metrics.get("f1") or metrics.get("accuracy") or 0.0
    return metrics.get("r2") or 0.0


def _get_importances(model, feature_names: list) -> dict:
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).flatten()[:len(feature_names)]
        else:
            return {}
        result = dict(zip(feature_names, imp.tolist()))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}