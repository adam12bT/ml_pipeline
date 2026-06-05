"""
Space 2 — Clean & Engineer
Hugging Face Space: your-username/ai-pipeline-clean

Responsibilities:
  - Receive dataset JSON + schema from Render backend
  - Remove duplicates
  - Impute missing values
  - Fix data types
  - Normalize strings
  - Clip outliers
  - Encode categorical columns
  - Scale numeric columns
  - Extract datetime features
  - Select best features
  - Return cleaned + engineered dataset as JSON
"""

from __future__ import annotations

import io
import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder,
    RobustScaler, StandardScaler,
)
from sklearn.feature_selection import VarianceThreshold


app = FastAPI(title="AI Pipeline — Space 2: Clean & Engineer")




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
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class CleanRequest(BaseModel):
    """
    Render sends the raw dataset as a list of row dicts
    plus the schema output from Space 1.
    """
    data: list[dict[str, Any]]
    schema: dict[str, Any]
    target_col: str
    problem_type: str
    imputation_strategy: str = "median"   # mean | median | mode | knn


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "space": "clean-engineer"}


# ═══════════════════════════════════════════════════════════════════════════════
# Main endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/run")
def run_clean(req: CleanRequest):
    """
    Clean and engineer a dataset.
    Returns the processed feature matrix X and target y,
    plus a full cleaning + engineering report.
    """
    df = pd.DataFrame(req.data)
    schema_cols = req.schema.get("columns", {})
    target_col  = req.target_col
    problem_type = req.problem_type

    cleaning_ops = []
    rows_before  = len(df)
    cols_before  = len(df.columns)

    # ── 1. Drop constant / >95% null / ID columns ─────────────────────────
    df, cleaning_ops = _drop_degenerate(df, schema_cols, target_col, cleaning_ops)

    # ── 2. Remove duplicate rows ──────────────────────────────────────────
    df, cleaning_ops, dupes_removed = _remove_dupes(df, cleaning_ops)

    # ── 3. Fix data types ─────────────────────────────────────────────────
    df, cleaning_ops = _fix_types(df, schema_cols, cleaning_ops)

    # ── 4. Normalize strings ──────────────────────────────────────────────
    df, cleaning_ops = _normalize_strings(df, schema_cols, cleaning_ops)

    # ── 5. Impute missing values ──────────────────────────────────────────
    df, cleaning_ops, nulls_imputed = _impute(
        df, schema_cols, target_col, req.imputation_strategy, cleaning_ops
    )

    # ── 6. Clip outliers ──────────────────────────────────────────────────
    df, cleaning_ops, outliers_handled = _clip_outliers(
        df, schema_cols, target_col, cleaning_ops
    )

    # ── 7. Extract datetime features ──────────────────────────────────────
    df, eng_ops = _extract_datetime(df, schema_cols)

    # ── 8. Separate X and y ───────────────────────────────────────────────
    y  = df[target_col].copy() if target_col in df.columns else None
    X  = df.drop(columns=[target_col], errors="ignore")

    # ── 9. Drop ID-like columns from X ────────────────────────────────────
    X = _drop_ids(X, schema_cols)

    # ── 10. Encode categoricals ───────────────────────────────────────────
    X, encoding_map, enc_ops = _encode(X, schema_cols, problem_type, y)
    eng_ops.extend(enc_ops)

    # ── 11. Scale numerics ────────────────────────────────────────────────
    X, scaling_map = _scale(X, problem_type)

    # ── 12. Encode target ─────────────────────────────────────────────────
    y_encoded, target_classes = _encode_target(y, problem_type)

    # ── 13. Feature selection ─────────────────────────────────────────────
    X, selected_cols = _select_features(X, y_encoded)

    return {
        "X": X.to_dict(orient="records"),
        "y": y_encoded.tolist() if y_encoded is not None else [],
        "feature_names": list(X.columns),
        "target_classes": target_classes,
        "cleaning_report": {
            "rows_before":       rows_before,
            "rows_after":        len(df),
            "cols_before":       cols_before,
            "cols_after":        len(df.columns),
            "duplicates_removed": dupes_removed,
            "nulls_imputed":     nulls_imputed,
            "outliers_handled":  outliers_handled,
            "operations":        cleaning_ops,
        },
        "engineering_report": {
            "original_features":   cols_before - 1,
            "engineered_features": len(X.columns),
            "encoding_map":        encoding_map,
            "scaling_map":         scaling_map,
            "operations":          eng_ops,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaning steps
# ═══════════════════════════════════════════════════════════════════════════════

def _drop_degenerate(df, schema_cols, target_col, ops):
    to_drop = []
    for col, info in schema_cols.items():
        if col == target_col or col not in df.columns:
            continue
        if info.get("is_constant"):
            to_drop.append(col)
            ops.append({"op": "drop_constant", "column": col})
        elif info.get("null_pct", 0) > 95:
            to_drop.append(col)
            ops.append({"op": "drop_high_null", "column": col,
                        "detail": f"{info['null_pct']}% nulls"})
    if to_drop:
        df = df.drop(columns=to_drop)
    return df, ops


def _remove_dupes(df, ops):
    n = int(df.duplicated().sum())
    if n > 0:
        df = df.drop_duplicates()
        ops.append({"op": "remove_duplicates", "rows_removed": n})
    return df, ops, n


def _fix_types(df, schema_cols, ops):
    for col, info in schema_cols.items():
        if col not in df.columns:
            continue
        t = info.get("inferred_type","")
        if t == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t == "datetime":
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass
        elif t == "boolean":
            bmap = {"true":True,"yes":True,"1":True,"false":False,"no":False,"0":False}
            df[col] = df[col].astype(str).str.lower().map(bmap)
    return df, ops


def _normalize_strings(df, schema_cols, ops):
    for col, info in schema_cols.items():
        if col not in df.columns:
            continue
        if info.get("inferred_type") in ("categorical","text"):
            df[col] = (df[col].astype(str).str.strip().str.lower()
                       .replace({"nan":np.nan,"none":np.nan,"null":np.nan,"":np.nan}))
    return df, ops


def _impute(df, schema_cols, target_col, strategy, ops):
    total = 0

    # Drop rows where target is null
    if target_col in df.columns and df[target_col].isna().any():
        n = int(df[target_col].isna().sum())
        df = df.dropna(subset=[target_col])
        ops.append({"op": "drop_target_nulls", "column": target_col, "rows": n})

    num_cols = [c for c in df.columns if c != target_col
                and pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()]
    cat_cols = [c for c in df.columns if c != target_col
                and not pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()]

    if num_cols:
        nb = int(df[num_cols].isna().sum().sum())
        if strategy == "knn":
            imp = KNNImputer(n_neighbors=5)
        else:
            strat = {"mean":"mean","median":"median","mode":"most_frequent"}.get(strategy,"median")
            imp = SimpleImputer(strategy=strat)
        df[num_cols] = imp.fit_transform(df[num_cols])
        filled = nb - int(df[num_cols].isna().sum().sum())
        total += filled
        ops.append({"op": "impute_numeric", "strategy": strategy, "filled": filled})

    for col in cat_cols:
        n = int(df[col].isna().sum())
        mode = df[col].mode()
        fill = mode.iloc[0] if not mode.empty else "__MISSING__"
        df[col] = df[col].fillna(fill)
        total += n
        ops.append({"op": "impute_categorical", "column": col, "filled": n})

    return df, ops, total


def _clip_outliers(df, schema_cols, target_col, ops):
    total = 0
    for col, info in schema_cols.items():
        if col not in df.columns or col == target_col:
            continue
        if not info.get("has_outliers"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        n = int(((df[col] < q1)|(df[col] > q99)).sum())
        df[col] = df[col].clip(q1, q99)
        total += n
        ops.append({"op": "clip_outliers", "column": col, "clipped": n})
    return df, ops, total


# ═══════════════════════════════════════════════════════════════════════════════
# Engineering steps
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_datetime(df, schema_cols):
    ops = []
    dt_cols = [c for c,i in schema_cols.items()
               if c in df.columns and i.get("inferred_type") == "datetime"]
    for col in dt_cols:
        try:
            dt   = pd.to_datetime(df[col], errors="coerce")
            base = col.rstrip("_date").rstrip("_time").rstrip("_at")
            for suffix, vals in [
                ("year",dt.dt.year),("month",dt.dt.month),
                ("day",dt.dt.day),("dayofweek",dt.dt.dayofweek),
                ("quarter",dt.dt.quarter),
            ]:
                df[f"{base}_{suffix}"] = vals
                ops.append({"op":"datetime_extract","new_col":f"{base}_{suffix}"})
            df = df.drop(columns=[col])
        except Exception:
            pass
    return df, ops


def _drop_ids(X, schema_cols):
    to_drop = [c for c in X.columns
               if schema_cols.get(c,{}).get("is_id_like") or
               schema_cols.get(c,{}).get("inferred_type") in ("uuid","email","url","id")]
    return X.drop(columns=to_drop, errors="ignore")


def _encode(X, schema_cols, problem_type, y):
    encoding_map = {}
    ops = []
    cat_cols = [c for c in X.columns
                if X[c].dtype == "object" or str(X[c].dtype) == "category"]

    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique <= 1:
            X = X.drop(columns=[col])
            continue
        if n_unique <= 15:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
            enc = ohe.fit_transform(X[[col]].astype(str))
            new_cols = [f"{col}__{c}" for c in ohe.categories_[0][1:]]
            ohe_df = pd.DataFrame(enc, columns=new_cols, index=X.index)
            X = pd.concat([X.drop(columns=[col]), ohe_df], axis=1)
            encoding_map[col] = "onehot"
            ops.append({"op":"onehot_encode","column":col,"new_cols":len(new_cols)})
        elif n_unique <= 50:
            freq = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq).fillna(0.0)
            encoding_map[col] = "frequency"
            ops.append({"op":"frequency_encode","column":col})
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoding_map[col] = "label"
            ops.append({"op":"label_encode","column":col})

    return X, encoding_map, ops


def _scale(X, problem_type):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return X, {}
    scaler = RobustScaler() if problem_type == "regression" else StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    strat = "robust" if problem_type == "regression" else "standard"
    return X, {col: strat for col in num_cols}


def _encode_target(y, problem_type):
    if y is None:
        return None, []
    if problem_type in ("binary_classification","multiclass_classification"):
        if y.dtype == "object" or str(y.dtype) == "category":
            le = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
            return y_enc, list(le.classes_)
        return y.astype(int), []
    return pd.to_numeric(y, errors="coerce").fillna(y.median()), []


def _select_features(X, y_encoded):
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0)
    try:
        vt   = VarianceThreshold(threshold=0.01)
        Xarr = vt.fit_transform(X)
        cols = [X.columns[i] for i in range(len(X.columns)) if vt.get_support()[i]]
        X    = pd.DataFrame(Xarr, columns=cols, index=X.index)
    except Exception:
        pass
    if len(X.columns) > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        X = X.drop(columns=to_drop, errors="ignore")
    return X, list(X.columns)