"""
Space 1 — Ingest & Profile
Hugging Face Space: your-username/ai-pipeline-ingest

Responsibilities:
  - Receive uploaded file
  - Parse CSV / Excel / JSON / Parquet
  - Analyse schema (column types, stats, nulls)
  - Detect target column
  - Return structured JSON result
"""

from __future__ import annotations

import io
import json
import math
import re
import chardet
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from typing import Any, Optional


app = FastAPI(title="AI Pipeline — Space 1: Ingest & Profile")


@app.middleware("http")
async def add_cors(request, call_next):
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

# ── Regex patterns ────────────────────────────────────────────────────────────
_RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RE_URL   = re.compile(r"^https?://\S+$", re.I)
_RE_UUID  = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

# ── Target detection vocabulary ───────────────────────────────────────────────
POSITIVE_KEYWORDS = {
    "target":1.0,"label":1.0,"class":0.95,"outcome":0.9,"churn":0.95,
    "fraud":0.95,"risk":0.85,"default":0.9,"diagnosis":0.9,"status":0.75,
    "price":0.85,"revenue":0.85,"sales":0.8,"salary":0.8,"score":0.75,
    "survival":0.9,"approved":0.8,"converted":0.8,"purchased":0.8,"y":0.5,
    "survived":0.95,"left":0.8,"attrition":0.9,"result":0.75,"outcome":0.9,
}
NEGATIVE_KEYWORDS = {
    "id","uuid","guid","pk","key","created_at","updated_at","timestamp",
    "email","url","link","image","path","hash","token","index",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "space": "ingest-profile"}


# ═══════════════════════════════════════════════════════════════════════════════
# Main endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/run")
async def run_ingest(file: UploadFile = File(...)):
    """
    Accept a dataset file, parse it, profile it, detect target.
    Returns full schema + target detection result as JSON.
    """
    content = await file.read()
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower()

    try:
        df = _parse_file(content, ext, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    schema  = _analyse_schema(df)
    target  = _detect_target(df, schema)
    quality = _quality_score(schema)

    sample_df = df.head(5)
    result = {
        "rows":    len(df),
        "columns": len(df.columns),
        "schema":  schema,
        "target_detection": target,
        "quality_score": quality,
        "sample":  sample_df.where(sample_df.notna(), other=None).to_dict(orient="records"),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    return JSONResponse(content=_sanitize(result))


# ═══════════════════════════════════════════════════════════════════════════════
# NaN / Inf sanitizer
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize(obj):
    """Recursively replace nan/inf with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# File parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_file(content: bytes, ext: str, filename: str) -> pd.DataFrame:
    if ext == "csv":
        enc = chardet.detect(content[:50000]).get("encoding","utf-8") or "utf-8"
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1", on_bad_lines="skip")

    elif ext in ("xlsx","xls"):
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")

    elif ext == "json":
        parsed = json.loads(content)
        if isinstance(parsed, list):
            df = pd.json_normalize(parsed)
        else:
            for key in ("data","records","rows","items","results"):
                if key in parsed and isinstance(parsed[key], list):
                    df = pd.json_normalize(parsed[key])
                    break
            else:
                df = pd.json_normalize([parsed])

    elif ext == "parquet":
        df = pq.read_table(io.BytesIO(content)).to_pandas()

    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # Sanitise column names
    df.columns = [
        str(c).strip().lower().replace(" ","_").replace("-","_")
        for c in df.columns
    ]
    df = df.loc[:, ~df.columns.str.match(r"^unnamed")]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Schema analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _analyse_schema(df: pd.DataFrame) -> dict:
    columns = {}
    for col in df.columns:
        s = df[col]
        n = len(s)
        null_count = int(s.isna().sum())
        non_null = s.dropna()
        unique   = int(non_null.nunique())
        col_type = _infer_type(s)

        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "inferred_type": col_type,
            "null_count": null_count,
            "null_pct": round(null_count / max(n,1) * 100, 2),
            "unique_count": unique,
            "cardinality_ratio": round(unique / max(n,1), 4),
            "is_constant": unique <= 1,
            "is_id_like": _is_id(s, col_type, col),
            "has_outliers": _has_outliers(s, col_type),
        }

        if col_type == "numeric":
            num = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(num):
                info.update({
                    "mean": round(float(num.mean()), 4),
                    "std":  round(float(num.std()), 4),
                    "min":  round(float(num.min()), 4),
                    "max":  round(float(num.max()), 4),
                    "q25":  round(float(num.quantile(0.25)), 4),
                    "q75":  round(float(num.quantile(0.75)), 4),
                    "skewness": round(float(scipy_stats.skew(num)), 4),
                })
        elif col_type == "categorical":
            vc = non_null.value_counts()
            info["top_values"] = [(str(k), int(v)) for k,v in vc.head(5).items()]
            info["mode"] = str(vc.index[0]) if len(vc) else None

        columns[col] = info

    dup = int(df.duplicated().sum())
    return {
        "columns": columns,
        "duplicate_count": dup,
        "duplicate_pct": round(dup / max(len(df),1) * 100, 2),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def _infer_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):           return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    if pd.api.types.is_numeric_dtype(s):        return "numeric"
    sample = s.dropna().astype(str)
    if sample.empty: return "unknown"
    if sample.str.match(_RE_UUID).mean()  > 0.8: return "uuid"
    if sample.str.match(_RE_EMAIL).mean() > 0.8: return "email"
    if sample.str.match(_RE_URL).mean()   > 0.8: return "url"
    try:
        if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9:
            return "numeric"
    except Exception:
        pass
    lower = set(sample.str.lower().unique())
    if lower.issubset({"true","false","yes","no","1","0","t","f","y","n"}):
        return "boolean"
    if sample.str.len().mean() > 50: return "text"
    return "categorical"


def _is_id(s: pd.Series, col_type: str, name: str) -> bool:
    if col_type in ("uuid","email","url"): return True
    if s.nunique() / max(len(s),1) > 0.95:
        if any(kw in name.lower() for kw in NEGATIVE_KEYWORDS): return True
    return False


def _has_outliers(s: pd.Series, col_type: str) -> bool:
    if col_type != "numeric": return False
    num = pd.to_numeric(s, errors="coerce").dropna()
    if len(num) < 10: return False
    q1,q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: return False
    return bool(((num < q1-3*iqr)|(num > q3+3*iqr)).mean() > 0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Target detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_target(df: pd.DataFrame, schema: dict) -> dict:
    cols   = list(df.columns)
    n_cols = len(cols)
    scores = {}

    for i, col in enumerate(cols):
        info = schema["columns"].get(col, {})
        if info.get("is_id_like") or info.get("is_constant"):
            continue
        if info.get("null_pct", 100) > 90:
            continue

        h = _heuristic(col, info)
        s = _statistical(col, df, cols)
        p = _positional(i, n_cols)
        scores[col] = {"h":h,"s":s,"p":p,"final": 0.45*h + 0.40*s + 0.15*p}

    if not scores:
        return {"predicted_target": cols[-1], "confidence": 0.1,
                "problem_type": "unknown", "reasoning": "Fallback to last column."}

    ranked = sorted(scores.items(), key=lambda x: x[1]["final"], reverse=True)
    best, sc = ranked[0]
    alts = [c for c,_ in ranked[1:4]]
    problem_type = _problem_type(df[best], schema["columns"].get(best,{}))

    gap = sc["final"] - (ranked[1][1]["final"] if len(ranked)>1 else 0)
    confidence = round(float(np.clip(sc["final"] * (0.7 + 0.3*min(1.0,gap*5)), 0, 0.99)), 3)

    return {
        "predicted_target": best,
        "confidence": confidence,
        "problem_type": problem_type,
        "alternative_targets": alts,
        "scores": {k: round(v,3) for k,v in sc.items()},
        "reasoning": _reasoning(best, sc, problem_type),
    }


def _heuristic(col: str, info: dict) -> float:
    name = col.lower()
    score = 0.0
    for kw, w in POSITIVE_KEYWORDS.items():
        if kw in name: score = max(score, w)
    for kw in NEGATIVE_KEYWORDS:
        if kw in name: return 0.0
    u = info.get("unique_count", 0)
    if u == 2:       score += 0.15
    elif 3<=u<=20:   score += 0.10
    elif u == 1:     score -= 0.5
    return float(np.clip(score, 0, 1))


def _statistical(col: str, df: pd.DataFrame, all_cols: list) -> float:
    try:
        feat_cols = [c for c in all_cols if c != col][:15]
        parts = []
        for fc in feat_cols:
            col_data = df[fc].copy()
            if col_data.dtype == "object":
                col_data = LabelEncoder().fit_transform(col_data.astype(str).fillna("NA"))
            else:
                col_data = pd.to_numeric(col_data, errors="coerce").fillna(0).values
            parts.append(np.array(col_data).reshape(-1,1))
        if not parts: return 0.5
        X = np.hstack(parts)
        y = df[col]
        n_unique = y.nunique()
        if n_unique > 20:
            y_enc = pd.to_numeric(y, errors="coerce").fillna(0).values
            mi = mutual_info_regression(X, y_enc, random_state=42)
        else:
            y_enc = LabelEncoder().fit_transform(y.astype(str).fillna("NA"))
            mi = mutual_info_classif(X, y_enc, random_state=42)
        mi_norm = mi / (mi.max()+1e-9)
        return float(np.clip(np.mean(mi_norm)*2, 0, 1))
    except Exception:
        return 0.0


def _positional(i: int, n: int) -> float:
    if n <= 1: return 0.5
    return float(np.clip(math.exp(-3*(1-i/(n-1))**2), 0, 1))


def _problem_type(series: pd.Series, info: dict) -> str:
    n_unique = info.get("unique_count", series.nunique())
    col_type = info.get("inferred_type","")
    if col_type == "datetime":              return "time_series"
    if col_type == "numeric" and n_unique > 20: return "regression"
    if n_unique == 2:                       return "binary_classification"
    if 3 <= n_unique <= 50:                 return "multiclass_classification"
    if col_type == "numeric":               return "regression"
    return "binary_classification"


def _reasoning(col: str, sc: dict, pt: str) -> str:
    parts = [f"'{col}' selected as target."]
    if sc.get("h",0) > 0.6: parts.append("Strong keyword match.")
    if sc.get("s",0) > 0.5: parts.append("High mutual information with features.")
    if sc.get("p",0) > 0.6: parts.append("Column near dataset end.")
    parts.append(f"Task: {pt}.")
    return " ".join(parts)


def _quality_score(schema: dict) -> float:
    cols = schema.get("columns",{})
    if not cols: return 0.0
    stats = list(cols.values())
    completeness    = 1 - np.mean([s.get("null_pct",0)/100 for s in stats])
    uniqueness      = 1 - schema.get("duplicate_pct",0)/100
    bad             = sum(1 for s in stats if s.get("is_constant") or s.get("is_id_like"))
    informativeness = 1 - bad/max(len(stats),1)
    return round(float((completeness*0.45 + uniqueness*0.30 + informativeness*0.25)*100), 1)