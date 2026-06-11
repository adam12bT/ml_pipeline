"""
Render Backend — FastAPI Orchestrator
Fixed CORS middleware for all FastAPI versions
+ Space pause/resume to stay within HF free-tier CPU quota
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
import httpx
from datetime import datetime
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AI Pipeline — Orchestrator")

@app.middleware("http")
async def add_cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ── Environment variables ─────────────────────────────────────────────────────
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = os.environ.get("SUPABASE_ANON_KEY", "")
SPACE_1_URL   = os.environ.get("SPACE_1_URL", "http://localhost:7861")
SPACE_2_URL   = os.environ.get("SPACE_2_URL", "http://localhost:7862")
SPACE_3_URL   = os.environ.get("SPACE_3_URL", "http://localhost:7863")
SPACE_4_URL   = os.environ.get("SPACE_4_URL", "http://localhost:7864")
MAX_FILE_MB   = int(os.environ.get("MAX_FILE_MB", "200"))
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

# HF username + space names (must match your HF repo slugs exactly)
HF_USER       = os.environ.get("HF_USER", "your-username")
SPACE_1_NAME  = os.environ.get("SPACE_1_NAME", "ai-pipeline-ingest")
SPACE_2_NAME  = os.environ.get("SPACE_2_NAME", "ai-pipeline-features")
SPACE_3_NAME  = os.environ.get("SPACE_3_NAME", "ai-pipeline-train")
SPACE_4_NAME  = os.environ.get("SPACE_4_NAME", "ai-pipeline-report")

# How long (seconds) to wait after resuming before hitting /run
# Increase if your Spaces are slow to cold-start
SPACE_BOOT_WAIT = int(os.environ.get("SPACE_BOOT_WAIT", "20"))

# ── Supabase client ───────────────────────────────────────────────────────────
supabase = None
if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"⚠️ Supabase connection failed: {e}")
else:
    print("ℹ️ Running without Supabase — using in-memory storage")

# ── In-memory store ───────────────────────────────────────────────────────────
_MEMORY_STORE: dict[str, dict] = {}

http_headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
TIMEOUT = httpx.Timeout(300.0)

STATUS_PENDING   = "pending"
STATUS_INGESTING = "ingesting"
STATUS_CLEANING  = "cleaning"
STATUS_TRAINING  = "training"
STATUS_REPORTING = "reporting"
STATUS_COMPLETED = "completed"
STATUS_FAILED    = "failed"

PROGRESS = {
    STATUS_PENDING:   5,
    STATUS_INGESTING: 20,
    STATUS_CLEANING:  45,
    STATUS_TRAINING:  75,
    STATUS_REPORTING: 92,
    STATUS_COMPLETED: 100,
    STATUS_FAILED:    0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# NaN sanitizer — must run before ANY json= call
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
# Space pause / resume helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hf_set_space_state(space_name: str, action: str) -> None:
    """
    Call the HF API to pause or restart a Space.
    action must be 'pause' or 'restart'.
    Silently skips if HF_TOKEN is not set (local dev mode).
    """
    if not HF_TOKEN:
        print(f"  ⚠️  HF_TOKEN not set — skipping {action} for {space_name}")
        return
    url = f"https://huggingface.co/api/spaces/{HF_USER}/{space_name}/{action}"
    try:
        r = httpx.post(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=30)
        if r.status_code in (200, 204):
            icon = "⏸️" if action == "pause" else "▶️"
            print(f"  {icon}  {action.capitalize()}d Space: {space_name}")
        else:
            # Non-fatal — log and continue so the pipeline doesn't break
            print(f"  ⚠️  {action} {space_name} → HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"  ⚠️  {action} {space_name} failed (non-fatal): {e}")


def _pause_space(space_name: str) -> None:
    _hf_set_space_state(space_name, "pause")


def _resume_space(space_name: str) -> None:
    """Resume a Space and wait for it to boot before returning."""
    _hf_set_space_state(space_name, "restart")
    if HF_TOKEN:
        print(f"  ⏳  Waiting {SPACE_BOOT_WAIT}s for {space_name} to boot…")
        time.sleep(SPACE_BOOT_WAIT)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase": "connected" if supabase else "not configured (using memory)",
        "spaces": {
            "space_1": SPACE_1_URL,
            "space_2": SPACE_2_URL,
            "space_3": SPACE_3_URL,
            "space_4": SPACE_4_URL,
        }
    }


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
):
    allowed = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    ext = "." + (file.filename or "f").rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not supported.")

    content = await file.read()
    size_mb = len(content) / 1e6
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Limit: {MAX_FILE_MB} MB.")

    run_id = str(uuid.uuid4())
    _db_insert(run_id, file.filename or "upload", len(content))

    background_tasks.add_task(
        _run_pipeline, run_id, content,
        file.filename or "upload", ext, target_column
    )

    return {
        "run_id":  run_id,
        "status":  STATUS_PENDING,
        "message": f"Pipeline started. Poll /status/{run_id} for progress.",
    }


@app.get("/status/{run_id}")
def get_status(run_id: str):
    row = _db_get(run_id)
    if not row:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    status = row.get("status", STATUS_PENDING)
    return {
        "run_id":        run_id,
        "status":        status,
        "progress_pct":  PROGRESS.get(status, 0),
        "current_stage": _stage_label(status),
        "error":         row.get("error_message"),
        "result":        row.get("result_json") if status == STATUS_COMPLETED else None,
        "created_at":    row.get("created_at"),
        "filename":      row.get("filename"),
    }


@app.get("/result/{run_id}")
def get_result(run_id: str):
    row = _db_get(run_id)
    if not row:
        raise HTTPException(404, "Run not found.")
    if row.get("status") != STATUS_COMPLETED:
        raise HTTPException(409, "Pipeline not finished yet.")
    return row.get("result_json", {})


@app.get("/report/{run_id}/html")
def get_html_report(run_id: str):
    row = _db_get(run_id)
    if not row:
        raise HTTPException(404, "Run not found.")
    result = row.get("result_json", {})
    html = result.get("html_report", "<p>Report not available.</p>")
    return HTMLResponse(content=html)


@app.get("/runs")
def list_runs():
    if supabase:
        try:
            resp = supabase.table("pipeline_runs")\
                .select("run_id,filename,status,quality_score,created_at")\
                .order("created_at", desc=True).limit(50).execute()
            return resp.data or []
        except Exception:
            pass
    runs = sorted(_MEMORY_STORE.values(), key=lambda r: r.get("created_at",""), reverse=True)
    return [{"run_id":r["run_id"],"filename":r["filename"],
             "status":r["status"],"quality_score":r.get("quality_score"),
             "created_at":r.get("created_at")} for r in runs]


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline  (one Space running at a time)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    run_id: str,
    content: bytes,
    filename: str,
    ext: str,
    target_override: Optional[str],
) -> None:
    try:
        with httpx.Client(timeout=TIMEOUT, headers=http_headers) as client:

            # ── Space 1 ──────────────────────────────────────────────────────
            _db_update(run_id, STATUS_INGESTING)
            _resume_space(SPACE_1_NAME)
            print(f"\n[{run_id[:8]}] 📂 Calling Space 1…")
            resp1 = client.post(
                f"{SPACE_1_URL}/run",
                files={"file": (filename, content, _mime(ext))},
            )
            _check(resp1, "Space 1")
            s1 = resp1.json()
            print(f"[{run_id[:8]}] ✅ Space 1 done — {s1['rows']} rows, "
                  f"target='{s1['target_detection']['predicted_target']}'")
            _pause_space(SPACE_1_NAME)

            if target_override:
                s1["target_detection"]["predicted_target"] = target_override

            target_col   = s1["target_detection"]["predicted_target"]
            problem_type = s1["target_detection"]["problem_type"]

            # Parse full file (CPU work on Render, no Space needed)
            full_data = _parse_full_file(content, ext, filename)
            print(f"[{run_id[:8]}] 📊 Full dataset: {len(full_data)} rows")

            # ── Space 2 ──────────────────────────────────────────────────────
            _db_update(run_id, STATUS_CLEANING)
            _resume_space(SPACE_2_NAME)
            print(f"[{run_id[:8]}] 🧹 Calling Space 2…")
            resp2 = client.post(
                f"{SPACE_2_URL}/run",
                json=_sanitize({
                    "data":         full_data,
                    "schema":       s1["schema"],
                    "target_col":   target_col,
                    "problem_type": problem_type,
                }),
            )
            _check(resp2, "Space 2")
            s2 = resp2.json()
            print(f"[{run_id[:8]}] ✅ Space 2 done — "
                  f"{s2['engineering_report']['engineered_features']} features, "
                  f"{len(s2['X'])} rows")
            _pause_space(SPACE_2_NAME)

            # ── Space 3 ──────────────────────────────────────────────────────
            _db_update(run_id, STATUS_TRAINING)
            _resume_space(SPACE_3_NAME)
            print(f"[{run_id[:8]}] 🏋️ Calling Space 3…")
            resp3 = client.post(
                f"{SPACE_3_URL}/run",
                json=_sanitize({
                    "X":              s2["X"],
                    "y":              s2["y"],
                    "feature_names":  s2["feature_names"],
                    "problem_type":   problem_type,
                    "target_classes": s2.get("target_classes", []),
                }),
            )
            _check(resp3, "Space 3")
            s3 = resp3.json()
            print(f"[{run_id[:8]}] ✅ Space 3 done — "
                  f"best={s3['best_model']} score={s3['best_score']}")
            _pause_space(SPACE_3_NAME)

            # ── Space 4 ──────────────────────────────────────────────────────
            _db_update(run_id, STATUS_REPORTING)
            _resume_space(SPACE_4_NAME)
            print(f"[{run_id[:8]}] 📄 Calling Space 4…")
            resp4 = client.post(
                f"{SPACE_4_URL}/run",
                json=_sanitize({
                    "run_id":              run_id,
                    "filename":            filename,
                    "rows":                s1["rows"],
                    "columns":             s1["columns"],
                    "quality_score":       s1["quality_score"],
                    "schema":              s1["schema"],
                    "target_detection":    s1["target_detection"],
                    "cleaning_report":     s2["cleaning_report"],
                    "engineering_report":  s2["engineering_report"],
                    "model_result":        s3,
                    "feature_importances": s3.get("feature_importances", {}),
                }),
            )
            _check(resp4, "Space 4")
            s4 = resp4.json()
            print(f"[{run_id[:8]}] ✅ Space 4 done")
            _pause_space(SPACE_4_NAME)

            # ── Done ─────────────────────────────────────────────────────────
            full_result = _sanitize({
                "ingestion":   s1,
                "cleaning":    s2,
                "training":    s3,
                "reporting":   s4,
                "html_report": s4.get("html_report"),
                "json_report": s4.get("json_report"),
            })
            _db_complete(run_id, full_result, s1.get("quality_score", 0))
            print(f"[{run_id[:8]}] 🎉 COMPLETED\n")

    except Exception as exc:
        print(f"[{run_id[:8]}] ❌ FAILED: {exc}")
        _db_fail(run_id, str(exc))
        # Best-effort: pause whichever Space might still be running
        for name in (SPACE_1_NAME, SPACE_2_NAME, SPACE_3_NAME, SPACE_4_NAME):
            _pause_space(name)


def _parse_full_file(content: bytes, ext: str, filename: str) -> list[dict]:
    import io, json
    import pandas as pd
    import chardet

    try:
        if ext == ".csv":
            enc = chardet.detect(content[:50000]).get("encoding", "utf-8") or "utf-8"
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(io.BytesIO(content), encoding="latin1", on_bad_lines="skip")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        elif ext == ".json":
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
        elif ext == ".parquet":
            import pyarrow.parquet as pq
            df = pq.read_table(io.BytesIO(content)).to_pandas()
        else:
            return []

        df.columns = [
            str(c).strip().lower().replace(" ","_").replace("-","_")
            for c in df.columns
        ]
        df = df.loc[:, ~df.columns.str.match(r"^unnamed")]
        records = df.where(df.notna(), other=None).to_dict(orient="records")
        return _sanitize(records)

    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════════════

def _db_insert(run_id, filename, size_bytes):
    row = {
        "run_id": run_id, "filename": filename,
        "size_bytes": size_bytes, "status": STATUS_PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    _MEMORY_STORE[run_id] = row
    if supabase:
        try: supabase.table("pipeline_runs").insert(row).execute()
        except Exception as e: print(f"DB insert error: {e}")

def _db_update(run_id, status):
    if run_id in _MEMORY_STORE:
        _MEMORY_STORE[run_id]["status"] = status
        _MEMORY_STORE[run_id]["updated_at"] = datetime.utcnow().isoformat()
    if supabase:
        try:
            supabase.table("pipeline_runs").update({
                "status": status, "updated_at": datetime.utcnow().isoformat()
            }).eq("run_id", run_id).execute()
        except Exception as e: print(f"DB update error: {e}")

def _db_complete(run_id, result, quality):
    if run_id in _MEMORY_STORE:
        _MEMORY_STORE[run_id].update({
            "status": STATUS_COMPLETED, "result_json": result,
            "quality_score": quality,
            "completed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        })
    if supabase:
        try:
            supabase.table("pipeline_runs").update({
                "status": STATUS_COMPLETED, "result_json": result,
                "quality_score": quality,
                "completed_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("run_id", run_id).execute()
        except Exception as e: print(f"DB complete error: {e}")

def _db_fail(run_id, error):
    if run_id in _MEMORY_STORE:
        _MEMORY_STORE[run_id].update({
            "status": STATUS_FAILED,
            "error_message": error[:500],
            "updated_at": datetime.utcnow().isoformat(),
        })
    if supabase:
        try:
            supabase.table("pipeline_runs").update({
                "status": STATUS_FAILED, "error_message": error[:500],
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("run_id", run_id).execute()
        except Exception as e: print(f"DB fail error: {e}")

def _db_get(run_id):
    if supabase:
        try:
            resp = supabase.table("pipeline_runs")\
                .select("*").eq("run_id", run_id).single().execute()
            if resp.data: return resp.data
        except Exception: pass
    return _MEMORY_STORE.get(run_id)

def _check(resp, label):
    if resp.status_code != 200:
        raise RuntimeError(f"{label} returned {resp.status_code}: {resp.text[:300]}")

def _mime(ext):
    return {
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".json": "application/json",
        ".parquet": "application/octet-stream",
    }.get(ext, "application/octet-stream")

def _stage_label(status):
    return {
        STATUS_PENDING:   "Waiting to start…",
        STATUS_INGESTING: "Reading & profiling file… (Space 1)",
        STATUS_CLEANING:  "Cleaning & engineering features… (Space 2)",
        STATUS_TRAINING:  "Training ML models… (Space 3)",
        STATUS_REPORTING: "Generating report… (Space 4)",
        STATUS_COMPLETED: "Done! ✅",
        STATUS_FAILED:    "Pipeline failed ❌",
    }.get(status, status)