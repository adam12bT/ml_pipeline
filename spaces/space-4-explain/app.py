"""
Space 4 — Explain & Report
Hugging Face Space: your-username/ai-pipeline-explain

Responsibilities:
  - Receive model results + feature importances from Render
  - Generate SHAP-style importance explanation
  - Build full HTML report
  - Build full JSON report
  - Return both reports as strings
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


app = FastAPI(title="AI Pipeline — Space 4: Explain & Report")




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

class ReportRequest(BaseModel):
    run_id: str
    filename: str

    # From Space 1
    rows: int
    columns: int
    quality_score: float
    schema: dict[str, Any]
    target_detection: dict[str, Any]

    # From Space 2
    cleaning_report: dict[str, Any]
    engineering_report: dict[str, Any]

    # From Space 3
    model_result: dict[str, Any]
    feature_importances: dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "space": "explain-report"}


# ═══════════════════════════════════════════════════════════════════════════════
# Main endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/run")
def run_report(req: ReportRequest):
    """
    Generate HTML + JSON reports from all pipeline results.
    Returns both as strings.
    """
    importance_summary = _importance_summary(req.feature_importances)
    next_steps         = _next_steps(req)
    html               = _build_html(req, importance_summary, next_steps)
    report_json        = _build_json(req)

    return {
        "html_report":  html,
        "json_report":  report_json,
        "importance_summary": importance_summary,
        "next_steps": next_steps,
    }


@app.post("/report/html", response_class=HTMLResponse)
def get_html_report(req: ReportRequest):
    """Return the HTML report directly (renders in browser)."""
    importance_summary = _importance_summary(req.feature_importances)
    next_steps         = _next_steps(req)
    return HTMLResponse(content=_build_html(req, importance_summary, next_steps))


# ═══════════════════════════════════════════════════════════════════════════════
# Report builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_html(req: ReportRequest, importance_summary: str, next_steps: list) -> str:
    q = req.quality_score
    qcolor = "#22c55e" if q >= 80 else "#f59e0b" if q >= 60 else "#ef4444"

    td = req.target_detection
    cr = req.cleaning_report
    er = req.engineering_report
    mr = req.model_result
    imp = req.feature_importances
    schema_cols = req.schema.get("columns", {})

    # ── Column table ─────────────────────────────────────────────────────────
    col_rows = ""
    for col, info in schema_cols.items():
        null_color = "#ef4444" if info.get("null_pct",0) > 20 else "#22c55e"
        col_rows += f"""<tr>
          <td><code>{col}</code></td>
          <td>{info.get('inferred_type','?')}</td>
          <td style="color:{null_color}">{info.get('null_pct',0):.1f}%</td>
          <td>{info.get('unique_count',0):,}</td>
          <td>{'⚠️' if info.get('has_outliers') else '✅'}</td>
        </tr>"""

    # ── Model rows ────────────────────────────────────────────────────────────
    model_rows = ""
    for m in mr.get("models", []):
        is_best = m.get("model_name") == mr.get("best_model")
        score = m.get("roc_auc") or m.get("f1") or m.get("accuracy") or m.get("r2") or 0
        model_rows += f"""<tr {'style="background:#f0fdf4;font-weight:700"' if is_best else ''}>
          <td>{'🏆 ' if is_best else ''}{m.get('model_name','')}</td>
          <td>{score:.4f}</td>
          <td>{m.get('cv_score_mean',0):.4f} ± {m.get('cv_score_std',0):.4f}</td>
          <td>{m.get('training_time_seconds',0):.1f}s</td>
        </tr>"""

    # ── Feature importance bars ───────────────────────────────────────────────
    total_imp = sum(imp.values()) or 1.0
    imp_rows = ""
    for feat, score in list(imp.items())[:15]:
        pct = score / total_imp * 100
        imp_rows += f"""<tr>
          <td><code>{feat}</code></td>
          <td style="width:200px">
            <div style="background:#e0e7ff;border-radius:4px;overflow:hidden">
              <div style="background:#6366f1;height:14px;width:{min(pct*3,100):.0f}%"></div>
            </div>
          </td>
          <td>{pct:.1f}%</td>
        </tr>"""

    # ── Cleaning ops ──────────────────────────────────────────────────────────
    clean_ops_rows = ""
    for op in cr.get("operations", []):
        clean_ops_rows += f"""<tr>
          <td><code>{op.get('op','')}</code></td>
          <td>{op.get('column','-')}</td>
          <td style="font-size:12px;color:#64748b">{op.get('detail') or op.get('filled') or op.get('rows_removed','')}</td>
        </tr>"""

    # ── Next steps ────────────────────────────────────────────────────────────
    next_steps_html = "\n".join(f"<li>{s}</li>" for s in next_steps)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AI Pipeline Report — {req.filename}</title>
<style>
  :root{{--primary:#6366f1;--bg:#f8fafc;--card:#fff;--text:#1e293b;--muted:#64748b;--border:#e2e8f0}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
  header{{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;padding:36px 40px}}
  header h1{{font-size:26px;margin-bottom:6px}}
  header p{{opacity:.85;font-size:13px}}
  main{{max-width:1100px;margin:0 auto;padding:32px 16px}}
  .card{{background:var(--card);border-radius:12px;padding:28px;margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,.07)}}
  .card h2{{font-size:17px;color:var(--primary);margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid #e0e7ff}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;margin-bottom:20px}}
  .stat{{background:var(--bg);border-radius:10px;padding:16px;text-align:center}}
  .stat-val{{font-size:22px;font-weight:700}}
  .stat-lbl{{font-size:11px;color:var(--muted);margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}}
  th{{background:var(--bg);padding:10px 12px;text-align:left;font-weight:600;color:var(--muted)}}
  td{{padding:9px 12px;border-bottom:1px solid var(--border)}}
  tr:hover td{{background:var(--bg)}}
  code{{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:12px;font-family:monospace}}
  ul{{padding-left:20px}} li{{margin:7px 0}}
  footer{{text-align:center;padding:24px;color:var(--muted);font-size:12px}}
</style>
</head>
<body>
<header>
  <h1>🤖 AI Dataset Intelligence Report</h1>
  <p>File: <strong>{req.filename}</strong> &nbsp;|&nbsp; Run: <code style="background:rgba(255,255,255,.2);padding:2px 8px;border-radius:4px">{req.run_id[:8]}…</code>
     &nbsp;|&nbsp; {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
</header>
<main>

<div class="card">
  <h2>📊 Dataset Overview</h2>
  <div class="grid">
    <div class="stat"><div class="stat-val">{req.rows:,}</div><div class="stat-lbl">Rows</div></div>
    <div class="stat"><div class="stat-val">{req.columns}</div><div class="stat-lbl">Columns</div></div>
    <div class="stat"><div class="stat-val" style="color:{qcolor}">{q:.0f}/100</div><div class="stat-lbl">Quality Score</div></div>
    <div class="stat"><div class="stat-val">{req.schema.get('duplicate_pct',0):.1f}%</div><div class="stat-lbl">Duplicates</div></div>
    <div class="stat"><div class="stat-val">{req.schema.get('memory_mb',0):.1f} MB</div><div class="stat-lbl">Memory</div></div>
  </div>
  <table><thead><tr><th>Column</th><th>Type</th><th>Nulls</th><th>Unique</th><th>Outliers</th></tr></thead>
  <tbody>{col_rows}</tbody></table>
</div>

<div class="card">
  <h2>🎯 Target Detection</h2>
  <div class="grid">
    <div class="stat"><div class="stat-val" style="color:#6366f1">{td.get('predicted_target','?')}</div><div class="stat-lbl">Target Column</div></div>
    <div class="stat"><div class="stat-val">{td.get('problem_type','?').replace('_',' ').title()}</div><div class="stat-lbl">Task Type</div></div>
    <div class="stat"><div class="stat-val" style="color:{qcolor}">{td.get('confidence',0)*100:.0f}%</div><div class="stat-lbl">Confidence</div></div>
  </div>
  <p><strong>Reasoning:</strong> {td.get('reasoning','')}</p>
</div>

<div class="card">
  <h2>🧹 Data Cleaning</h2>
  <div class="grid">
    <div class="stat"><div class="stat-val">{cr.get('duplicates_removed',0):,}</div><div class="stat-lbl">Duplicates Removed</div></div>
    <div class="stat"><div class="stat-val">{cr.get('nulls_imputed',0):,}</div><div class="stat-lbl">Nulls Filled</div></div>
    <div class="stat"><div class="stat-val">{cr.get('outliers_handled',0):,}</div><div class="stat-lbl">Outliers Clipped</div></div>
    <div class="stat"><div class="stat-val">{cr.get('rows_before',0) - cr.get('rows_after',0):,}</div><div class="stat-lbl">Rows Removed</div></div>
  </div>
  <table><thead><tr><th>Operation</th><th>Column</th><th>Detail</th></tr></thead>
  <tbody>{clean_ops_rows}</tbody></table>
</div>

<div class="card">
  <h2>⚙️ Feature Engineering</h2>
  <div class="grid">
    <div class="stat"><div class="stat-val">{er.get('original_features',0)}</div><div class="stat-lbl">Original Features</div></div>
    <div class="stat"><div class="stat-val">{er.get('engineered_features',0)}</div><div class="stat-lbl">Final Features</div></div>
  </div>
</div>

<div class="card">
  <h2>🏋️ Model Results</h2>
  <p style="margin-bottom:14px">Best model: <strong style="color:#6366f1">{mr.get('best_model','')}</strong>
     — score <strong>{mr.get('best_score',0):.4f}</strong></p>
  <table><thead><tr><th>Model</th><th>Score</th><th>CV (mean ± std)</th><th>Time</th></tr></thead>
  <tbody>{model_rows}</tbody></table>
</div>

<div class="card">
  <h2>🔬 Feature Importance</h2>
  <table><thead><tr><th>Feature</th><th>Importance</th><th>%</th></tr></thead>
  <tbody>{imp_rows}</tbody></table>
</div>

<div class="card">
  <h2>💡 Recommended Next Steps</h2>
  <ul>{next_steps_html}</ul>
</div>

</main>
<footer>Generated by AI Dataset Intelligence Pipeline · {datetime.utcnow().year}</footer>
</body>
</html>"""


def _build_json(req: ReportRequest) -> dict:
    return {
        "run_id":          req.run_id,
        "filename":        req.filename,
        "generated_at":    datetime.utcnow().isoformat(),
        "quality_score":   req.quality_score,
        "dataset":         {"rows": req.rows, "columns": req.columns},
        "target_detection": req.target_detection,
        "cleaning_report": req.cleaning_report,
        "engineering_report": req.engineering_report,
        "model_result":    req.model_result,
        "feature_importances": req.feature_importances,
    }


def _importance_summary(importance: dict) -> str:
    if not importance:
        return "Feature importance not available."
    total = sum(importance.values()) or 1.0
    lines = ["Top features driving predictions:"]
    for i, (feat, score) in enumerate(list(importance.items())[:10], 1):
        pct = round(score / total * 100, 1)
        lines.append(f"  {i}. '{feat}' — {pct}%")
    return "\n".join(lines)


def _next_steps(req: ReportRequest) -> list:
    steps = []
    q = req.quality_score
    mr = req.model_result

    if q < 70:
        steps.append("⚠️ Dataset quality is below 70 — review data collection or add more data.")
    if req.target_detection.get("confidence", 1) < 0.7:
        steps.append("🎯 Target detection confidence is low — manually confirm your target column.")
    best_score = mr.get("best_score", 0)
    best_model = mr.get("best_model", "")
    if best_score < 0.7:
        steps.append("📈 Best model score is below 0.7 — try collecting more data or tuning hyperparameters.")
    else:
        steps.append(f"✅ {best_model} achieved {best_score:.2f} — ready for hyperparameter tuning.")
    steps += [
        "🔧 Run hyperparameter tuning (Optuna / GridSearchCV) on the best model.",
        "📊 Review feature importance — remove low-value columns to simplify the model.",
        "🚀 Export the best model and deploy it to a prediction API.",
        "🔁 Set up automated retraining when new data arrives.",
    ]
    return steps