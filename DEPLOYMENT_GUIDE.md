# 🚀 AI Pipeline — Full Deployment Guide

Deploy everything for **free** using:
- 🤗 Hugging Face Spaces (4 ML engine spaces)
- ⚡ Render (FastAPI backend)
- 🗄️ Supabase (PostgreSQL database)
- 🎨 Vercel (React frontend)

---

## Folder Structure

```
ai-pipeline-deploy/
├── spaces/
│   ├── space-1-ingest/     → HF Space: your-name/ai-pipeline-ingest
│   ├── space-2-clean/      → HF Space: your-name/ai-pipeline-clean
│   ├── space-3-train/      → HF Space: your-name/ai-pipeline-train
│   └── space-4-explain/    → HF Space: your-name/ai-pipeline-explain
├── backend/                → Render web service
├── frontend/               → Vercel static site
└── database/               → Supabase SQL migration
```

---

## Step 1 — Supabase (Database) ~5 minutes

1. Go to **supabase.com** → create a free account
2. Click **"New Project"** → give it a name → set a password → click Create
3. Wait ~2 minutes for it to start
4. Go to **SQL Editor** → click **"New Query"**
5. Copy the contents of `database/migration.sql` and paste it → click **Run**
6. Go to **Project Settings → API**
7. Copy:
   - **Project URL** → save as `SUPABASE_URL`
   - **anon/public key** → save as `SUPABASE_ANON_KEY`

---

## Step 2 — Hugging Face Spaces (4 spaces) ~20 minutes

### Create account
1. Go to **huggingface.co** → Sign up (free)

### Deploy each Space (repeat 4 times)

#### Space 1 — Ingest & Profile
1. Click your profile → **New Space**
2. Name: `ai-pipeline-ingest`
3. License: MIT
4. SDK: **Docker**
5. Click **Create Space**
6. Upload ALL files from `spaces/space-1-ingest/`:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `Dockerfile`
7. Wait for it to build (2-3 minutes — watch the build logs)
8. Once it shows **"Running"**, copy the URL:
   `https://your-username-ai-pipeline-ingest.hf.space`

#### Space 2 — Clean & Engineer
Repeat above with files from `spaces/space-2-clean/`
Name: `ai-pipeline-clean`

#### Space 3 — Train & Evaluate
Repeat above with files from `spaces/space-3-train/`
Name: `ai-pipeline-train`
⚠️ This one installs XGBoost/LightGBM/CatBoost so build takes ~5 minutes

#### Space 4 — Explain & Report
Repeat above with files from `spaces/space-4-explain/`
Name: `ai-pipeline-explain`

### Test each Space
Visit each space URL + `/health` e.g.:
`https://your-username-ai-pipeline-ingest.hf.space/health`
Should return: `{"status":"ok",...}`

---

## Step 3 — Render (Backend) ~10 minutes

1. Go to **render.com** → create a free account
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Create a new GitHub repo called `ai-pipeline-backend`
5. Upload all files from `backend/` to that repo:
   - `main.py`
   - `requirements.txt`
   - `render.yaml`
6. In Render → select that repo → click **Connect**
7. Settings:
   - **Name**: `ai-pipeline-backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
8. Click **"Environment"** tab → add these variables:

| Key | Value |
|-----|-------|
| `SUPABASE_URL` | your Supabase project URL |
| `SUPABASE_ANON_KEY` | your Supabase anon key |
| `SPACE_1_URL` | `https://your-username-ai-pipeline-ingest.hf.space` |
| `SPACE_2_URL` | `https://your-username-ai-pipeline-clean.hf.space` |
| `SPACE_3_URL` | `https://your-username-ai-pipeline-train.hf.space` |
| `SPACE_4_URL` | `https://your-username-ai-pipeline-explain.hf.space` |
| `MAX_FILE_MB` | `200` |

9. Click **Deploy**
10. Wait ~3 minutes
11. Copy your Render URL: `https://ai-pipeline-backend.onrender.com`
12. Test it: visit `https://ai-pipeline-backend.onrender.com/health`

---

## Step 4 — Vercel (Frontend) ~5 minutes

1. Open `frontend/index.html`
2. Find this line near the top of the script:
   ```js
   const API = window.BACKEND_URL || "https://your-backend.onrender.com";
   ```
3. Replace `https://your-backend.onrender.com` with your actual Render URL
4. Save the file
5. Go to **vercel.com** → create a free account
6. Click **"Add New Project"**
7. Create a GitHub repo called `ai-pipeline-frontend`
8. Upload `frontend/index.html` and `frontend/vercel.json` to it
9. In Vercel → import that repo → click **Deploy**
10. Your app is live at: `https://your-project.vercel.app` 🎉

---

## Verification Checklist

After deploying everything, test in this order:

- [ ] `SUPABASE_URL/health` — Supabase project is active
- [ ] `SPACE_1_URL/health` → `{"status":"ok","space":"ingest-profile"}`
- [ ] `SPACE_2_URL/health` → `{"status":"ok","space":"clean-engineer"}`
- [ ] `SPACE_3_URL/health` → `{"status":"ok","space":"train-evaluate"}`
- [ ] `SPACE_4_URL/health` → `{"status":"ok","space":"explain-report"}`
- [ ] `RENDER_URL/health` → shows all 4 space URLs
- [ ] Open Vercel URL → upload a CSV → watch the pipeline run end to end

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Space shows "Building" forever | Check the build logs — usually a missing library |
| Space shows "Error" | Check app.py for syntax errors |
| Render backend 502 | Check Render logs — usually an env var is missing |
| Pipeline fails at Space 2 | File too large — dataset > 200MB won't fit in Space RAM |
| Supabase connection error | Double-check SUPABASE_URL and SUPABASE_ANON_KEY |
| CORS error in browser | Make sure Render backend has CORS middleware (it does) |

---

## Free Tier Limits

| Service | Limit | What happens when exceeded |
|---------|-------|---------------------------|
| HF Spaces | 2GB RAM per Space | Space crashes — reduce dataset size |
| Render | 750 hrs/month, sleeps after 15min idle | First request after sleep takes ~30s |
| Supabase | 500MB DB, 2 projects | Old runs may need deletion |
| Vercel | Unlimited static hosting | No limit for frontend |

---

## Architecture Reminder

```
Browser (Vercel)
    ↓ upload file
Render Backend
    ↓ POST /run (file)          ↓ saves status
Space 1: Ingest          Supabase DB
    ↓ schema + target
Space 2: Clean
    ↓ X, y
Space 3: Train
    ↓ model results
Space 4: Report
    ↓ html + json
Render Backend → saves to Supabase
    ↓
Browser polls /status → shows results
```
