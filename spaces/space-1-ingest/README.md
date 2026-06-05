---
title: AI Pipeline Ingest Profile
emoji: 📂
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Space 1 — Ingest & Profile

This Space is part of the **AI Dataset Intelligence Pipeline**.

## What it does
- Receives uploaded dataset files (CSV, Excel, JSON, Parquet)
- Parses and sanitises column names
- Profiles every column (types, nulls, stats, outliers)
- Auto-detects the target column using heuristics + statistics
- Returns structured JSON to the Render backend

## Endpoint
`POST /run` — send a file, get back full schema + target detection

## Part of
| Space | Job |
|---|---|
| **Space 1 (this one)** | Ingest & Profile |
| Space 2 | Clean & Engineer |
| Space 3 | Train & Evaluate |
| Space 4 | Explain & Report |
