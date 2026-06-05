---
title: AI Pipeline Explain Report
emoji: 📄
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# Space 4 — Explain & Report

Part of the **AI Dataset Intelligence Pipeline**.

## What it does
- Generates feature importance explanations
- Builds a full HTML report
- Builds a structured JSON report
- Generates recommended next steps

## Endpoint
`POST /run` — send all pipeline results, get back HTML + JSON reports
`POST /report/html` — get the HTML report rendered directly in browser
