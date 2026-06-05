---
title: AI Pipeline Clean Engineer
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Space 2 — Clean & Engineer

Part of the **AI Dataset Intelligence Pipeline**.

## What it does
- Removes duplicates and constant columns
- Imputes missing values (mean / median / mode / KNN)
- Normalizes string columns
- Clips outliers using IQR
- Extracts datetime features (year, month, day, etc.)
- Encodes categorical columns (one-hot / frequency / label)
- Scales numeric columns (standard / robust)
- Selects best features (variance threshold + correlation pruning)

## Endpoint
`POST /run` — send schema + raw data, get back cleaned X and y
