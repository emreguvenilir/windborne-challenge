# 🌎 Windborne Global Balloon Tracking System

A full-stack, ML-powered **real-time atmospheric modeling platform** that collects, cleans, and predicts global weather balloon telemetry.  
The system automates data ingestion, transformation, forecasting, and visualization — providing live, short-term weather predictions from thousands of balloon data streams.

---

## 🚀 Overview

**Windborne** integrates a Python ETL pipeline, weather API ingestion, data interpolation, and an **LSTM forecasting model**, all deployed through a Dockerized microservice architecture.

Key features include:
- Automated 15-minute inference cycles via **APScheduler**
- Resilient batch ingestion with **adaptive rate limiting** and **caching**
- Robust **data validation and interpolation** for missing or partial weather data
- **LSTM-based forecasting** for short-term atmospheric predictions
- Fully **containerized** deployment on a **VPS (DigitalOcean Droplet)**
- **Interactive Flask UI** for real-time visualization and monitoring

### 🧩 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Languages** | Python |
| **Frameworks** | Flask, Gunicorn, TensorFlow |
| **Data & ML** | Pandas, NumPy, scikit-learn |
| **Infrastructure** | Docker, APScheduler, DigitalOcean VPS |
| **APIs** | Open-Meteo API, Windborne Systems API |
| **Visualization** | Flask UI, Matplotlib |
| **Storage** | Cloudflare R2 (optional for persisted data) |

---
