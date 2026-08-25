# 🚀 AeroGuard AI

**AI-Driven Anomaly Detection in Component Burn-In & Environmental Stress Screening (ESS)**

*Developed for the Indian Space Research Organisation (ISRO) — Problem Statement ID: 26170*

---

## 📌 Executive Summary

High-reliability electronic components undergo rigorous Environmental Stress Screening (ESS), including Burn-In testing (operating components at elevated temperatures, e.g., 125°C, for up to 168 hours).

Traditional screening relies on static parametric pass/fail limits (e.g., maximum datasheet leakage current). However, latent defects—components that satisfy static limits but exhibit anomalous relative drift over time—frequently escape into flight payloads, leading to mission-critical field failures in orbit.

**AeroGuard AI** is a space-grade anomaly detection system that replaces rigid static cutoffs with:
- **Dynamic Lot Normalization (IQR Scaling)**
- **Predictive Time-Series Regression (0h to 24h forecasting 168h)**
- **SHAP Explainability**
- **Deterministic Mission Assurance Rules Engine**

---

## 🛠 Tech Stack

| Layer | Technologies & Frameworks |
| --- | --- |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Recharts, Lucide Icons |
| **Backend API** | FastAPI, Uvicorn, Pydantic, CORS Middleware |
| **ML Engine** | Python 3.12, Scikit-Learn, NumPy, Pandas, Joblib |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Containerization** | Docker, Docker Compose (Air-Gapped Ready) |

---

## 🏗 System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUTOMATED TEST EQUIPMENT (ATE)                        │
│                   Raw Time-Series Burn-In Telemetry (CSV)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FRONTEND: React 18 + Tailwind UI                       │
│  - Secure Mission Assurance Auth Gate                                       │
│  - Telemetry Drift Visualization (Recharts)                                 │
│  - Real-Time Inspection Logs & QA Decision Controls                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST API (JSON)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKEND API: FastAPI / Rules Engine                     │
│  - Dynamic IQR Feature Scaling per Production Lot                           │
│  - Deterministic Safety Catch (Z-Score > 2.5, 168h Forecast > 20µA)         │
│  - CORS & Async Data Pipelining                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Loads Serialized .pkl Models
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MACHINE LEARNING ENGINE                          │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Module A: Dynamic Anomaly Detector (Robust IQR + Isolation Forest)      │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │ Module B: Time-Series Drift Predictor (Ridge Regression L2, alpha=10.0) │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │ Explainability: SHAP LinearExplainer                                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### Module A: Dynamic Lot-Relative Anomaly Detection
- Isolates components drifting from their specific production batch/wafer.
- Utilizes Interquartile Range (IQR) Scaling to withstand low-resolution sensor quantization and eliminate zero-variance division errors.
- Leverages an Isolation Forest trained on robust scores to identify subtle relative outliers.

### Module B: Early Time-Series Drift Predictor
- Inputs early telemetry points (0h, 24h, and velocity) to forecast 168h leakage.
- Employs regularized Ridge Regression (L2 regularization, alpha=10.0) to eliminate overfitting on limited early time steps.
- Achieves a Mean Absolute Error (MAE) of **2.16 µA** on synthetic test datasets.

### Deterministic Mission Assurance Rules Engine
- Incorporates space-grade failsafes to guarantee Zero False Negatives.
- Merges ML outputs with mathematical threshold overrides (IQR Score > 2.5 or Predicted 168h > 20.0 µA).

### Human-in-the-Loop SHAP Explainability
- Translates machine learning decisions into readable parametric attribution reports for mission assurance inspectors.

---

## 📂 Project Repository Structure

```text
aeroguard-ai/
│
├── backend/                         # FastAPI Backend Application
│   ├── main.py                      # REST Endpoints, CORS, & Rules Engine
│   └── requirements.txt             # Backend Dependencies
│
├── frontend/                        # React 18 + Vite UI
│   ├── public/                      # Static Assets
│   ├── src/
│   │   ├── App.jsx                  # Auth Gate & Control Dashboard
│   │   ├── main.jsx                 # React Entrypoint
│   │   └── index.css                # Tailwind CSS Directives
│   ├── package.json                 # JavaScript Dependencies
│   └── tailwind.config.js           # Tailwind Styling Configuration
│
├── ml_engine/                       # Machine Learning Subsystem
│   ├── generate_data.py             # Synthetic ISRO Burn-In Data Generator
│   ├── ml_pipeline.py               # Model Training (Modules A & B + SHAP)
│   ├── requirements.txt             # ML Dependencies
│   ├── Data/
│   │   └── isro_burn_in_dataset.csv # Generated Time-Series Telemetry
│   └── saved_models/                # Serialized Model Artifacts (.pkl)
│       ├── module_a_iso_forest.pkl
│       └── module_b_ridge.pkl
│
├── .gitignore                       # Ignored Caches, Models, & node_modules
├── docker-compose.yml               # Container Orchestration
└── README.md                        # Documentation
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Node.js 18+ and npm
- Git

### Step 1: Clone the Repository & Setup Environment
```bash
git clone https://github.com/Utkarshkarki/AeroGuard-AI.git
cd AeroGuard-AI

# Create virtual environment
python -m venv aeroguard_env

# Activate Virtual Environment
# Windows (Git Bash):
source aeroguard_env/Scripts/activate
# Windows (PowerShell):
.\aeroguard_env\Scripts\activate
# Linux/macOS:
source aeroguard_env/bin/activate
```

### Step 2: Machine Learning Subsystem

Install ML Dependencies:
```bash
pip install pandas numpy scikit-learn shap joblib
```

Generate Synthetic Burn-In Data:
```bash
cd ml_engine/Data
python generate_data.py
cd ../..
```

Train Models & Generate SHAP Metrics:
```bash
cd ml_engine
python ml_pipeline.py
cd ..
```
*Models will be saved to `ml_engine/saved_models/`.*

### Step 3: Launch FastAPI Backend

Navigate & Install Dependencies:
```bash
cd backend
pip install fastapi uvicorn pydantic pandas joblib
```

Run Server:
```bash
uvicorn main:app --reload
```
*Backend will run on `http://127.0.0.1:8000`. Interactive API Docs are accessible at `http://127.0.0.1:8000/docs`.*

### Step 4: Launch React Dashboard

Open a new terminal window and navigate to frontend:
```bash
cd frontend
npm install
npm run dev
```
*Open Dashboard: Navigate to `http://localhost:5173` in your browser.*

---

## 📡 API Reference Specification

### `POST /analyze_batch`
Processes a batch of component telemetry and outputs anomaly classifications along with 168h drift predictions.

#### Request Body
```json
[
  {
    "Component_ID": "COMP_001",
    "Lot_ID": "LOT_01",
    "Leakage_0h_uA": 5.2,
    "Leakage_24h_uA": 5.4
  },
  {
    "Component_ID": "COMP_BROKEN",
    "Lot_ID": "LOT_01",
    "Leakage_0h_uA": 5.1,
    "Leakage_24h_uA": 25.4
  }
]
```

#### Response Body
```json
{
  "batch_size": 2,
  "anomalies_detected": 1,
  "results": [
    {
      "Component_ID": "COMP_001",
      "Lot_ID": "LOT_01",
      "Is_Anomaly": false,
      "Predicted_168h_uA": 6.15
    },
    {
      "Component_ID": "COMP_BROKEN",
      "Lot_ID": "LOT_01",
      "Is_Anomaly": true,
      "Predicted_168h_uA": 33.78
    }
  ]
}
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.