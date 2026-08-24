from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

app = FastAPI(title="AeroGuard AI Backend", description="ISRO Burn-In Anomaly Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you would put your React app's URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the expected incoming data structure
class ComponentData(BaseModel):
    Component_ID: str
    Lot_ID: str
    Leakage_0h_uA: float
    Leakage_24h_uA: float

# Load the saved ML models globally when the server starts
# We point to the ml_engine/saved_models folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_engine", "saved_models")

try:
    iso_forest = joblib.load(os.path.join(MODEL_DIR, "module_a_iso_forest.pkl"))
    ridge_model = joblib.load(os.path.join(MODEL_DIR, "module_b_ridge.pkl"))
    print("✅ ML Models loaded successfully into backend.")
except Exception as e:
    print(f"⚠️ Warning: Could not load models. Did you run ml_pipeline.py? Error: {e}")

@app.get("/")
def read_root():
    return {"status": "AeroGuard AI Backend is active and running!"}

@app.post("/analyze_batch")
def analyze_batch(components: List[ComponentData]):
    """
    Takes a batch of components, calculates IQR robust stats, 
    and returns predictions combined with a Safety Rules Engine.
    """
    if not components:
        raise HTTPException(status_code=400, detail="Empty batch provided.")

    df = pd.DataFrame([comp.dict() for comp in components])
    
    # 1. Prepare data for Module A (Dynamic Limits - IQR Robust)
    df['Lot_24h_Median'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform('median')
    df['Lot_24h_Q1'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform(lambda x: x.quantile(0.25))
    df['Lot_24h_Q3'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform(lambda x: x.quantile(0.75))
    df['Lot_24h_IQR'] = df['Lot_24h_Q3'] - df['Lot_24h_Q1']
    
    df['Robust_IQR_Score'] = (df['Leakage_24h_uA'] - df['Lot_24h_Median']) / (df['Lot_24h_IQR'] + 1e-5)
    
    # Run Module A ML
    df['Module_A_ML_Pred'] = iso_forest.predict(df[['Robust_IQR_Score']])
    
    # 2. Prepare data for Module B (Drift Predictor)
    df['Velocity_0_to_24'] = df['Leakage_24h_uA'] - df['Leakage_0h_uA']
    features_B = ['Leakage_0h_uA', 'Leakage_24h_uA', 'Velocity_0_to_24']
    
    # Run Module B ML
    df['Predicted_168h_uA'] = ridge_model.predict(df[features_B])
    df['Predicted_168h_uA'] = df['Predicted_168h_uA'].round(2)
    
    # ========================================================
    # 🛡️ THE RULES ENGINE (ISRO Mission Assurance Logic)
    # ========================================================
    def determine_anomaly(row):
        # Rule 1: The ML Isolation Forest caught it
        if row['Module_A_ML_Pred'] == -1: 
            return True
            
        # Rule 2: Pure Math Dynamic Limit (IQR Score > 2.5 means it is a statistical outlier)
        if row['Robust_IQR_Score'] > 2.5: 
            return True
            
        # Rule 3: Predictive Danger (If Module B predicts it will cross 20µA at 168h)
        if row['Predicted_168h_uA'] > 20.0: 
            return True
            
        return False
        
    df['Is_Anomaly'] = df.apply(determine_anomaly, axis=1)
    
    # Format the response
    results = df[['Component_ID', 'Lot_ID', 'Is_Anomaly', 'Predicted_168h_uA']].to_dict(orient='records')
    
    return {
        "batch_size": len(results),
        "anomalies_detected": int(df['Is_Anomaly'].sum()),
        "results": results
    }

