from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

app = FastAPI(title="AeroGuard AI Backend", description="ISRO Burn-In Anomaly Detection API")

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
    Takes a batch of components, calculates lot-relative stats, 
    and returns predictions for both Module A and Module B.
    """
    if not components:
        raise HTTPException(status_code=400, detail="Empty batch provided.")

    # Convert incoming JSON to Pandas DataFrame
    df = pd.DataFrame([comp.dict() for comp in components])
    
    # 1. Prepare data for Module A (Dynamic Limits)
    df['Lot_24h_Mean'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform('mean')
    df['Lot_24h_Std'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform('std')
    df['Relative_Z_Score_24h'] = (df['Leakage_24h_uA'] - df['Lot_24h_Mean']) / (df['Lot_24h_Std'] + 1e-5)
    
    # Run Module A
    df['Module_A_Anomaly'] = iso_forest.predict(df[['Relative_Z_Score_24h']])
    df['Is_Anomaly'] = df['Module_A_Anomaly'].apply(lambda x: True if x == -1 else False)
    
    # 2. Prepare data for Module B (Drift Predictor)
    df['Velocity_0_to_24'] = df['Leakage_24h_uA'] - df['Leakage_0h_uA']
    features_B = ['Leakage_0h_uA', 'Leakage_24h_uA', 'Velocity_0_to_24']
    
    # Run Module B
    df['Predicted_168h_uA'] = ridge_model.predict(df[features_B])
    df['Predicted_168h_uA'] = df['Predicted_168h_uA'].round(2)
    
    # Format the response
    results = df[['Component_ID', 'Lot_ID', 'Is_Anomaly', 'Predicted_168h_uA']].to_dict(orient='records')
    
    return {
        "batch_size": len(results),
        "anomalies_detected": int(df['Is_Anomaly'].sum()),
        "results": results
    }
