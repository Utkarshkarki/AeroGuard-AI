import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('saved_models', exist_ok=True)

def load_data(filepath="Data/isro_burn_in_dataset.csv"):
    return pd.read_csv(filepath)
def train_module_a(df):
    print("--- Training Module A: Dynamic Anomaly Detection (IQR Robust) ---")
    
    # 1. Calculate Median, Q1 (25th percentile), and Q3 (75th percentile) per Lot
    df['Lot_24h_Median'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform('median')
    df['Lot_24h_Q1'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform(lambda x: x.quantile(0.25))
    df['Lot_24h_Q3'] = df.groupby('Lot_ID')['Leakage_24h_uA'].transform(lambda x: x.quantile(0.75))
    
    # 2. Calculate IQR (Interquartile Range)
    df['Lot_24h_IQR'] = df['Lot_24h_Q3'] - df['Lot_24h_Q1']
    
    # 3. Calculate Robust IQR Score
    # We add 1e-5 to prevent division by zero if chips in a lot read the exact same value
    df['Robust_IQR_Score'] = (df['Leakage_24h_uA'] - df['Lot_24h_Median']) / (df['Lot_24h_IQR'] + 1e-5)
    
    # 4. Train Isolation Forest on the new Robust metric
    iso_forest = IsolationForest(contamination=0.04, random_state=42)
    features_A = ['Robust_IQR_Score']
    
    df['Module_A_Pred'] = iso_forest.fit_predict(df[features_A])
    df['Is_Anomaly_Pred'] = df['Module_A_Pred'].apply(lambda x: 1 if x == -1 else 0)
    
    # Save the model
    joblib.dump(iso_forest, 'saved_models/module_a_iso_forest.pkl')
    print("✅ Module A Model saved to 'saved_models/module_a_iso_forest.pkl'")
    
    # Evaluate Module A
    df['Is_Anomaly_Actual'] = df['Label'].apply(lambda x: 1 if x in ['Latent_Defect', 'Static_Failure'] else 0)
    print("\n--- Module A Evaluation Metrics ---")
    print(classification_report(df['Is_Anomaly_Actual'], df['Is_Anomaly_Pred'], target_names=['Normal', 'Anomaly']))
    
    return df


def train_module_b(df):
    print("\n--- Training Module B: 168h Drift Prediction ---")
    
    # Feature Engineering
    df['Velocity_0_to_24'] = df['Leakage_24h_uA'] - df['Leakage_0h_uA']
    
    features_B = ['Leakage_0h_uA', 'Leakage_24h_uA', 'Velocity_0_to_24']
    X = df[features_B]
    y = df['Leakage_168h_uA']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Ridge Regression
    ridge_model = Ridge(alpha=10.0) 
    ridge_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(ridge_model, 'saved_models/module_b_ridge.pkl')
    print("✅ Module B Model saved to 'saved_models/module_b_ridge.pkl'")
    
    # Evaluate
    predictions = ridge_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error (MAE) at 168h: {mae:.2f} µA")
    
    return ridge_model, X_train

def generate_shap_explanations(ridge_model, X_train):
    print("\n--- Generating SHAP Explainability (For QA Inspectors) ---")
    
    # Use SHAP LinearExplainer for our Ridge model
    explainer = shap.LinearExplainer(ridge_model, X_train)
    
    # Let's explain a single prediction (e.g., the 5th component in our dataset)
    sample_component = X_train.iloc[[5]]
    shap_values = explainer.shap_values(sample_component)
    
    print(f"Base Value (Average 168h prediction): {explainer.expected_value:.2f} µA")
    print(f"Features contributing to this specific prediction:")
    
    for feature, shap_val, actual_val in zip(X_train.columns, shap_values[0], sample_component.values[0]):
        impact = "increased" if shap_val > 0 else "decreased"
        print(f" - {feature} (Value: {actual_val:.2f}) {impact} the prediction by {abs(shap_val):.2f} µA")

if __name__ == "__main__":
    df = load_data()
    df_evaluated = train_module_a(df)
    model_b, X_train_b = train_module_b(df_evaluated)
    
    # Run explainability test
    generate_shap_explanations(model_b, X_train_b)
