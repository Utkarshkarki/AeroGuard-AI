import pandas as pd
import numpy as np

def generate_isro_burn_in_data(num_lots=5, chips_per_lot=200, anomaly_rate=0.03):
    """
    Generates synthetic time-series parametric data for electronic component Burn-In testing.
    Simulates measurements at 0h, 24h, 96h, and 168h.
    """
    np.random.seed(42) # For reproducibility
    
    data = []
    component_id_counter = 1
    
    for lot in range(1, num_lots + 1):
        # Each lot has its own baseline manufacturing "normal"
        lot_base_leakage = np.random.uniform(5.0, 15.0) 
        
        for _ in range(chips_per_lot):
            is_anomaly = np.random.rand() < anomaly_rate
            
            if not is_anomaly:
                # Normal Component: Tiny variations, very stable drift over time
                val_0h = np.random.normal(lot_base_leakage, 1.0)
                val_24h = val_0h + np.random.normal(0.2, 0.1)
                val_96h = val_24h + np.random.normal(0.5, 0.2)
                val_168h = val_96h + np.random.normal(0.5, 0.2)
                label = "Normal"
            else:
                # Latent Defect Component: Starts somewhat normal, but drifts aggressively
                # It might still stay under the strict 50uA static limit, making it hard to catch!
                val_0h = np.random.normal(lot_base_leakage, 2.0)
                val_24h = val_0h + np.random.uniform(1.0, 5.0)   # Early warning sign
                val_96h = val_24h + np.random.uniform(5.0, 15.0) # Degradation accelerating
                val_168h = val_96h + np.random.uniform(10.0, 25.0) # Severe drift
                label = "Latent_Defect"
                
            data.append({
                "Component_ID": f"COMP_{component_id_counter:04d}",
                "Lot_ID": f"LOT_{lot:02d}",
                "Leakage_0h_uA": round(max(0.1, val_0h), 2),
                "Leakage_24h_uA": round(max(0.1, val_24h), 2),
                "Leakage_96h_uA": round(max(0.1, val_96h), 2),
                "Leakage_168h_uA": round(max(0.1, val_168h), 2),
                "Label": label
            })
            component_id_counter += 1
            
    df = pd.DataFrame(data)
    
    # Introduce a few "Static Failures" (chips that are just completely broken from the start > 50uA)
    static_fails_idx = df.sample(frac=0.01).index
    df.loc[static_fails_idx, 'Leakage_0h_uA'] = np.random.uniform(55.0, 80.0)
    df.loc[static_fails_idx, 'Label'] = "Static_Failure"
    
    return df

if __name__ == "__main__":
    print("Initializing ISRO Burn-In Data Simulation...")
    df = generate_isro_burn_in_data(num_lots=10, chips_per_lot=500, anomaly_rate=0.04)
    
    output_filename = "isro_burn_in_dataset.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"Success! Generated {len(df)} components across {df['Lot_ID'].nunique()} lots.")
    print(f"Data saved to: {output_filename}")
    print("\nDataset Breakdown:")
    print(df['Label'].value_counts())
