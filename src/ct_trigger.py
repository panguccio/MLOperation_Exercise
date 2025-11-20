import time
import yaml
import pandas as pd
from scipy.stats import ks_2samp

from src.data_loader import load_data, get_new_data_batch
from src.train import train_or_update_model

def check_drift(reference_data, new_data, threshold=0.05):
    """
    Drift Detection: Uses Kolmogorov-Smirnov (KS) Test.
    Checks if the 'sepal length' (col 0) distribution has changed.
    """
    ref_feature = reference_data.iloc[:, 0]
    new_feature = new_data.iloc[:, 0]
    
    # Statistic comparison
    statistic, p_value = ks_2samp(ref_feature, new_feature)
    
    # If p_value is low, distributions are different -> DRIFT
    return p_value < threshold, p_value

def continuous_training_loop():
    print("🔄 Starting MLOps Monitor...")
    
    config = yaml.safe_load(open("configs/config.yaml", "r"))
    threshold = config["ct_pipeline"]["drift_p_value_threshold"]
    
    # 1. Load Reference Data (Baseline History)
    reference_data, _ = load_data()
    
    # 2. Ensure Initial Model Exists
    print("Checking for base model...")
    try:
        # Passing None forces it to check or train from scratch
        train_or_update_model(new_data_df=None)
    except Exception as e:
        print(f"Initial setup: {e}")

    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n--- Cycle {iteration} ---")
        
        # 3. Simulate Incoming Data
        # Triggers drift simulation every 3rd cycle
        simulate_drift = (iteration % 3 == 0) 
        new_data = get_new_data_batch(n_samples=50, drift_simulation=simulate_drift)
        
        # 4. Check Drift
        is_drift, p_val = check_drift(reference_data, new_data, threshold)
        print(f"📉 Drift P-Value: {p_val:.5f} (Threshold: {threshold})")
        
        if is_drift:
            print("🚨 DRIFT DETECTED! Triggering Incremental Learning...")
            
            # 5. TRIGGER UPDATE
            # We pass the 'bad' data to the model so it adapts to it
            train_or_update_model(new_data_df=new_data)
            
            # Update reference to include this new reality
            reference_data = pd.concat([reference_data, new_data]).tail(150)
            
        else:
            print("✅ Data looks normal. No update needed.")
            
        # Wait before next check
        time.sleep(config["ct_pipeline"]["check_interval_seconds"])

if __name__ == "__main__":
    try:
        continuous_training_loop()
    except KeyboardInterrupt:
        print("\n🛑 Monitor Stopped.")