import pandas as pd
import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.datasets import load_iris
import os
import time
import random
import neptune
import yaml
import pickle

# ---------- 1. Load YAML configuration ----------
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract relevant configuration parameters
experiment_cfg = config["experiment"]
data_cfg = config["data"]
model_cfg = config["model"]
training_cfg = config["training"]

# ---------- 2. Initialize Neptune for experiment tracking ----------
run = neptune.init_run(
    project="YourWorkspace/YourProject",  # Replace with your actual project name
    api_token="YOUR_API_TOKEN",           # Or set it as an environment variable
)

# Log the configuration to Neptune
run["config"] = config

# ---------- 3. Create iris data stream ----------
def create_iris_stream():
    """Create a stream of iris data samples"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Shuffle the data to simulate random streaming
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    return X_shuffled, y_shuffled

# ---------- 4. Online learning with Hoeffding Tree ----------
def online_learning_iris():
    """Online learning with Hoeffding Tree on iris data"""
    print("=== Online Learning with Hoeffding Tree ===")
    
    # Setup Hoeffding Tree estimator
    ht = HoeffdingTreeClassifier(random_state=model_cfg["random_state"])
    
    # Setup variables to control loop and track performance
    n_samples = 0
    correct_cnt = 0
    
    # Create iris data stream
    X_stream, y_stream = create_iris_stream()
    total_samples = len(X_stream)
    
    print(f"Starting online learning with {total_samples} iris samples")
    print("Simulating data stream (one sample every 2 seconds)...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for i in range(total_samples):
            # Get current sample from stream
            X_sample = X_stream[i].reshape(1, -1)
            y_sample = np.array([y_stream[i]])
            
            # Make prediction if model has been trained
            if n_samples > 0:
                y_pred = ht.predict(X_sample)
                if y_sample[0] == y_pred[0]:
                    correct_cnt += 1
                
                # Calculate and display current accuracy
                accuracy = correct_cnt / n_samples
                run["metrics/accuracy"].log(accuracy)  # Log accuracy to Neptune
                print(f"Sample {n_samples+1}: Features={X_sample[0]}, Predicted={y_pred[0]}, Actual={y_sample[0]}, Accuracy={accuracy:.4f}")
            else:
                print(f"Sample {n_samples+1}: Features={X_sample[0]}, Actual={y_sample[0]} (First sample - no prediction)")

            # Partial fit (online learning)
            ht.partial_fit(X_sample, y_sample)
            n_samples += 1
            
            # Simulate streaming delay
            time.sleep(training_cfg["delay"])  # Wait specified delay between samples
            
    except KeyboardInterrupt:
        print(f"\n=== Online Learning Interrupted ===")
    
    # Final results
    print(f"\n=== Online Learning Complete ===")
    if n_samples > 1:
        final_accuracy = correct_cnt / (n_samples - 1)  # Exclude first sample (no prediction made)
        print(f"Total samples processed: {n_samples}")
        print(f"Correct predictions: {correct_cnt}/{n_samples - 1}")
        print(f"Final accuracy: {final_accuracy:.4f}")
    else:
        print("No predictions made (need at least 2 samples)")
    
    # ---------- 5. Save the trained model as a pickle file ----------
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ht, f)
    
    # Upload the model to Neptune
    run["artifacts/model"].upload(model_path)

    # Stop Neptune run
    run.stop()

# ---------- 6. Main execution ----------
if __name__ == "__main__":
    online_learning_iris()
