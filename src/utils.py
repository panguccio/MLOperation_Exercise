import os
import re
import joblib

def get_next_version(model_dir="models"):
    """Scans the model directory to determine the next available version number."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1

    files = os.listdir(model_dir)
    version_pattern = re.compile(r"model_v(\d+).pkl")
    
    versions = []
    for f in files:
        match = version_pattern.match(f)
        if match:
            versions.append(int(match.group(1)))
    
    if not versions:
        return 1
    
    return max(versions) + 1

def save_model(model, scaler, metrics, model_dir="models"):
    """Saves the River model to a pickle file with a new version number."""
    version = get_next_version(model_dir)
    
    model_path = os.path.join(model_dir, f"model_v{version}.pkl")
    
    # Save Model (River pipelines include the scaler, so we just save the model object)
    joblib.dump(model, model_path)
    
    print(f"✅ Model saved as Version {version} (Acc: {metrics.get('accuracy', 0):.2f})")
    return model_path, version

def load_latest_model(model_dir="models"):
    """Loads the most recent model version."""
    current_v = get_next_version(model_dir) - 1
    if current_v < 1:
        raise FileNotFoundError("No models found. Please run initial training first.")
        
    model_path = os.path.join(model_dir, f"model_v{current_v}.pkl")
    
    model = joblib.load(model_path)
    
    return model, None, current_v