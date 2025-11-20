from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

def load_data():
    """Loads the standard Iris dataset for initial training."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.feature_names

def get_new_data_batch(n_samples=50, drift_simulation=False):
    """
    Simulates a batch of new incoming data.
    
    Args:
        n_samples: How many rows to simulate.
        drift_simulation: If True, adds noise to 'sepal length' to confuse the model.
    """
    iris = load_iris()
    X = iris.data
    
    # Randomly sample data from the original dataset
    indices = np.random.choice(len(X), n_samples)
    batch_X = X[indices]
    
    if drift_simulation:
        print("⚠️ SIMULATING DATA DRIFT (Shift +2.0 to Sepal Length)")
        # We modify the first column (sepal length) to create statistical drift
        batch_X[:, 0] += 2.0 
        
    df = pd.DataFrame(data=batch_X, columns=iris.feature_names)
    df['target'] = iris.target[indices] # We assume we eventually get labels (for training)
    
    return df