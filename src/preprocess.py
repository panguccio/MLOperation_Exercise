import pandas as pd
from sklearn.model_selection import train_test_split

def df_to_river_format(X, y=None):
    """
    Converts a Pandas DataFrame into a list of dictionaries.
    River processes data one row (one dict) at a time.
    """
    stream_X = X.to_dict(orient='records')
    
    if y is not None:
        stream_y = y.tolist()
        return stream_X, stream_y
    
    return stream_X

def preprocess_data(df, target_col='target', test_size=0.2):
    """Splits the DataFrame into Train and Test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test