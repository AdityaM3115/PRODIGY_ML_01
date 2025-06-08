import pandas as pd

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data = data.dropna()  # Remove rows with missing values
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
