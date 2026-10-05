

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        print('='   * 50)
        print(f"Data loaded successfully from {file_path}")
        print('='   * 50)
        print(f"{df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print('='   * 50)
        print(f"An error occurred while loading data: {e}")
        print('='   * 50)
        return None
