"""
This module load full dataset
"""

import pandas as pd
def full_data_loader(file_path: str):
    """
    Load full dataset
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame of all data.
    """
    df = pd.read_csv(file_path)
    return df
