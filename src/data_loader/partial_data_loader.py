"""
This model loads randomly selected rows from the full dataset
"""
import pandas as pd
def load_random_rows(file_path: str, num_rows=int(5e5), chunk_size=int(1e6)):
    """
    Load a specified number of random rows from a large CSV file 
    without loading the entire file into memory.
    
    Args:
        file_path (str): The path to the CSV file.
        num_rows (int): The number of random rows to load.
        chunk_size (int): The number of rows to read at a time. Default is 10,000.
    
    Returns:
        pd.DataFrame: A DataFrame containing the specified number of random rows.
    
    Raises:
        ValueError: If the requested number of rows exceeds the total number of rows in the file.
    """
    random_rows = []
    reader = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in reader:
        if len(chunk) < num_rows:
            random_rows.append(chunk.sample(frac=1))
        else:
            random_rows.append(chunk.sample(n=num_rows))
            random_rows_df = pd.concat(random_rows).sample(n=num_rows, random_state=1)
    return random_rows_df
