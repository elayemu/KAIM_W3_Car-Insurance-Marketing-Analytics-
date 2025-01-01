# src/data_loader.py
# 
import pandas as pd

def load_data(txt_file_path: str, delimiter: str = "\t"):
    """
    Loads data from a .txt file and converts it to a pandas DataFrame.
    
    :param txt_file_path: Path to the .txt file
    :param delimiter: Delimiter used in the .txt file (default is tab-separated)
    :return: DataFrame containing the data from the .txt file
    """
    df = pd.read_csv(txt_file_path, delimiter='|', encoding='utf-8', low_memory=False)
    return df


def save_to_csv(df, csv_file_path: str):
    """
    Saves the DataFrame to a CSV file.
    
    :param df: DataFrame to be saved
    :param csv_file_path: Path to the output .csv file
    """
    df=df.to_csv(csv_file_path, index=False)
    # df = pd.read_csv(csv_file_path, low_memory=False)
    print(f"File saved to {csv_file_path}")
    

    