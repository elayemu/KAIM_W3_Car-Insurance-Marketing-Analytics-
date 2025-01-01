import pandas as pd
import numpy as np
def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Handles missing values by:
    1. Dropping columns with >50% missing values or 100% missing values.
    2. Filling missing values:
        - Mode for categorical columns.
        - Mean/Median for numerical columns based on skewness.

    Parameters:
        df (pd.DataFrame): The input dataset.
        threshold (float): Threshold for dropping columns with missing values.

    Returns:
        pd.DataFrame: Dataset with handled missing values.
    """
    # Step 1: Drop Columns with >50% Missing Values or 100% Missing Values
    missing_percentage = df.isnull().mean()
    
    # Drop columns with missing values greater than the threshold (default 50%)
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    # Add columns with 100% missing values to the drop list
    cols_to_drop = cols_to_drop.append(missing_percentage[missing_percentage == 1.0].index)
    
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns with >{threshold*100}% or 100% missing values: {list(cols_to_drop)}")
    
    # Step 2: Fill Missing Values
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            # Categorical Columns - Fill with mode
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif df[col].dtype in ['int64', 'float64']:
            # Numerical Columns - Fill based on skewness
            if df[col].isna().sum() > 0:
                if abs(df[col].skew()) > 1:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())
    
    print("Filled missing values: Mode for categorical, Mean/Median for numerical columns.")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles outliers using the IQR (Interquartile Range) method.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
    
    Returns:
        pd.DataFrame: Dataset with capped outliers.
    """
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap values outside bounds
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    print("Capped outliers using IQR method for numerical columns.")
    return df
def detect_missing_values(df: pd.DataFrame):
    """
    Detects missing values in the dataset and returns a summary.
    """
    missing_summary = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_summary,
        'Percentage Missing': missing_percentage
    })
    return missing_df[missing_df['Missing Values'] > 0]



def detect_outliers(df: pd.DataFrame, threshold: float = 1.5):
    """
    Detects outliers based on the IQR method.
    """
    outlier_info = {}
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'Total Outliers': len(outliers),
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        }
    
    return pd.DataFrame.from_dict(outlier_info, orient='index')


