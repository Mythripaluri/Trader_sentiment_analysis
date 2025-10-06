"""
Data utilities for trader sentiment analysis project.
This module contains functions for downloading, loading, and preprocessing data.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def download_file_from_gdrive(url, destination_path):
    """
    Download a file from Google Drive using the provided URL.
    
    Args:
        url (str): Google Drive download URL
        destination_path (str): Path where the file should be saved
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Successfully downloaded file to {destination_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def load_trader_data(file_path):
    """
    Load and perform initial preprocessing of trader data.
    
    Args:
        file_path (str): Path to the trader data CSV file
    
    Returns:
        pd.DataFrame: Loaded trader data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded trader data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading trader data: {e}")
        return None

def load_sentiment_data(file_path):
    """
    Load and perform initial preprocessing of sentiment data.
    
    Args:
        file_path (str): Path to the sentiment data CSV file
    
    Returns:
        pd.DataFrame: Loaded sentiment data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded sentiment data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        return None

def initial_data_inspection(df, data_name="Dataset"):
    """
    Perform initial inspection of the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to inspect
        data_name (str): Name of the dataset for display purposes
    
    Returns:
        None
    """
    print(f"\n=== {data_name} Initial Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())

def standardize_datetime(df, datetime_col):
    """
    Standardize datetime column format.
    
    Args:
        df (pd.DataFrame): DataFrame containing the datetime column
        datetime_col (str): Name of the datetime column
    
    Returns:
        pd.DataFrame: DataFrame with standardized datetime
    """
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        df = df.sort_values(datetime_col).reset_index(drop=True)
        print(f"Successfully standardized {datetime_col} column")
        return df
    except Exception as e:
        print(f"Error standardizing datetime column {datetime_col}: {e}")
        return df

def detect_outliers(df, numeric_cols, method='iqr', factor=1.5):
    """
    Detect outliers in numeric columns using IQR method.
    
    Args:
        df (pd.DataFrame): Dataset
        numeric_cols (list): List of numeric columns to check
        method (str): Method for outlier detection ('iqr' or 'zscore')
        factor (float): Factor for outlier detection
    
    Returns:
        dict: Dictionary with outlier information for each column
    """
    outliers_info = {}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outliers_info[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'bounds': (lower_bound, upper_bound) if method == 'iqr' else None
        }
        
        print(f"Column '{col}': {outliers_info[col]['count']} outliers ({outliers_info[col]['percentage']:.2f}%)")
    
    return outliers_info