"""
Data preprocessing utilities for trader sentiment analysis.
This module handles data cleaning, standardization, and merging operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

def preprocess_trader_data(df):
    """
    Clean and preprocess the trader data.
    
    Args:
        df (pd.DataFrame): Raw trader data
    
    Returns:
        pd.DataFrame: Cleaned trader data
    """
    print("Starting trader data preprocessing...")
    
    # Create a copy to avoid modifying original data
    trader_clean = df.copy()
    
    # Convert timestamps to datetime
    # The timestamp appears to be in Unix format (milliseconds since epoch)
    trader_clean['datetime'] = pd.to_datetime(trader_clean['Timestamp'], unit='ms')
    trader_clean['date'] = trader_clean['datetime'].dt.date
    
    # Standardize column names (remove spaces, make lowercase)
    column_mapping = {
        'Closed PnL': 'closed_pnl',
        'Size USD': 'size_usd',
        'Size Tokens': 'size_tokens',
        'Execution Price': 'execution_price',
        'Start Position': 'start_position',
        'Timestamp IST': 'timestamp_ist',
        'Transaction Hash': 'transaction_hash',
        'Order ID': 'order_id',
        'Trade ID': 'trade_id'
    }
    
    trader_clean = trader_clean.rename(columns=column_mapping)
    trader_clean.columns = trader_clean.columns.str.lower()
    
    # Remove rows with zero or negative prices (likely data errors)
    initial_count = len(trader_clean)
    trader_clean = trader_clean[trader_clean['execution_price'] > 0]
    removed_count = initial_count - len(trader_clean)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with invalid execution prices")
    
    # Handle extreme outliers in PnL (keeping reasonable range)
    pnl_q99 = trader_clean['closed_pnl'].quantile(0.99)
    pnl_q01 = trader_clean['closed_pnl'].quantile(0.01)
    
    # Flag extreme outliers but don't remove them yet
    trader_clean['is_outlier_pnl'] = (
        (trader_clean['closed_pnl'] > pnl_q99) | 
        (trader_clean['closed_pnl'] < pnl_q01)
    )
    
    # Create additional features
    trader_clean['is_profitable'] = trader_clean['closed_pnl'] > 0
    trader_clean['abs_pnl'] = trader_clean['closed_pnl'].abs()
    trader_clean['is_long'] = trader_clean['direction'] == 'Long'
    trader_clean['hour'] = trader_clean['datetime'].dt.hour
    trader_clean['day_of_week'] = trader_clean['datetime'].dt.dayofweek
    
    # Calculate approximate leverage (if start_position is not zero)
    trader_clean['leverage_approx'] = np.where(
        trader_clean['start_position'] != 0,
        trader_clean['size_usd'] / trader_clean['start_position'].abs(),
        np.nan
    )
    
    print(f"Trader data preprocessing completed. Final shape: {trader_clean.shape}")
    return trader_clean

def preprocess_sentiment_data(df):
    """
    Clean and preprocess the sentiment data.
    
    Args:
        df (pd.DataFrame): Raw sentiment data
    
    Returns:
        pd.DataFrame: Cleaned sentiment data
    """
    print("Starting sentiment data preprocessing...")
    
    sentiment_clean = df.copy()
    
    # Convert timestamp to datetime
    sentiment_clean['datetime'] = pd.to_datetime(sentiment_clean['timestamp'], unit='s')
    sentiment_clean['date'] = pd.to_datetime(sentiment_clean['date'])
    
    # Create binary sentiment categories
    fear_categories = ['Extreme Fear', 'Fear']
    greed_categories = ['Greed', 'Extreme Greed']
    
    sentiment_clean['sentiment_binary'] = sentiment_clean['classification'].apply(
        lambda x: 'Fear' if x in fear_categories else 
                 ('Greed' if x in greed_categories else 'Neutral')
    )
    
    # Create numeric sentiment score (normalized)
    # Lower values = more fear, higher values = more greed
    sentiment_clean['sentiment_score_normalized'] = (sentiment_clean['value'] - 50) / 50
    
    # Create categorical sentiment levels
    def categorize_sentiment(value):
        if value <= 20:
            return 'Extreme Fear'
        elif value <= 40:
            return 'Fear'
        elif value <= 60:
            return 'Neutral'
        elif value <= 80:
            return 'Greed'
        else:
            return 'Extreme Greed'
    
    sentiment_clean['sentiment_category'] = sentiment_clean['value'].apply(categorize_sentiment)
    
    print(f"Sentiment data preprocessing completed. Final shape: {sentiment_clean.shape}")
    return sentiment_clean

def merge_trader_sentiment_data(trader_df, sentiment_df):
    """
    Merge trader data with sentiment data based on dates.
    
    Args:
        trader_df (pd.DataFrame): Preprocessed trader data
        sentiment_df (pd.DataFrame): Preprocessed sentiment data
    
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("Starting data merging...")
    
    # Ensure both dataframes have date columns
    if 'date' not in trader_df.columns:
        trader_df['date'] = pd.to_datetime(trader_df['datetime']).dt.date
    if 'date' not in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['datetime']).dt.date
        
    # Convert date columns to datetime for consistent merging
    trader_df['merge_date'] = pd.to_datetime(trader_df['date'])
    sentiment_df['merge_date'] = pd.to_datetime(sentiment_df['date'])
    
    # Merge on date
    merged_df = trader_df.merge(
        sentiment_df[['merge_date', 'value', 'classification', 'sentiment_binary', 
                     'sentiment_score_normalized', 'sentiment_category']],
        on='merge_date',
        how='left'
    )
    
    # Check merge results
    merged_count = len(merged_df)
    sentiment_matched = merged_df['classification'].notna().sum()
    
    print(f"Merge completed:")
    print(f"- Total records: {merged_count:,}")
    print(f"- Records with sentiment data: {sentiment_matched:,}")
    print(f"- Records without sentiment data: {merged_count - sentiment_matched:,}")
    
    # Remove records without sentiment data for analysis
    merged_df_clean = merged_df.dropna(subset=['classification']).copy()
    
    print(f"Final merged dataset shape: {merged_df_clean.shape}")
    
    # Basic statistics of merged data
    print("\\nSentiment distribution in merged data:")
    print(merged_df_clean['sentiment_binary'].value_counts())
    
    return merged_df_clean

def calculate_performance_metrics(df):
    """
    Calculate additional performance metrics for analysis.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        pd.DataFrame: Dataset with additional performance metrics
    """
    print("Calculating performance metrics...")
    
    # Group by account and calculate metrics
    account_metrics = df.groupby('account').agg({
        'closed_pnl': ['sum', 'mean', 'std', 'count'],
        'size_usd': ['sum', 'mean'],
        'is_profitable': 'mean',
        'leverage_approx': 'mean'
    }).round(4)
    
    # Flatten column names
    account_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                              for col in account_metrics.columns]
    
    # Calculate win rate and other metrics
    account_metrics['win_rate'] = account_metrics['is_profitable_mean']
    account_metrics['total_pnl'] = account_metrics['closed_pnl_sum']
    account_metrics['avg_pnl'] = account_metrics['closed_pnl_mean']
    account_metrics['trade_count'] = account_metrics['closed_pnl_count']
    account_metrics['total_volume'] = account_metrics['size_usd_sum']
    
    # Reset index to make account a column
    account_metrics = account_metrics.reset_index()
    
    # Merge back with original data
    df_enhanced = df.merge(
        account_metrics[['account', 'win_rate', 'total_pnl', 'avg_pnl', 
                        'trade_count', 'total_volume']],
        on='account',
        how='left'
    )
    
    print(f"Performance metrics calculated for {len(account_metrics)} unique accounts")
    
    return df_enhanced, account_metrics

def save_processed_data(df, filename):
    """
    Save processed data to CSV file.
    
    Args:
        df (pd.DataFrame): Data to save
        filename (str): Output filename
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main_preprocessing():
    """
    Main preprocessing pipeline.
    """
    print("=== Starting Data Preprocessing Pipeline ===\\n")
    
    # Load raw data
    from data_utils import load_trader_data, load_sentiment_data
    
    trader_raw = load_trader_data('data/trader_data.csv')
    sentiment_raw = load_sentiment_data('data/sentiment_data.csv')
    
    if trader_raw is None or sentiment_raw is None:
        print("Error: Could not load raw data files")
        return None, None, None
    
    # Preprocess individual datasets
    trader_clean = preprocess_trader_data(trader_raw)
    sentiment_clean = preprocess_sentiment_data(sentiment_raw)
    
    # Merge datasets
    merged_data = merge_trader_sentiment_data(trader_clean, sentiment_clean)
    
    # Calculate performance metrics
    enhanced_data, account_metrics = calculate_performance_metrics(merged_data)
    
    # Save processed data
    save_processed_data(enhanced_data, 'data/merged_trader_sentiment_data.csv')
    save_processed_data(account_metrics, 'data/account_performance_metrics.csv')
    
    print("\\n=== Data Preprocessing Pipeline Completed ===")
    
    return enhanced_data, account_metrics, merged_data

if __name__ == "__main__":
    main_preprocessing()