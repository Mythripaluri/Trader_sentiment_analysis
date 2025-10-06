#!/usr/bin/env python3
"""
Script to download datasets from Google Drive for the trader sentiment analysis project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import download_file_from_gdrive, load_trader_data, load_sentiment_data, initial_data_inspection

def main():
    # URLs provided in the prompt
    trader_data_url = "https://drive.google.com/uc?export=download&id=1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs"
    sentiment_data_url = "https://drive.google.com/uc?export=download&id=1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf"
    
    # Destination paths
    trader_data_path = "data/trader_data.csv"
    sentiment_data_path = "data/sentiment_data.csv"
    
    print("Starting data download...")
    
    # Download trader data
    print("\n1. Downloading trader data...")
    if download_file_from_gdrive(trader_data_url, trader_data_path):
        print("✅ Trader data downloaded successfully")
        
        # Load and inspect trader data
        trader_df = load_trader_data(trader_data_path)
        if trader_df is not None:
            initial_data_inspection(trader_df, "Trader Data")
    else:
        print("❌ Failed to download trader data")
    
    # Download sentiment data
    print("\n2. Downloading sentiment data...")
    if download_file_from_gdrive(sentiment_data_url, sentiment_data_path):
        print("✅ Sentiment data downloaded successfully")
        
        # Load and inspect sentiment data
        sentiment_df = load_sentiment_data(sentiment_data_path)
        if sentiment_df is not None:
            initial_data_inspection(sentiment_df, "Sentiment Data")
    else:
        print("❌ Failed to download sentiment data")
    
    print("\n🎉 Data download and inspection completed!")

if __name__ == "__main__":
    main()