from data_collector import ForexDataCollector
from datetime import datetime, timedelta
import pandas as pd

def test_forex_data():
    # Initialize collector
    collector = ForexDataCollector(currency_pair="EUR/USD", interval="60min")
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Fetch data
        print(f"Fetching data from {start_date} to {end_date}")
        data = collector.fetch_forex_data(start_date=start_date, end_date=end_date)
        
        # Print data info
        print("\nData Info:")
        print(f"Shape: {data.shape}")
        print(f"Date Range: {data.index[0]} to {data.index[-1]}")
        print(f"Number of rows: {len(data)}")
        
        # Print first few rows
        print("\nFirst few rows:")
        print(data.head())
        
        return data
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    test_forex_data() 