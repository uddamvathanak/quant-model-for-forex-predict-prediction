import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_forex_data():
    """Test different ways to fetch EUR/USD data"""
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print("\nTesting EUR/USD data fetching...")
    
    # Test different symbol formats
    symbols = [
        "EURUSD=X",
        "EUR=X",
        "EURUSD%3DX",  # URL encoded
        "EURUSD.X",
        "EUR/USD",
        "6E=F"  # Euro FX Futures
    ]
    
    # Test different intervals
    intervals = ["1h", "1d", "1wk"]
    
    best_data = None
    best_symbol = None
    best_interval = None
    max_rows = 0
    
    for symbol in symbols:
        for interval in intervals:
            try:
                print(f"\nTrying symbol: {symbol} with interval: {interval}")
                data = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval=interval,
                    progress=False
                )
                print(f"Data shape: {data.shape}")
                
                if not data.empty and len(data) > max_rows:
                    max_rows = len(data)
                    best_data = data
                    best_symbol = symbol
                    best_interval = interval
                    
                if not data.empty:
                    print("\nFirst few rows:")
                    print(data.head(2))
                    print("\nLast few rows:")
                    print(data.tail(2))
                    
            except Exception as e:
                print(f"Error: {str(e)}")
    
    if best_data is not None:
        print(f"\nBest results achieved with:")
        print(f"Symbol: {best_symbol}")
        print(f"Interval: {best_interval}")
        print(f"Number of rows: {max_rows}")
        return best_data, best_symbol, best_interval
    else:
        print("\nNo successful data fetches")
        return None, None, None

if __name__ == "__main__":
    data, symbol, interval = test_forex_data()
    
    if data is not None:
        print("\nData Info:")
        print(data.info()) 