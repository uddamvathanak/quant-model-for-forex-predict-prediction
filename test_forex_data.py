import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import pandas_datareader as pdr

def test_direct_yfinance():
    """Test direct yfinance download"""
    print("\nTesting direct yfinance download:")
    print("=" * 50)
    
    # Test symbols in different formats
    test_cases = [
        {"symbol": "USDJPY=X", "description": "Standard forex pair"},
        {"symbol": "EUR=X", "description": "Base currency only"},
        {"symbol": "6E=F", "description": "Euro futures"},
        {"symbol": "EURUSD%3DX", "description": "URL encoded"},
        {"symbol": "EURUSD.FX", "description": "FX suffix"}
    ]
    
    for test in test_cases:
        symbol = test["symbol"]
        print(f"\nTesting {symbol} ({test['description']}):")
        
        try:
            # Try to get data
            data = yf.download(
                symbol,
                start=datetime.now() - timedelta(days=5),
                end=datetime.now(),
                interval='1h',
                progress=False
            )
            
            if not data.empty:
                print(f"✓ Success! Found {len(data)} data points")
                print("\nSample data:")
                print(data.head(3))
                print("\nColumns:", data.columns.tolist())
            else:
                print("✗ No data received")
                
        except Exception as e:
            print(f"✗ Error: {type(e).__name__} - {str(e)}")
        
        time.sleep(2)  # Avoid rate limiting

def test_ticker_info():
    """Test getting ticker information"""
    print("\nTesting ticker information:")
    print("=" * 50)
    
    symbol = "EURUSD=X"
    print(f"\nTesting {symbol} info:")
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        print("\nTicker information:")
        for key in sorted(info.keys()):
            print(f"{key}: {info[key]}")
            
    except Exception as e:
        print(f"✗ Error getting ticker info: {type(e).__name__} - {str(e)}")

def test_historical_data():
    """Test getting historical data with different periods"""
    print("\nTesting historical data retrieval:")
    print("=" * 50)
    
    symbol = "USDJPY=X"
    periods = ['1d', '5d', '1mo', '3mo']
    intervals = ['1h', '1d']
    
    for period in periods:
        for interval in intervals:
            print(f"\nTesting {symbol} with period={period}, interval={interval}")
            
            try:
                data = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    progress=False
                )
                
                if not data.empty:
                    print(f"✓ Success! Found {len(data)} data points")
                    print(f"Date range: {data.index[0]} to {data.index[-1]}")
                else:
                    print("✗ No data received")
                    
            except Exception as e:
                print(f"✗ Error: {type(e).__name__} - {str(e)}")
            
            time.sleep(2)  # Avoid rate limiting

def test_pandas_datareader():
    """Test using pandas_datareader as alternative"""
    print("\nTesting pandas_datareader:")
    print("=" * 50)
    
    symbol = "USDJPY=X"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    try:
        data = pdr.get_data_yahoo(
            symbol,
            start=start_date,
            end=end_date
        )
        
        if not data.empty:
            print(f"✓ Success! Found {len(data)} data points")
            print("\nSample data:")
            print(data.head(3))
        else:
            print("✗ No data received")
            
    except Exception as e:
        print(f"✗ Error: {type(e).__name__} - {str(e)}")

def main():
    """Main function to run all tests"""
    print("Starting forex data tests...")
    print("=" * 50)
    
    # Run all tests
    test_direct_yfinance()
    time.sleep(3)
    
    test_ticker_info()
    time.sleep(3)
    
    test_historical_data()
    time.sleep(3)
    
    test_pandas_datareader()
    
    print("\nTests completed.")
    print("=" * 50)

if __name__ == "__main__":
    main() 