import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def get_major_currency_pairs():
    """Get list of major currency pairs"""
    base_currencies = ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
    pairs = []
    
    # Generate all possible combinations
    for base in base_currencies:
        for quote in base_currencies:
            if base != quote:
                pairs.append(f"{base}{quote}=X")
    
    return pairs

def get_exotic_pairs():
    """Get list of some common exotic currency pairs"""
    exotic_pairs = [
        # Asian currencies
        "SGDUSD=X", "HKDUSD=X", "CNHUSD=X", "THBUSD=X", "KRWUSD=X",
        # European crosses
        "SEKUSD=X", "NOKUSD=X", "DKKUSD=X", "PLNUSD=X", "HUFUSD=X",
        # Other major crosses
        "MXNUSD=X", "ZARUSD=X", "TRYUSD=X", "BRLUSD=X", "RUBUSD=X"
    ]
    return exotic_pairs

def test_symbol(symbol):
    """Test if a symbol is valid and returns data"""
    try:
        # Try to get just one day of data
        data = yf.download(
            symbol,
            start=datetime.now() - timedelta(days=1),
            end=datetime.now(),
            progress=False
        )
        
        if not data.empty:
            return True, len(data)
        return False, 0
        
    except Exception as e:
        return False, str(e)

def main():
    print("Testing Forex Symbols Availability")
    print("=" * 50)
    
    # Get all pairs to test
    major_pairs = get_major_currency_pairs()
    exotic_pairs = get_exotic_pairs()
    all_pairs = major_pairs + exotic_pairs
    
    results = []
    
    print(f"\nTesting {len(all_pairs)} forex pairs...")
    
    for symbol in all_pairs:
        print(f"\nTesting {symbol}...", end=" ")
        success, info = test_symbol(symbol)
        
        if success:
            print("✓ Available")
            results.append({
                'Symbol': symbol,
                'Status': 'Available',
                'Data Points': info
            })
        else:
            print("✗ Not available")
            results.append({
                'Symbol': symbol,
                'Status': 'Not Available',
                'Error': info
            })
        
        time.sleep(1)  # Avoid rate limiting
    
    # Create a DataFrame with results
    df = pd.DataFrame(results)
    
    # Display summary
    print("\nSummary:")
    print("=" * 50)
    print(f"\nTotal pairs tested: {len(df)}")
    print(f"Available pairs: {len(df[df['Status'] == 'Available'])}")
    print(f"Unavailable pairs: {len(df[df['Status'] == 'Not Available'])}")
    
    # Display working pairs
    working_pairs = df[df['Status'] == 'Available']
    if not working_pairs.empty:
        print("\nWorking Pairs:")
        print("=" * 50)
        print(working_pairs.to_string(index=False))
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"forex_symbols_test_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nDetailed results saved to {filename}")

if __name__ == "__main__":
    main() 