import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_alpha_vantage():
    """Test Alpha Vantage API for forex data"""
    print("\nTesting Alpha Vantage API:")
    print("=" * 50)
    
    # You can get a free API key from https://www.alphavantage.co/support/#api-key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    if not api_key:
        print("✗ Alpha Vantage API key not found in .env file")
        print("Get a free key from: https://www.alphavantage.co/support/#api-key")
        return None
    
    symbol = 'EUR'
    to_currency = 'USD'
    
    try:
        # Test forex rate endpoint
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={symbol}&to_currency={to_currency}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Realtime Currency Exchange Rate" in data:
            rate_data = data["Realtime Currency Exchange Rate"]
            print("✓ Successfully got real-time exchange rate")
            print(f"Exchange Rate: {rate_data.get('5. Exchange Rate', 'N/A')}")
            print(f"Last Updated: {rate_data.get('6. Last Refreshed', 'N/A')}")
            
            # Test historical data
            url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={symbol}&to_symbol={to_currency}&apikey={api_key}'
            response = requests.get(url)
            data = response.json()
            
            if "Time Series FX (Daily)" in data:
                print("\n✓ Successfully got historical data")
                time_series = data["Time Series FX (Daily)"]
                print(f"Number of historical data points: {len(time_series)}")
                return True
            else:
                print("✗ Could not get historical data")
                return False
        else:
            print("✗ Could not get exchange rate data")
            print("Error message:", data.get("Note", "Unknown error"))
            return False
            
    except Exception as e:
        print(f"✗ Error testing Alpha Vantage: {str(e)}")
        return False

def main():
    """Test Alpha Vantage API"""
    print("Testing Alpha Vantage API for Forex Data")
    print("=" * 50)
    
    success = test_alpha_vantage()
    
    # Summary
    print("\nAPI Test Summary:")
    print("=" * 50)
    status = "✓ Working" if success is True else "✗ Not Working" if success is False else "? Not Tested"
    print(f"Alpha Vantage: {status}")

if __name__ == "__main__":
    main() 