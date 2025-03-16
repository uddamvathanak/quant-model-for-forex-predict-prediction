import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collector import ForexDataCollector
import os
from dotenv import load_dotenv

class TestForexDataCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        load_dotenv()
        cls.collector = ForexDataCollector(
            currency_pair="EURUSD=X",
            interval="45m",
            db_path="sqlite:///test_forex_data.db"
        )
        
        # Test period
        cls.end_date = datetime.now()
        cls.start_date = cls.end_date - timedelta(days=7)
        
    def test_init(self):
        """Test initialization of ForexDataCollector"""
        self.assertEqual(self.collector.currency_pair, "EURUSD=X")
        self.assertEqual(self.collector.interval, "45m")
        self.assertTrue(hasattr(self.collector, 'engine'))
        self.assertTrue(hasattr(self.collector, 'news_api'))
        
    def test_fetch_forex_data(self):
        """Test fetching forex data from Yahoo Finance"""
        # Test with date range
        data = self.collector.fetch_forex_data(
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d")
        )
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(len(data) > 0)
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
            
        # Check index is datetime
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        
    def test_process_data(self):
        """Test data processing functionality"""
        # Get sample data
        data = self.collector.fetch_forex_data(
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d")
        )
        
        # Check technical indicators
        self.assertIn('Weekly_VWAP', data.columns)
        self.assertIn('ATR', data.columns)
        self.assertIn('SMA20', data.columns)
        self.assertIn('SMA50', data.columns)
        self.assertIn('Support', data.columns)
        self.assertIn('Resistance', data.columns)
        
        # Check for NaN values
        self.assertFalse(data.isnull().any().any())
        
    def test_fetch_news_data(self):
        """Test news data fetching"""
        if not os.getenv('NEWS_API_KEY'):
            self.skipTest("NEWS_API_KEY not found in environment")
            
        news_data = self.collector.fetch_news_data()
        
        # Check if we got a DataFrame
        self.assertIsInstance(news_data, pd.DataFrame)
        
        if not news_data.empty:
            # Check required columns
            required_columns = ['title', 'description', 'url', 'content']
            for col in required_columns:
                self.assertIn(col, news_data.columns)
                
            # Check index is datetime
            self.assertIsInstance(news_data.index, pd.DatetimeIndex)
            
    def test_detect_breakouts(self):
        """Test breakout detection"""
        data = self.collector.fetch_forex_data(
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d")
        )
        
        data_with_breakouts = self.collector.detect_breakouts(data)
        
        # Check Breakout column exists
        self.assertIn('Breakout', data_with_breakouts.columns)
        
        # Check Breakout values are valid
        breakout_values = data_with_breakouts['Breakout'].unique()
        self.assertTrue(all(val in [-1, 0, 1] for val in breakout_values))
        
    def test_calculate_tp_sl_levels(self):
        """Test calculation of take profit and stop loss levels"""
        data = self.collector.fetch_forex_data(
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d")
        )
        
        data_with_levels = self.collector.calculate_tp_sl_levels(data)
        
        # Check TP/SL columns exist
        required_columns = ['TP_Long', 'SL_Long', 'TP_Short', 'SL_Short']
        for col in required_columns:
            self.assertIn(col, data_with_levels.columns)
            
        # Verify TP/SL relationships
        self.assertTrue((data_with_levels['TP_Long'] > data_with_levels['Close']).all())
        self.assertTrue((data_with_levels['SL_Long'] < data_with_levels['Close']).all())
        self.assertTrue((data_with_levels['TP_Short'] < data_with_levels['Close']).all())
        self.assertTrue((data_with_levels['SL_Short'] > data_with_levels['Close']).all())
        
    def test_database_operations(self):
        """Test database save and load operations"""
        # Get sample data
        data = self.collector.fetch_forex_data(
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d")
        )
        
        # Test save operation
        self.collector.save_to_database(data, table_name='test_forex_data')
        
        # Test load operation
        loaded_data = self.collector.load_from_database(
            table_name='test_forex_data',
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check loaded data
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertTrue(len(loaded_data) > 0)
        self.assertEqual(set(data.columns), set(loaded_data.columns))
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with invalid date range
        future_date = datetime.now() + timedelta(days=365)
        with self.assertRaises(ValueError):
            self.collector.fetch_forex_data(
                start_date=future_date.strftime("%Y-%m-%d")
            )
            
        # Test with None data
        self.assertTrue(self.collector.detect_breakouts(None).empty)
        self.assertTrue(self.collector.calculate_tp_sl_levels(None).empty)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        self.assertTrue(self.collector.detect_breakouts(empty_df).empty)
        self.assertTrue(self.collector.calculate_tp_sl_levels(empty_df).empty)

if __name__ == '__main__':
    unittest.main() 