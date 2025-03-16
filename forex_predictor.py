import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import optuna
from datetime import datetime, timedelta
import joblib
import requests
import os
from dotenv import load_dotenv

class ForexPredictor:
    def __init__(self, currency_pair="EUR/USD", prediction_horizon=1):
        """
        Initialize the ForexPredictor class
        
        Parameters:
        -----------
        currency_pair : str
            The forex pair to predict (default: "EUR/USD")
            Format should be "BASE/QUOTE" (e.g., "EUR/USD", "GBP/JPY")
        prediction_horizon : int
            Number of days to predict ahead (default: 1)
        """
        self.base_currency, self.quote_currency = currency_pair.split('/')
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        
        # Load Alpha Vantage API key
        load_dotenv()
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not found in .env file")
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch historical forex data from Alpha Vantage
        """
        params = {
            "function": "FX_DAILY",
            "from_symbol": self.base_currency,
            "to_symbol": self.quote_currency,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full"
        }
        
        try:
            response = requests.get("https://www.alphavantage.co/query", params=params)
            data = response.json()
            
            # Get the time series data
            time_series_key = "Time Series FX (Daily)"
            if time_series_key not in data:
                raise ValueError(f"No data received from Alpha Vantage: {data.get('Note', 'Unknown error')}")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            
            # Rename columns
            column_map = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close'
            }
            df = df.rename(columns=column_map)
            
            # Convert values to float
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            return df
            
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {str(e)}")
            return pd.DataFrame()
    
    def create_features(self, data):
        """
        Create technical indicators as features
        """
        if data.empty:
            return data
            
        # Add all technical analysis features
        data = add_all_ta_features(
            data, 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close",
            volume="Volume" if "Volume" in data.columns else None,
            fillna=True
        )
        
        # Create target variable (future returns)
        data['target'] = data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        return data
    
    def prepare_data(self, data):
        """
        Prepare data for training
        """
        # Drop the target column and any non-feature columns
        feature_data = data.drop(['target', 'Open', 'High', 'Low', 'Close'] + 
                               (['Volume'] if 'Volume' in data.columns else []), axis=1)
        
        # Scale the features
        X = self.scaler.fit_transform(feature_data)
        y = data['target'].values
        
        return X, y
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100):
        """
        Optimize LightGBM hyperparameters using Optuna
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            }
            
            model = LGBMRegressor(**params, random_state=42)
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50,
                     verbose=False)
            
            return model.best_score_['valid_0']['l2']
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train(self, data=None, start_date=None, end_date=None, test_size=0.2, val_size=0.2):
        """
        Train the model
        """
        # Fetch data if not provided
        if data is None:
            data = self.fetch_data(start_date, end_date)
        
        # Create features
        data = self.create_features(data)
        
        if data.empty:
            raise ValueError("No data available for training")
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adj, shuffle=False)
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Train final model
        self.model = LGBMRegressor(**best_params, random_state=42)
        self.model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=50,
                      verbose=True)
        
        # Return test set performance
        test_score = self.model.score(X_test, y_test)
        return test_score
    
    def predict(self, data=None):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Fetch latest data if not provided
        if data is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Get last 30 days for features
            data = self.fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        # Create features
        data = self.create_features(data)
        
        if data.empty:
            raise ValueError("No data available for prediction")
        
        # Prepare data
        feature_data = data.drop(['target', 'Open', 'High', 'Low', 'Close'] + 
                               (['Volume'] if 'Volume' in data.columns else []), axis=1)
        X = self.scaler.transform(feature_data)
        
        # Make prediction
        predictions = self.model.predict(X)
        
        return predictions[-1]  # Return the most recent prediction
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model
        """
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler'] 