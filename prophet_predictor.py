from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# Disable Prophet logging to clean up console output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class ProphetPredictor:
    def __init__(self, prediction_horizon=1):
        """
        Initialize Prophet predictor
        
        Parameters:
        -----------
        prediction_horizon : int
            Number of periods to predict ahead
        """
        try:
            # Initialize with holidays disabled to avoid incompatibility issues
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                holidays=None  # Disable holidays to avoid compatibility issues
            )
        except Exception as e:
            # Fallback to simpler model if initialization fails
            print(f"Error initializing Prophet with full settings: {str(e)}")
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True
            )
            
        self.prediction_horizon = prediction_horizon
        
    def prepare_data(self, data):
        """
        Prepare data for Prophet - simplified version that only uses the price column
        without any additional regressors to avoid NaN issues
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame that either already has 'ds' and 'y' columns
            or has a datetime index and 'Close' column
        """
        print("\n==== PROPHET DATA PREPARATION ====")
        print(f"Input data shape: {data.shape}")
        print(f"Input data columns: {data.columns.tolist()}")
        print(f"Input data types: {data.dtypes}")
        
        # Create a clean DataFrame with only the required columns for Prophet
        prophet_data = pd.DataFrame()
        
        # Set the datetime column
        if 'ds' in data.columns:
            prophet_data['ds'] = data['ds']
            print("Using 'ds' column from input data")
        else:
            prophet_data['ds'] = data.index
            print(f"Using index as 'ds' column, index type: {type(data.index)}")
        
        # Ensure ds is datetime type
        if not pd.api.types.is_datetime64_any_dtype(prophet_data['ds']):
            print(f"WARNING: 'ds' column is not datetime type: {prophet_data['ds'].dtype}")
            try:
                prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                print("Successfully converted 'ds' to datetime")
            except Exception as e:
                print(f"ERROR converting 'ds' to datetime: {str(e)}")
                print(f"First few values of 'ds': {prophet_data['ds'].head()}")
        
        # Handle potential NaT (Not a Time) values in ds column
        nat_count = pd.isna(prophet_data['ds']).sum()
        if nat_count > 0:
            print(f"WARNING: Found {nat_count} NaT values in 'ds' column")
            # Drop rows with NaT in ds since Prophet requires valid datetime
            prophet_data = prophet_data[~pd.isna(prophet_data['ds'])]
            print(f"Dropped rows with NaT values, new shape: {prophet_data.shape}")
        
        # Set the target value column - with special handling for None values
        if 'y' in data.columns:
            # Get values directly as array and convert to numeric
            y_values = pd.to_numeric(data['y'].values, errors='coerce')
            prophet_data['y'] = y_values
            print("Using 'y' column from input data (converted to numeric)")
        else:
            # Extract the target column values directly and convert to numeric
            if 'Close' in data.columns:
                y_values = pd.to_numeric(data['Close'].values, errors='coerce')
                prophet_data['y'] = y_values
                print("Using 'Close' column converted to numeric array")
            else:
                print("ERROR: No 'y' or 'Close' column found in input data")
                return pd.DataFrame()  # Return empty DataFrame if no valid target column
        
        # Check for explicit None values separately from NaN
        none_count = (prophet_data['y'] == None).sum()
        if none_count > 0:
            print(f"WARNING: Found {none_count} explicit None values in 'y' column")
            # Replace None with NaN so they can be handled in the next step
            prophet_data['y'] = prophet_data['y'].replace(None, np.nan)
        
        # Check for NaN values in the y column
        nan_count_before = prophet_data['y'].isna().sum()
        if nan_count_before > 0:
            print(f"WARNING: Found {nan_count_before} NaN values in 'y' column")
            
            # Calculate the mean of valid values for filling
            y_mean = prophet_data['y'].mean()
            if pd.isna(y_mean) or y_mean == 0:  # If mean is also NaN or zero, use a default value
                print("WARNING: Cannot calculate mean for filling. Using default value 1.0")
                y_mean = 1.0
            
            # Fill NaN values with the mean
            prophet_data['y'] = prophet_data['y'].fillna(y_mean)
            print(f"Filled NaN values with mean: {y_mean}")
            
            # Verify all NaN values have been fixed
            nan_count_after = prophet_data['y'].isna().sum()
            print(f"NaN values after filling: {nan_count_after}")
        
        # Ensure we have at least 2 valid data points
        if len(prophet_data) < 2:
            print("ERROR: Less than 2 valid data points after preparation")
            # Return empty DataFrame to signal error
            return pd.DataFrame()
        
        print(f"Final prophet_data shape: {prophet_data.shape}")
        print(f"prophet_data head: \n{prophet_data.head()}")
        print(f"prophet_data tail: \n{prophet_data.tail()}")
        print("==== END OF PROPHET DATA PREPARATION ====\n")
        
        return prophet_data
    
    def train(self, data):
        """
        Train the Prophet model
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with datetime index and price data
        """
        # Prepare data for Prophet - ensuring only essential columns are used
        prophet_data = self.prepare_data(data)
        
        # Check if we have enough data
        if len(prophet_data) < 2:
            raise ValueError("Dataframe has less than 2 non-NaN rows")
        
        # Reset the model to a clean state
        try:
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                holidays=None  # Disable holidays to avoid compatibility issues
            )
        except Exception as e:
            print(f"Error resetting Prophet model: {str(e)}")
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True
            )
        
        # Fit the model with only the essential data (no regressors)
        print(f"Fitting Prophet model with {len(prophet_data)} data points")
        self.model.fit(prophet_data)
        print("Prophet model fitting completed")
        
    def predict(self, data, periods=None):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with datetime index and price data
        periods : int
            Number of periods to forecast ahead
        """
        if periods is None:
            periods = self.prediction_horizon
            
        # Prepare data for Prophet - ensuring only essential columns are used
        prophet_data = self.prepare_data(data)
        
        try:
            # Create future dataframe more carefully
            print(f"Creating future dataframe for {periods} periods ahead")
            
            # Get the last date from our data
            last_date = prophet_data['ds'].max()
            print(f"Last date in data: {last_date}")
            
            # Create a clean date range for future predictions
            try:
                # Create future dates starting from the day after the last date
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=periods,
                    freq='D'
                )
                print(f"Created future dates: {future_dates}")
                
                # Create a dataframe with all historical and future dates
                historical_dates = prophet_data['ds'].sort_values().reset_index(drop=True)
                all_dates = pd.concat([
                    pd.Series(historical_dates),
                    pd.Series(future_dates)
                ]).reset_index(drop=True)
                
                future = pd.DataFrame({'ds': all_dates})
                print(f"Created future dataframe with shape: {future.shape}")
                
            except Exception as e:
                print(f"Error creating custom date range: {str(e)}, falling back to Prophet's method")
                # Fall back to Prophet's built-in method if our approach fails
                future = self.model.make_future_dataframe(
                    periods=periods,
                    freq='D',
                    include_history=True
                )
        except Exception as e:
            print(f"Error preparing future dataframe: {str(e)}, using Prophet's default method")
            # Use Prophet's built-in method as a last resort
            future = self.model.make_future_dataframe(
                periods=periods,
                freq='D',
                include_history=True
            )
        
        # Make predictions with better error handling
        try:
            print(f"Generating forecast for next {periods} periods")
            forecast = self.model.predict(future)
            print("Forecast generation completed successfully")
        except Exception as e:
            print(f"Error during forecast generation: {str(e)}")
            # If we hit an error, try one more approach - use only the 'ds' and none of the regressors
            print("Attempting fallback prediction method...")
            minimal_future = pd.DataFrame({'ds': future['ds']})
            forecast = self.model.predict(minimal_future)
            print("Fallback forecast generation completed")
        
        return forecast
    
    def evaluate(self, data):
        """
        Evaluate model performance
        """
        # Make predictions for the entire dataset
        forecast = self.predict(data)
        
        # Calculate metrics
        y_true = data['Close']
        y_pred = forecast['yhat'][-len(y_true):]
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate directional accuracy
        actual_direction = np.sign(y_true.diff())
        predicted_direction = np.sign(y_pred.diff())
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        return {
            'RMSE': rmse,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    def get_confidence_score(self, forecast, current_price):
        """
        Calculate confidence score based on prediction interval and trend strength
        """
        # Get the latest prediction
        latest_pred = forecast.iloc[-1]
        
        # Calculate confidence based on prediction interval width
        interval_width = latest_pred['yhat_upper'] - latest_pred['yhat_lower']
        interval_confidence = 1 - min((interval_width / current_price), 1.0)
        
        # Calculate trend strength (normalized)
        trend_strength = min(abs(latest_pred['trend'] - current_price) / current_price, 1.0)
        
        # Combine both factors
        confidence_score = (interval_confidence * 0.7) + (trend_strength * 0.3)
        
        return min(max(confidence_score, 0), 1)  # Ensure score is between 0 and 1
    
    def save_model(self, path):
        """
        Save the trained model
        """
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """
        Load a trained model
        """
        self.model = joblib.load(path) 