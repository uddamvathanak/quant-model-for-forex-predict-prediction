import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_collector import ForexDataCollector
import numpy as np

# Initialize session state to track if prophet initialization was successful
if 'prophet_import_success' not in st.session_state:
    try:
        from prophet_predictor import ProphetPredictor
        st.session_state.prophet_import_success = True
        st.session_state.prophet_error = None
    except Exception as e:
        st.session_state.prophet_import_success = False
        st.session_state.prophet_error = str(e)

# Page config
st.set_page_config(
    page_title="Forex Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Forex Prediction Dashboard")
st.markdown("""
This dashboard uses Facebook Prophet to predict forex price movements using Alpha Vantage data.
Default analysis period is 100 days.
""")

# Sidebar
st.sidebar.header("Settings")

# Currency pair selection
currency_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD"
]
selected_pair = st.sidebar.selectbox(
    "Select Currency Pair",
    currency_pairs
)

# Timeframe selection - only daily for simplicity
selected_timeframe = "daily"

# Date range selection - increase the default range
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Default to 1 year instead of 100 days

# Add date range selector with a limit of 2 years
min_date = end_date - timedelta(days=730)  # Allow up to 2 years historical data

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(start_date, end_date),
    min_value=min_date,
    max_value=end_date,
    help="For reliable forecasting, select at least 60 days of data. More data generally improves forecast accuracy."
)

# Add API usage info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### API Usage Notes
- Using Alpha Vantage API
- Free tier allows 5 API calls per minute
- 500 API calls per day
- Daily data is available for long periods
- Requires API key in .env file
""")

# Initialize data collector and predictor
@st.cache_resource(show_spinner=False)
def get_collectors(pair, timeframe):
    try:
        collector = ForexDataCollector(
            currency_pair=pair,
            interval=timeframe
        )
        
        # Only initialize prophet if it was successfully imported
        if st.session_state.prophet_import_success:
            from prophet_predictor import ProphetPredictor
            predictor = ProphetPredictor(prediction_horizon=7)  # Predict 7 days ahead
        else:
            predictor = None
            
        return collector, predictor
    except Exception as e:
        st.error(f"Error initializing collectors: {str(e)}")
        if "Alpha Vantage API key not found" in str(e):
            st.error("""
            Please add your Alpha Vantage API key to your .env file:
            ALPHA_VANTAGE_API_KEY=your_api_key
            
            You can get a free API key at: https://www.alphavantage.co/support/#api-key
            """)
        return None, None

collector, predictor = get_collectors(selected_pair, selected_timeframe)

# Fetch data with progress indicator
@st.cache_data(ttl=300, max_entries=10, show_spinner=False)
def fetch_data(pair, timeframe, start_date, end_date):
    collector, predictor = get_collectors(pair, timeframe)
    if collector is None:
        return pd.DataFrame()
        
    try:        
        data = collector.fetch_forex_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Check if we got enough data
        if data.empty:
            st.error("No data returned from API for the selected date range.")
            return pd.DataFrame()
            
        # Data validation and cleanup
        # First, print data types to check if 'Close' is actually a numeric column
        print(f"Data types before validation: {data.dtypes}")
        
        # Check if Close column is numeric - if not, try to convert
        if not pd.api.types.is_numeric_dtype(data['Close']):
            print(f"Close column is not numeric. Attempting to convert. Sample values: {data['Close'].head()}")
            # Try to convert to numeric, coercing errors to NaN
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        
        # Log data quality after ensuring column is numeric
        nan_count = data['Close'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in Close column")
            # Show some examples of rows with NaN values for debugging
            if nan_count > 0:
                nan_indices = data.index[data['Close'].isna()]
                print(f"First 5 indices with NaN in Close: {nan_indices[:5]}")
                if len(nan_indices) > 0:
                    print(f"Sample rows with NaN Close values:\n{data.loc[nan_indices[:5]]}")
            
            if nan_count > len(data) / 2:
                st.warning(f"Data quality issue: {nan_count} missing values out of {len(data)} data points.")
        
        return data
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        if "API limit" in str(e):
            st.warning("Please wait a minute before trying again (5 calls/minute limit).")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error during data fetching: {str(e)}")
        return pd.DataFrame()

try:
    with st.spinner(f"Fetching {selected_pair} data..."):
        data = fetch_data(selected_pair, selected_timeframe, date_range[0], date_range[1])
        
    if data.empty:
        st.warning("""
        No data available for the selected period. This might be due to:
        - Selected date range is too short
        - Market was closed during selected period
        - API rate limit reached
        
        Try:
        - Using a different date range
        - Waiting a minute before trying again
        - Selecting a different currency pair
        """)
        st.stop()
    
    # Add data info
    st.info(f"Showing {selected_pair} data from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')} ({len(data)} data points)")
    
    # Add download button after data info
    if not data.empty:
        # Create a download button
        csv = data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"forex_data_{selected_pair}_{date_range[0].strftime('%Y%m%d')}_{date_range[1].strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download the displayed data as a CSV file"
        )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Price Chart", "Technical Indicators", "Predictions"])
    
    with tab1:
        st.subheader("Price Chart")
        
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        fig.update_layout(
            title=f"{selected_pair} Price Chart",
            yaxis_title="Price",
            xaxis_title="Date",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"{data['Close'].iloc[-1]:.4f}")
        with col2:
            daily_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
            st.metric("Daily Return", f"{daily_return:.2f}%")
        with col3:
            volatility = data['Close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        with col4:
            trend = "Bullish 📈" if data['Close'].iloc[-1] > data['Close'].iloc[-20] else "Bearish 📉"
            st.metric("Trend", trend)
    
    with tab2:
        st.subheader("Technical Indicators")
        
        # Technical indicators plots
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Plot
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name="RSI"
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(
                title="Relative Strength Index (RSI)",
                yaxis_title="RSI",
                template="plotly_dark"
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # MACD Plot
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                name="MACD"
            ))
            fig_macd.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                name="Signal"
            ))
            fig_macd.update_layout(
                title="MACD",
                yaxis_title="Value",
                template="plotly_dark"
            )
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name="Price"
        ))
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['Bollinger_Upper'],
            name="Upper Band",
            line=dict(dash="dash")
        ))
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['Bollinger_Lower'],
            name="Lower Band",
            line=dict(dash="dash")
        ))
        fig_bb.update_layout(
            title="Bollinger Bands",
            yaxis_title="Price",
            template="plotly_dark"
        )
        st.plotly_chart(fig_bb, use_container_width=True)
    
    with tab3:
        st.subheader("Price Predictions with Prophet")
        
        # Check if Prophet was successfully imported
        if not st.session_state.prophet_import_success:
            st.error(f"Prophet initialization failed: {st.session_state.prophet_error}")
            st.warning("""
            There was an issue initializing Prophet. This is likely due to a dependency conflict.
            
            To fix this issue, try running these commands:
            ```bash
            pip uninstall -y prophet pystan holidays
            pip install prophet==1.1.1 holidays==0.21.13
            ```
            
            Common issues include:
            1. Incompatible holidays package (need version < 2.0.0)
            2. Missing or incompatible pystan
            3. Python version incompatibility
            """)
        elif predictor is None:
            st.error("Prophet predictor could not be initialized.")
        else:
            # Add price column selection
            price_column = st.selectbox(
                "Select Price Data to Forecast",
                options=["Close", "Open", "High", "Low"],
                index=0,  # Default to Close price
                help="Choose which price data to use for forecasting"
            )
            
            # Add note about the simplified model
            st.info("""
            📊 **Simplified Model**: The forecast uses only the selected price column (no additional features) to avoid NaN value issues.
            """)
            
            if st.button("Generate Prophet Forecast"):
                with st.spinner(f"Generating forecast for {price_column} prices..."):
                    try:
                        # Create a copy of the data to avoid modifying the original
                        forecast_data = data.copy()
                        
                        # Detailed data diagnostics
                        with st.expander("Data Diagnostics", expanded=False):
                            st.write("Data Types:", forecast_data.dtypes)
                            st.write("Data Shape:", forecast_data.shape)
                            
                            # Check for NaN values in selected column
                            nan_count = forecast_data[price_column].isna().sum()
                            st.write(f"NaN values in {price_column}: {nan_count} ({nan_count/len(forecast_data):.1%} of data)")
                            
                            # If NaN values exist, show sample rows
                            if nan_count > 0:
                                nan_rows = forecast_data[forecast_data[price_column].isna()].head(5)
                                if not nan_rows.empty:
                                    st.write("Sample rows with NaN values:")
                                    st.write(nan_rows)
                        
                        # Ensure numeric data type for selected price column
                        if not pd.api.types.is_numeric_dtype(forecast_data[price_column]):
                            st.warning(f"{price_column} column is not numeric. Converting to numeric type.")
                            forecast_data[price_column] = pd.to_numeric(forecast_data[price_column], errors='coerce')
                        
                        # Handle any NaN values in the selected price column
                        nan_count = forecast_data[price_column].isna().sum()
                        if nan_count > 0:
                            st.warning(f"Found {nan_count} NaN values in {price_column} column ({nan_count/len(forecast_data):.1%} of data)")
                            
                            # Fill NaN values in the selected price column
                            forecast_data[price_column] = forecast_data[price_column].fillna(method='ffill').fillna(method='bfill')
                            st.info(f"NaN values in {price_column} have been filled using forward/backward fill")
                        
                        # Prepare simple data for Prophet with only the essential columns
                        prophet_data = pd.DataFrame({
                            'ds': forecast_data.index,  # The datetime index
                            'y': forecast_data[price_column].values  # Get the values directly as an array
                        })
                        
                        # Specifically check for None/NaN values in the 'y' column
                        none_mask = prophet_data['y'].isna() | (prophet_data['y'] == None)
                        num_none_values = none_mask.sum()
                        
                        # If None values are detected, show diagnostics and try to fix
                        if num_none_values > 0:
                            st.warning(f"Found {num_none_values} None/NaN values in the 'y' column.")
                            
                            # Show the problematic rows
                            st.write("Sample of rows with None values:")
                            st.write(prophet_data[none_mask].head())
                            
                            # Show the corresponding rows in the original data
                            st.write("Corresponding rows in the original data:")
                            none_indices = forecast_data.index[none_mask]
                            st.write(forecast_data.loc[none_indices].head())
                            
                            # Try to repair the data by using a simpler direct copy
                            st.info("Attempting to fix by copying data directly...")
                            
                            # Create a fresh copy using a different method
                            fixed_data = pd.DataFrame()
                            fixed_data['ds'] = pd.to_datetime(forecast_data.index)
                            # Convert to numeric and handle errors by filling with the column mean
                            y_values = pd.to_numeric(forecast_data[price_column], errors='coerce')
                            # Calculate the mean of non-NaN values
                            y_mean = y_values[~y_values.isna()].mean()
                            # Fill NaN with the mean
                            fixed_data['y'] = y_values.fillna(y_mean)
                            
                            st.write("Fixed data (NaN values replaced with mean):")
                            st.write(fixed_data.head())
                            
                            # Use the fixed data
                            prophet_data = fixed_data
                            st.success(f"Data has been fixed. Using {len(prophet_data)} valid data points.")
                        else:
                            st.success(f"No None values detected in the 'y' column. All {len(prophet_data)} data points are valid.")
                        
                        # Validate datetime index
                        st.write("Validating data before forecasting...")
                        
                        # Validate price column
                        st.write(f"Validating {price_column} column:")
                        st.write(f"Data type: {prophet_data['y'].dtype}")
                        st.write(f"Range: {prophet_data['y'].min()} to {prophet_data['y'].max()}")
                        st.write(f"NaN count: {prophet_data['y'].isna().sum()}")
                        
                        # Show the first and last few rows of the prepared data
                        st.write("First few rows of prepared data:")
                        st.write(prophet_data.head())
                        st.write("Last few rows of prepared data:")
                        st.write(prophet_data.tail())

                        # Final check for any remaining NaN values
                        if prophet_data['y'].isna().sum() > 0:
                            st.error(f"Still found {prophet_data['y'].isna().sum()} NaN values in the forecast data after filling.")
                            st.error("Please select a different price column or date range.")
                            st.stop()
                        
                        # Check if we have enough data
                        if len(prophet_data) < 2:
                            st.error(f"Not enough valid data points for forecasting. Only {len(prophet_data)} valid points found.")
                            st.error("Please select a larger date range or a different currency pair.")
                            st.stop()
                        
                        # Make a clean copy of the data for Prophet
                        final_prophet_data = prophet_data.copy()
                        st.success(f"Data validation complete. Ready to forecast with {len(final_prophet_data)} data points.")

                        # Train Prophet model with only essential data
                        try:
                            st.info("Training Prophet model...")
                            predictor.train(final_prophet_data)
                            st.success("Prophet model trained successfully")
                            
                            # Make prediction for next 7 days
                            st.info("Generating forecast...")
                            forecast = predictor.predict(final_prophet_data, periods=7)
                            st.success("Forecast generated successfully")
                            
                            # Extract forecast data
                            forecast_dates = forecast['ds'].iloc[-7:] 
                            forecast_values = forecast['yhat'].iloc[-7:]
                            lower_bound = forecast['yhat_lower'].iloc[-7:]
                            upper_bound = forecast['yhat_upper'].iloc[-7:]
                        except Exception as e:
                            st.error(f"Error during Prophet model training or prediction: {str(e)}")
                            
                            # Show detailed error information for debugging
                            with st.expander("Detailed Error Information", expanded=True):
                                st.write("Exception details:", str(e))
                                st.write("Data shape:", final_prophet_data.shape)
                                st.write("Data types:", final_prophet_data.dtypes)
                                st.write("First few rows:", final_prophet_data.head())
                                st.write("Last few rows:", final_prophet_data.tail())
                                
                                # Check specifically for None values again
                                st.write("None check in ds:", (final_prophet_data['ds'] == None).sum())
                                st.write("None check in y:", (final_prophet_data['y'] == None).sum())
                                
                                # Count NaN values specifically
                                st.write("NaN count in ds:", final_prophet_data['ds'].isna().sum())
                                st.write("NaN count in y:", final_prophet_data['y'].isna().sum())
                                
                                # As a last resort, try a simpler model with minimal data
                                st.info("Attempting to create a minimal test dataset...")
                                
                                # Create a very simple dataset with just 3 points
                                test_data = pd.DataFrame({
                                    'ds': pd.date_range(start='2023-01-01', periods=3),
                                    'y': [1.0, 2.0, 3.0]
                                })
                                
                                st.write("Test data:", test_data)
                                
                                try:
                                    st.info("Testing Prophet with minimal dataset...")
                                    predictor.train(test_data)
                                    test_forecast = predictor.predict(test_data, periods=3)
                                    st.success("Test forecast successful! The issue is with your specific data.")
                                    st.write("Test forecast head:", test_forecast.head())
                                except Exception as test_e:
                                    st.error(f"Test also failed with error: {str(test_e)}")
                                    st.warning("There may be an issue with the Prophet installation itself.")
                        
                        # Create forecast plot
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[price_column],
                            name="Historical",
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_values,
                            name="Forecast",
                            line=dict(color='red')
                        ))
                        
                        # Add confidence interval
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_dates, forecast_dates[::-1]]),
                            y=pd.concat([upper_bound, lower_bound[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,0,0,0)'),
                            hoverinfo="skip",
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_pair} {price_column} Price Forecast (Next 7 Days)",
                            yaxis_title=f"{price_column} Price",
                            xaxis_title="Date",
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast metrics
                        current_price = data[price_column].iloc[-1]
                        next_day_pred = forecast['yhat'].iloc[-7]
                        pred_change = ((next_day_pred / current_price) - 1) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                f"Current {price_column}",
                                f"{current_price:.4f}"
                            )
                        
                        with col2:
                            st.metric(
                                f"Next Day {price_column} Forecast",
                                f"{next_day_pred:.4f}",
                                delta=f"{pred_change:.2f}%"
                            )
                        
                        with col3:
                            confidence = predictor.get_confidence_score(forecast, current_price)
                            st.metric(
                                "Forecast Confidence",
                                f"{confidence:.1%}"
                            )
                        
                        # Show forecast table
                        st.subheader("7-Day Forecast")
                        forecast_table = pd.DataFrame({
                            'Date': forecast_dates.dt.date,
                            'Forecast': forecast_values.values.round(4),
                            'Lower Bound': lower_bound.values.round(4),
                            'Upper Bound': upper_bound.values.round(4)
                        })
                        st.dataframe(forecast_table)
                        
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e) if str(e) else 'Unknown error'}")
                        st.info("""
                        Forecast failed. This can happen due to:
                        - Data issues (missing values, insufficient data points)
                        - Date range too small (need at least 2 days of data)
                        - Prophet installation issues
                        
                        Try:
                        - Selecting a larger date range
                        - Checking that Prophet is properly installed
                        - Using a different currency pair
                        """)
                        
                        # Show additional debugging info
                        if st.checkbox("Show debugging info"):
                            st.write("Data shape:", data.shape)
                            st.write(f"NaN values in {price_column} column:", data[price_column].isna().sum())
                            st.write("First few rows of data:")
                            st.write(data.head())
            else:
                st.info("Click 'Generate Prophet Forecast' to create a 7-day price prediction using Facebook Prophet.")

            # Trading signals section
            st.subheader("Trading Signals")
            
            # Check if signals exist in the data
            if 'Signal' not in data.columns:
                st.warning("Trading signals could not be calculated. Please ensure your data has technical indicators.")
            elif (data['Signal'] != 0).sum() == 0:
                st.info("No trading signals detected in the current date range. Try selecting a different period or currency pair.")
            else:
                # Get the latest signal
                latest_data = data.iloc[-1]
                signal = latest_data['Signal']
                confidence = latest_data['Confidence']
                
                # Display current trading signal
                if signal != 0 and confidence >= 0.8:
                    signal_type = "BUY 🟢" if signal == 1 else "SELL 🔴"
                    st.success(f"### Strong {signal_type} Signal Detected!")
                    
                    # Display entry and exit points
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Entry Price",
                            f"{latest_data['Entry_Price']:.4f}",
                            help="Recommended entry price for the trade"
                        )
                    with col2:
                        st.metric(
                            "Take Profit",
                            f"{latest_data['Take_Profit']:.4f}",
                            delta=f"{(latest_data['Take_Profit'] - latest_data['Entry_Price']):.4f}",
                            help="Recommended take profit level"
                        )
                    with col3:
                        st.metric(
                            "Stop Loss",
                            f"{latest_data['Stop_Loss']:.4f}",
                            delta=f"{(latest_data['Stop_Loss'] - latest_data['Entry_Price']):.4f}",
                            help="Recommended stop loss level"
                        )
                    
                    # Display confidence and risk metrics
                    st.info(f"Signal Confidence: {confidence:.1%}")
                    
                    # Calculate risk-reward ratio
                    if signal == 1:  # Buy signal
                        risk = latest_data['Entry_Price'] - latest_data['Stop_Loss']
                        reward = latest_data['Take_Profit'] - latest_data['Entry_Price']
                    else:  # Sell signal
                        risk = latest_data['Stop_Loss'] - latest_data['Entry_Price']
                        reward = latest_data['Entry_Price'] - latest_data['Take_Profit']
                    
                    risk_reward = abs(reward / risk) if risk != 0 else 0
                    st.metric("Risk-Reward Ratio", f"{risk_reward:.2f}", help="Ratio of potential reward to risk")
                    
                    # Trading checklist
                    st.markdown("### Pre-Trade Checklist")
                    checklist = """
                    - [ ] Check current market conditions
                    - [ ] Verify position size based on risk management
                    - [ ] Set stop loss and take profit orders
                    - [ ] Check for upcoming news events
                    - [ ] Ensure sufficient margin/balance
                    """
                    st.markdown(checklist)
                else:
                    st.warning("No high-confidence trading signals at the moment. Wait for better opportunities.")
                
                # Display signal history
                st.subheader("Recent Signals")
                signal_history = data[data['Signal'] != 0].tail(5)[['Signal', 'Confidence', 'Entry_Price', 'Take_Profit', 'Stop_Loss']]
                if not signal_history.empty:
                    signal_history['Signal'] = signal_history['Signal'].map({1: 'BUY 🟢', -1: 'SELL 🔴'})
                    signal_history['Confidence'] = signal_history['Confidence'].map('{:.1%}'.format)
                    st.dataframe(signal_history)
                else:
                    st.info("No recent trading signals in the selected period.")
                
                # Add refresh button
                if st.button("🔄 Refresh Signals"):
                    st.cache_data.clear()
                    st.rerun()

except Exception as e:
    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Data provided by Alpha Vantage | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True) 