# Forex Price Movement Prediction Dashboard

This project implements a comprehensive dashboard for forex price movement analysis and prediction using Alpha Vantage data and Facebook Prophet time series forecasting.

## Features

- Historical forex data fetching using Alpha Vantage API
- Interactive Streamlit dashboard with technical indicators
- Time series forecasting using Facebook Prophet
- Technical indicators: RSI, MACD, Bollinger Bands
- Trading signals generation with confidence levels
- Prophet-enhanced trading signals with entry, take profit and stop loss
- Advanced data visualization with Plotly
- Data download as CSV
- Easy one-click forecasting

## Prerequisites

- Python 3.9+
- Streamlit
- Pandas
- Facebook Prophet
- Plotly
- Alpha Vantage API key

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd quant-model-for-forex-predict-prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Alpha Vantage API key:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

You can get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

## Usage

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

This will launch the forex prediction dashboard in your browser.

## Dashboard Features

### 1. Data Selection and Visualization
- Select currency pairs (EUR/USD, GBP/USD, etc.)
- Choose date range (up to 2 years of historical data)
- View candlestick charts and price statistics

### 2. Technical Indicators
- **RSI (Relative Strength Index)**: Helps identify overbought or oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Shows momentum trends
- **Bollinger Bands**: Display volatility and potential price breakouts

### 3. Price Forecasting with Facebook Prophet
The dashboard uses Facebook Prophet for time series forecasting with these features:

- **Price Selection**: Choose which price to forecast (Open, High, Low, Close)
- **Simplified Model**: Uses only the essential price data for reliable forecasting
- **7-Day Forecast**: Provides a week-ahead price prediction with confidence intervals
- **Forecast Metrics**: Displays current price, predicted price, and confidence level
- **Interactive Charts**: Visualize historical data alongside forecasts

### 4. Trading Signals
The application offers two types of trading signals:

#### Technical Indicator Signals
- Based on RSI, MACD, and Bollinger Bands
- Entry price recommendations derived from current price levels
- Take profit and stop loss levels calculated using ATR (Average True Range)
- Confidence metrics for each signal
- Risk-reward ratio calculations
- Signal history tracking

#### Prophet-Enhanced Trading Signals
- Combines technical indicators with Prophet forecasts
- Entry, take profit, and stop loss levels optimized by forecast data
- Directional probability based on forecast confidence intervals
- Signal confidence metrics that factor in both technical and forecast data
- Detailed signal analysis comparing technical and forecast indicators
- Enhanced pre-trade checklist

## Facebook Prophet Model

The implementation uses Facebook Prophet's time series forecasting capabilities:

### How It Works
1. **Data Preparation**: Historical price data is cleaned and formatted for Prophet
2. **Model Training**: Prophet identifies patterns in the data, including:
   - Trend component
   - Seasonality (daily, weekly, yearly)
   - Holiday effects
3. **Forecasting**: The model generates predictions with uncertainty intervals

### Model Features
- **Simplified Input**: Uses a single price column (Close, Open, High, or Low)
- **Robust Data Handling**: Automatically handles missing values and outliers
- **Confidence Intervals**: Shows prediction uncertainty

## Best Practices for Accurate Forecasts

1. **Use Sufficient Data**: Select at least 60 days of historical data
2. **Choose the Right Currency Pair**: Some pairs have more predictable patterns
3. **Consider Timeframe**: Daily data usually works best with Prophet
4. **Check Technical Indicators**: Use indicators alongside forecasts for confirmation

## API Usage Notes

- Alpha Vantage free tier allows 5 API calls per minute
- Daily limit of 500 API calls
- Data for most major currency pairs is available

## Limitations

- Forecasts are based on historical patterns and may not predict unexpected events
- The free Alpha Vantage API has rate limits
- Intraday data is limited to the most recent periods

## Future Enhancements

Potential future enhancements could include:
- Support for more technical indicators
- Integration with news sentiment analysis
- Backtesting of trading strategies
- Portfolio optimization
- Real-time alerts for trading signals

## License

[MIT License](LICENSE)

## Acknowledgements

- Alpha Vantage for providing the forex data API
- Facebook for the Prophet forecasting library
- Streamlit for the interactive dashboard framework
