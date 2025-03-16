from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from data_collector import ForexDataCollector
from prophet_predictor import ProphetPredictor
from signal_generator import SignalGenerator
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Forex Trading Signals API")

# Initialize components
data_collector = ForexDataCollector(currency_pair="EURUSD=X", interval="45m")
prophet_predictor = ProphetPredictor(prediction_horizon=1)
signal_generator = SignalGenerator(confidence_threshold=0.7)

# Model paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "prophet_model.joblib")

# Load pre-trained model if it exists, otherwise train and save a new one
if not os.path.exists(MODEL_PATH):
    # Train and save model on first run
    data = data_collector.fetch_forex_data(
        start_date=(datetime.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    prophet_predictor.train(data)
    prophet_predictor.save_model(MODEL_PATH)
else:
    prophet_predictor.load_model(MODEL_PATH)

class SignalResponse(BaseModel):
    timestamp: str
    current_price: float
    predicted_price: float
    signal: str
    confidence: float
    take_profit: float = None
    stop_loss: float = None
    reason: str

@app.get("/")
async def root():
    return {"message": "Forex Trading Signals API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/signal", response_model=SignalResponse)
async def get_trading_signal():
    try:
        # Fetch latest data
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=30)  # Get last 30 days of data
        
        data = data_collector.fetch_forex_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Process data and add technical indicators
        data = data_collector.detect_breakouts(data)
        
        # Get news sentiment
        news_data = data_collector.fetch_news_data()
        sentiment_score = 0  # TODO: Implement sentiment analysis with DeepSeek
        
        # Generate Prophet forecast using pre-trained model
        forecast = prophet_predictor.predict(data)
        
        # Generate trading signal
        signal = signal_generator.generate_signal(
            data,
            forecast,
            sentiment_score
        )
        
        return SignalResponse(
            timestamp=datetime.now().isoformat(),
            current_price=signal['current_price'],
            predicted_price=signal['predicted_price'],
            signal=signal['signal_type'].name,
            confidence=signal['confidence_score'],
            take_profit=signal['take_profit'],
            stop_loss=signal['stop_loss'],
            reason=signal['reason']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical")
async def get_historical_data(days: int = 30):
    try:
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        data = data_collector.fetch_forex_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Process data to ensure it has required columns
        if 'Weekly_VWAP' not in data.columns:
            data = data_collector._process_data(data)
        
        # Convert DataFrame to records and handle datetime
        records = data.reset_index().to_dict(orient='records')
        
        # Convert datetime objects to ISO format strings
        for record in records:
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif isinstance(value, np.integer):
                    record[key] = int(value)
                elif isinstance(value, np.floating):
                    record[key] = float(value)
        
        return records
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Failed to fetch historical data",
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 