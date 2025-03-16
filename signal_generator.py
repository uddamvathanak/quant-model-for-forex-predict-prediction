import numpy as np
from enum import Enum

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class SignalGenerator:
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize SignalGenerator
        
        Parameters:
        -----------
        confidence_threshold : float
            Minimum confidence level required for generating signals
        """
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(self, price_data, prophet_forecast, sentiment_score=None):
        """
        Generate trading signal based on multiple factors
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Current price data with technical indicators
        prophet_forecast : pd.DataFrame
            Prophet model forecast
        sentiment_score : float
            News sentiment score (-1 to 1)
        
        Returns:
        --------
        dict
            Trading signal information
        """
        current_price = price_data['Close'].iloc[-1]
        vwap = price_data['Weekly_VWAP'].iloc[-1]
        
        # Get predicted direction and confidence from Prophet
        latest_forecast = prophet_forecast.iloc[-1]
        predicted_price = latest_forecast['yhat']
        predicted_direction = np.sign(predicted_price - current_price)
        
        # Calculate confidence scores
        technical_confidence = self._get_technical_confidence(price_data)
        prophet_confidence = self._get_prophet_confidence(latest_forecast, current_price)
        
        # Combine confidence scores
        if sentiment_score is not None:
            total_confidence = (technical_confidence + prophet_confidence + abs(sentiment_score)) / 3
        else:
            total_confidence = (technical_confidence + prophet_confidence) / 2
        
        # Generate signal
        signal = self._determine_signal(
            predicted_direction,
            current_price,
            vwap,
            price_data['Breakout'].iloc[-1],
            sentiment_score,
            total_confidence
        )
        
        # Calculate TP/SL levels
        tp_sl_levels = self._calculate_tp_sl(
            signal['signal_type'],
            current_price,
            price_data['ATR'].iloc[-1]
        )
        
        return {
            **signal,
            **tp_sl_levels,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'confidence_score': total_confidence
        }
    
    def _get_technical_confidence(self, price_data):
        """
        Calculate confidence score based on technical indicators
        """
        # VWAP trend strength
        vwap_trend = abs(price_data['Close'].iloc[-1] - price_data['Weekly_VWAP'].iloc[-1])
        vwap_trend = vwap_trend / price_data['Weekly_VWAP'].iloc[-1]
        
        # Breakout strength
        breakout_strength = abs(price_data['Breakout'].iloc[-1])
        
        # Combine factors
        technical_confidence = (vwap_trend + breakout_strength) / 2
        
        return min(technical_confidence, 1.0)
    
    def _get_prophet_confidence(self, forecast, current_price):
        """
        Calculate confidence score based on Prophet forecast
        """
        # Calculate based on prediction interval width
        interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
        confidence = 1 - (interval_width / current_price)
        
        return max(min(confidence, 1.0), 0.0)
    
    def _determine_signal(self, predicted_direction, current_price, vwap, breakout, sentiment_score, confidence):
        """
        Determine the final trading signal
        """
        if confidence < self.confidence_threshold:
            return {'signal_type': SignalType.HOLD, 'reason': 'Low confidence'}
        
        # Bullish conditions
        if (predicted_direction > 0 and
            current_price > vwap and
            breakout >= 0 and
            (sentiment_score is None or sentiment_score >= 0)):
            return {'signal_type': SignalType.BUY, 'reason': 'Bullish conditions met'}
        
        # Bearish conditions
        elif (predicted_direction < 0 and
              current_price < vwap and
              breakout <= 0 and
              (sentiment_score is None or sentiment_score <= 0)):
            return {'signal_type': SignalType.SELL, 'reason': 'Bearish conditions met'}
        
        return {'signal_type': SignalType.HOLD, 'reason': 'No clear signal'}
    
    def _calculate_tp_sl(self, signal_type, current_price, atr, tp_multiplier=1.5, sl_multiplier=1.0):
        """
        Calculate Take Profit and Stop Loss levels
        """
        if signal_type == SignalType.BUY:
            take_profit = current_price + (atr * tp_multiplier)
            stop_loss = current_price - (atr * sl_multiplier)
        elif signal_type == SignalType.SELL:
            take_profit = current_price - (atr * tp_multiplier)
            stop_loss = current_price + (atr * sl_multiplier)
        else:
            take_profit = None
            stop_loss = None
            
        return {
            'take_profit': take_profit,
            'stop_loss': stop_loss
        } 