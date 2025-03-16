import numpy as np
import pandas as pd
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
        # Ensure we have the Close column
        if 'Close' not in price_data.columns:
            raise ValueError("Price data must contain 'Close' column")
            
        current_price = price_data['Close'].iloc[-1]
        
        # Check and handle Weekly_VWAP
        if 'Weekly_VWAP' not in price_data.columns:
            print("Weekly_VWAP not found, calculating simple 5-day moving average")
            price_data['Weekly_VWAP'] = price_data['Close'].rolling(window=5).mean()
            # Use the last non-NA value or fall back to current price
            vwap = price_data['Weekly_VWAP'].iloc[-1]
            if pd.isna(vwap):
                vwap = current_price
        else:
            vwap = price_data['Weekly_VWAP'].iloc[-1]
            # If VWAP is NA, use current price
            if pd.isna(vwap):
                vwap = current_price
        
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
        
        # Check for Breakout column
        if 'Breakout' not in price_data.columns:
            print("Breakout column not found, using default value of 0")
            breakout_value = 0
        else:
            breakout_value = price_data['Breakout'].iloc[-1]
            # If Breakout is NA, use 0
            if pd.isna(breakout_value):
                breakout_value = 0
        
        # Generate signal
        signal = self._determine_signal(
            predicted_direction,
            current_price,
            vwap,
            breakout_value,
            sentiment_score,
            total_confidence
        )
        
        # Check for ATR column
        if 'ATR' not in price_data.columns:
            print("ATR column not found, using default value of 2% of current price")
            atr_value = current_price * 0.02  # Default to 2% of current price
        else:
            atr_value = price_data['ATR'].iloc[-1]
            # If ATR is NA, use 2% of current price
            if pd.isna(atr_value):
                atr_value = current_price * 0.02
        
        # Calculate TP/SL levels
        tp_sl_levels = self._calculate_tp_sl(
            signal['signal_type'],
            current_price,
            atr_value
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
        try:
            # Ensure we have Close column
            if 'Close' not in price_data.columns:
                return 0.5  # Default confidence if no data available
            
            current_price = price_data['Close'].iloc[-1]
            
            # VWAP trend strength
            if 'Weekly_VWAP' not in price_data.columns:
                # Use simple moving average as fallback
                vwap = price_data['Close'].rolling(window=5).mean().iloc[-1]
                if pd.isna(vwap):
                    vwap = current_price
            else:
                vwap = price_data['Weekly_VWAP'].iloc[-1]
                if pd.isna(vwap):
                    vwap = current_price
                    
            vwap_trend = abs(current_price - vwap)
            vwap_trend = vwap_trend / vwap if vwap != 0 else 0
            
            # Breakout strength
            if 'Breakout' not in price_data.columns:
                breakout_strength = 0  # No breakout info available
            else:
                breakout_value = price_data['Breakout'].iloc[-1]
                breakout_strength = abs(breakout_value) if not pd.isna(breakout_value) else 0
            
            # Combine factors
            technical_confidence = (vwap_trend + breakout_strength) / 2
            
            return min(technical_confidence, 1.0)
        except Exception as e:
            print(f"Error calculating technical confidence: {str(e)}")
            return 0.5  # Default to neutral confidence on error
    
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