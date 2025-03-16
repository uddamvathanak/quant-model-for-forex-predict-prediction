import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="Forex Trading Signals Dashboard",
    page_icon="📈",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

def fetch_trading_signal():
    """Fetch latest trading signal from API"""
    try:
        response = requests.get(f"{API_URL}/signal")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching trading signal: {str(e)}")
        return None

def fetch_historical_data(days=30):
    """Fetch historical data from API"""
    try:
        response = requests.get(f"{API_URL}/historical", params={"days": days})
        data = response.json()
        
        # Check if response is empty or invalid
        if not data:
            st.warning("No historical data available")
            return None
            
        # Convert to DataFrame and process dates
        df = pd.DataFrame(data)
        if 'index' in df.columns:
            df['ds'] = pd.to_datetime(df['index'])
            df = df.drop('index', axis=1)
        elif 'date' in df.columns:
            df['ds'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        else:
            df['ds'] = pd.to_datetime(df.index)
            
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Weekly_VWAP']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Missing required columns: {', '.join(missing_columns)}")
            # Add missing columns with default values
            for col in missing_columns:
                df[col] = 0
                
        return df
        
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        st.error("Response content: " + str(response.text if 'response' in locals() else "No response"))
        return None

def plot_forex_chart(data):
    """Create an interactive forex chart with indicators"""
    if data is None or data.empty:
        st.warning("No data available for plotting")
        return None
        
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & VWAP', 'Volume'),
        row_width=[0.7, 0.3]
    )

    try:
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['ds'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # VWAP
        if 'Weekly_VWAP' in data.columns and not data['Weekly_VWAP'].isnull().all():
            fig.add_trace(
                go.Scatter(
                    x=data['ds'],
                    y=data['Weekly_VWAP'],
                    name='VWAP',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )

        # Volume bars
        if 'Volume' in data.columns and not data['Volume'].isnull().all():
            fig.add_trace(
                go.Bar(
                    x=data['ds'],
                    y=data['Volume'],
                    name='Volume'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title='EUR/USD Price Action',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def display_signal_card(signal):
    """Display trading signal in a card format"""
    if not signal:
        st.warning("No signal data available")
        return

    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            "Signal",
            signal.get('signal_type', 'UNKNOWN'),
            delta="Active" if signal.get('signal_type') != "HOLD" else "Neutral"
        )
        
    with cols[1]:
        current_price = signal.get('current_price', 0)
        predicted_price = signal.get('predicted_price', 0)
        st.metric(
            "Current Price",
            f"${current_price:.4f}" if current_price else "N/A",
            delta=f"{(predicted_price - current_price):.4f}" if all([current_price, predicted_price]) else None
        )
        
    with cols[2]:
        confidence = signal.get('confidence_score', 0)
        st.metric(
            "Confidence",
            f"{confidence*100:.1f}%" if confidence else "N/A",
            delta=None
        )
        
    with cols[3]:
        tp = signal.get('take_profit')
        sl = signal.get('stop_loss')
        if tp and sl:
            st.metric(
                "Risk/Reward",
                f"TP: ${tp:.4f}",
                delta=f"SL: ${sl:.4f}"
            )
        else:
            st.metric(
                "Risk/Reward",
                "N/A",
                delta=None
            )

def main():
    st.title("📈 Forex Trading Signals Dashboard")
    
    # Sidebar
    st.sidebar.title("Settings")
    days = st.sidebar.slider("Historical Data (days)", 5, 90, 30)
    update_frequency = st.sidebar.slider("Update Frequency (seconds)", 30, 300, 60)
    
    # Main content
    signal_container = st.container()
    chart_container = st.container()
    
    # Auto-refresh
    if st.sidebar.button("Refresh Data") or 'last_refresh' not in st.session_state:
        with st.spinner("Fetching data..."):
            # Fetch latest signal
            signal = fetch_trading_signal()
            if signal:
                with signal_container:
                    st.subheader("Latest Trading Signal")
                    display_signal_card(signal)
                    st.caption(f"Last updated: {signal.get('timestamp', 'N/A')}")
                    st.markdown(f"**Reason:** {signal.get('reason', 'N/A')}")
            
            # Fetch and plot historical data
            data = fetch_historical_data(days)
            if data is not None:
                with chart_container:
                    st.plotly_chart(plot_forex_chart(data), use_container_width=True)
            
            st.session_state.last_refresh = datetime.now()
    
    # Display last refresh time
    if 'last_refresh' in st.session_state:
        st.sidebar.caption(
            f"Last refresh: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
        )

if __name__ == "__main__":
    main() 