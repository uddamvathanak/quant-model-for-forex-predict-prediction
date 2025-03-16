# Forex Price Movement Prediction Model

This project implements a machine learning model for predicting forex price movements using technical indicators and the LightGBM algorithm.

## Features

- Historical data fetching using Yahoo Finance
- Comprehensive technical indicator generation
- Advanced feature engineering
- Hyperparameter optimization using Optuna
- Model evaluation and visualization
- Easy-to-use API for predictions

## Prerequisites

### 1. Install Miniconda

First, you need to install Miniconda, which is a minimal installer for Conda.

#### Windows:
1. Download the Miniconda installer for Windows from [here](https://docs.conda.io/en/latest/miniconda.html)
2. Run the installer (`.exe` file)
3. Follow the installation prompts
4. Open "Anaconda Prompt (Miniconda3)" from the Start Menu to verify installation

#### macOS/Linux:
1. Download the Miniconda installer:
```bash
# macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Linux
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Run the installer:
```bash
# macOS
bash Miniconda3-latest-MacOSX-x86_64.sh

# Linux
bash Miniconda3-latest-Linux-x86_64.sh
```

3. Follow the installation prompts
4. Restart your terminal or run:
```bash
source ~/.bashrc  # Linux
source ~/.zshrc   # macOS
```

### 2. Create and Activate Virtual Environment

After installing Miniconda, create and activate a new virtual environment:

```bash
# Create new environment named 'forex_env' with Python 3.9
conda create -n forex_env python=3.9

# Activate the environment
conda activate forex_env

# Verify Python installation
python --version
```

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

3. Install TA-Lib (Technical Analysis Library):

**Recommended Method (All Platforms)**
```bash
conda install -c conda-forge ta-lib
```
This is the easiest and most reliable method for all operating systems (Windows, macOS, and Linux).

If the above method doesn't work, you can try the platform-specific methods below:

#### Windows Alternative Methods:
**Method 1: Using Pre-compiled Wheels**
1. Download the appropriate wheel file for your Python version:
   - For Python 3.9 64-bit: [TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl](https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl)
   - For other versions, visit [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

2. Install the downloaded wheel file:
```bash
# Navigate to the download directory
cd path/to/download/directory

# Install the wheel file
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```

**Method 2: Building from Source**
If Method 1 doesn't work, you can build from source:

1. Install Visual Studio Build Tools:
   - Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Run the installer
   - Select "Desktop development with C++"
   - Make sure to include "Windows 10 SDK" and "MSVC v140"

2. Download and Install TA-Lib:
   - Download [ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
   - Unzip to `C:\ta-lib`
   - Open "x64 Native Tools Command Prompt for VS 2022" (or your VS version)
   - Run:
   ```bash
   cd C:\ta-lib
   lib /machine:x64 /def:c:\ta-lib\ta-lib.def
   ```

3. Set up environment variables:
   - Open System Properties → Advanced → Environment Variables
   - Add to System Variables:
     - Variable: `LIB`
     - Value: `C:\ta-lib\lib`
   - Add to Path:
     - `C:\ta-lib\bin`

4. Finally, install TA-Lib for Python:
```bash
pip install ta-lib
```

If you encounter the error "Cannot find ta-lib library", try these steps:
1. Make sure you've completed all steps above
2. Try installing an older version:
```bash
pip install ta-lib==0.4.21
```
3. If still having issues, try:
```bash
conda install -c conda-forge ta-lib
```

#### macOS Alternative Method:
```bash
# Using Homebrew
brew install ta-lib
pip install ta-lib
```

#### Linux Alternative Method:
```bash
# Install dependencies
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

## Environment Setup

1. Copy the example environment file to create your own `.env` file:
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

2. Open the `.env` file in your text editor and update the values:
```ini
# Required: Get your API key from https://newsapi.org
NEWS_API_KEY=your_news_api_key_here

# Optional: Modify these parameters as needed
DATABASE_URL=sqlite:///forex_data.db
PREDICTION_HORIZON=1
CONFIDENCE_THRESHOLD=0.7
HISTORICAL_DATA_DAYS=30

# Technical Analysis Parameters
ATR_PERIOD=14
VWAP_PERIOD=5
BREAKOUT_WINDOW=20

# Trading Parameters
TP_MULTIPLIER=1.5
SL_MULTIPLIER=1.0
```

The most important setting is the `NEWS_API_KEY`. You can get one by:
1. Go to [https://newsapi.org](https://newsapi.org)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Paste it in the `.env` file

Other parameters can be left at their default values or adjusted based on your trading preferences:
- `PREDICTION_HORIZON`: Number of periods to predict ahead
- `CONFIDENCE_THRESHOLD`: Minimum confidence required for trading signals (0.0 to 1.0)
- `HISTORICAL_DATA_DAYS`: Days of historical data to use for training
- `ATR_PERIOD`: Period for Average True Range calculation
- `VWAP_PERIOD`: Period for Volume Weighted Average Price calculation
- `TP_MULTIPLIER`: Take Profit multiplier relative to ATR
- `SL_MULTIPLIER`: Stop Loss multiplier relative to ATR

## Usage

### Quick Start

1. Activate the virtual environment:
```bash
conda activate forex_env
```

2. Start the FastAPI backend:
```bash
python api.py
```

3. In a new terminal (with forex_env activated), start the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

This will:
1. Download EUR/USD historical data
2. Create technical indicators
3. Train a model with optimized hyperparameters
4. Evaluate the model performance
5. Generate a visualization of predictions
6. Save the trained model

### Using the ForexPredictor Class

```python
from forex_predictor import ForexPredictor

# Initialize predictor
predictor = ForexPredictor(currency_pair="EURUSD=X", prediction_horizon=1)

# Fetch and prepare data
data = predictor.fetch_data(start_date="2020-01-01")
data_with_features = predictor.create_features(data)
X_train, X_test, y_train, y_test = predictor.prepare_data(data_with_features)

# Train model
predictor.train(X_train, y_train, optimize=True)

# Make predictions
predictions = predictor.predict(X_test)

# Evaluate performance
metrics = predictor.evaluate(X_test, y_test)
print(metrics)

# Save model
predictor.save_model('my_model.joblib')
```

## Model Details

The model uses the following components:

1. **Data Collection**: Historical forex data from Yahoo Finance
2. **Feature Engineering**:
   - Technical indicators (momentum, trend, volatility, volume)
   - Price action patterns
   - Custom features based on market behavior

3. **Model Architecture**:
   - LightGBM Regressor
   - Hyperparameter optimization using Optuna
   - Standardized features

4. **Evaluation Metrics**:
   - RMSE (Root Mean Square Error)
   - R² Score

## Customization

You can customize the model by:

1. Modifying the currency pair:
```python
predictor = ForexPredictor(currency_pair="GBPUSD=X")
```

2. Changing the prediction horizon:
```python
predictor = ForexPredictor(prediction_horizon=5)  # 5-day ahead prediction
```

3. Adjusting hyperparameter optimization:
```python
predictor.train(X_train, y_train, optimize=True, n_trials=100)
```

## Troubleshooting

1. If you encounter issues with TA-Lib installation:
   - Windows: Download the appropriate wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Linux: Make sure you have build essentials installed: `sudo apt-get install build-essential`

2. If you get a "ModuleNotFoundError":
   - Make sure you're in the correct virtual environment: `conda activate forex_env`
   - Verify all packages are installed: `pip list`

3. If the API fails to start:
   - Check if the port 8000 is available
   - Verify your NEWS_API_KEY in the .env file

## Disclaimer

This model is for educational purposes only. Trading forex carries significant risks, and past performance does not guarantee future results. Always conduct thorough research and consider consulting with financial advisors before making trading decisions.
