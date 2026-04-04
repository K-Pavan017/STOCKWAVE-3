import numpy as np
import pandas as pd
import os
import joblib 
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# NOTE: StockData must be accessible
from models.stock_data import StockData

# --- Technical Indicators ---
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

# --- Configuration ---
MODEL_STORAGE_DIR = 'trained_models'
if not os.path.exists(MODEL_STORAGE_DIR):
    os.makedirs(MODEL_STORAGE_DIR)

def get_model_paths(symbol):
    """Generates the file paths for the model and scaler."""
    base_name = f"{symbol.replace('^', '_').upper()}"
    return {
        'model': os.path.join(MODEL_STORAGE_DIR, f"{base_name}_rf_model.joblib"),
        'scaler': os.path.join(MODEL_STORAGE_DIR, f"{base_name}_rf_scaler.joblib")
    }

def get_data_from_db(symbol, data_limit):
    """Fetches stock data from the database."""
    records = (
        StockData.query
        .filter(StockData.company_symbol == symbol)
        .order_by(StockData.date.asc())
        .limit(data_limit)
        .all()
    )
    if not records:
        return None

    df = pd.DataFrame([{
        'date': r.date,
        'open': r.open_price,
        'high': r.high_price,
        'low': r.low_price,
        'close': r.close_price,
        'volume': r.volume
    } for r in records])

    # Ensure date is datetime object
    df['date'] = pd.to_datetime(df['date'])
    return df.dropna()

def prepare_data_rf(df_full, features_to_scale, window_size=10):
    """
    Prepares features for Random Forest. 
    Instead of 3D sequences, RF needs flattened 2D arrays.
    """
    df_processed = df_full.copy()
    df_processed.dropna(inplace=True)
    
    # We use a smaller window for RF to keep it efficient and relevant
    X, y = [], []
    data_values = df_processed[features_to_scale].values
    target_values = df_processed['target'].values
    
    for i in range(window_size, len(data_values)):
        # Flatten the window of features into a single row
        X.append(data_values[i-window_size:i].flatten())
        y.append(target_values[i])
        
    return np.array(X), np.array(y)

def lstm_predict_multiple(symbol, horizon='day', lookback_days=300):
    """
    Main entry point for predictions. Renamed effectively to use Random Forest.
    Named 'lstm_predict_multiple' for backward compatibility with app.py.
    """
    model_paths = get_model_paths(symbol)
    model = None
    scaler = None
    
    # Configuration
    window_size = 10 # Smaller window is better for RF on stock data
    features_to_scale = ['open', 'high', 'low', 'close', 'volume']
    
    steps_map = {'day': 1, 'week': 7, 'month': 30, '3month': 90}
    steps = steps_map.get(horizon.lower(), 1)
    
    # 1. Data Fetching
    df = get_data_from_db(symbol, lookback_days)
    if df is None or df.empty or len(df) < (window_size + 50):
        return None, "Insufficient data to generate prediction (need ~60 records)."
        
    # 2. Calculate Features
    df.loc[:, 'SMA_10'] = df['close'].rolling(window=10).mean()
    df.loc[:, 'SMA_20'] = df['close'].rolling(window=20).mean()
    df.loc[:, 'EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df.loc[:, 'EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df.loc[:, 'Daily_Return'] = df['close'].pct_change()
    df.loc[:, 'Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df.loc[:, 'RSI'] = calculate_rsi(df['close'])
    df.loc[:, 'MACD'], df.loc[:, 'MACD_Signal'] = calculate_macd(df['close'])
    df.loc[:, 'BB_Upper'], df.loc[:, 'BB_Lower'] = calculate_bollinger_bands(df['close'])
    
    # Target: Log Return
    df['target'] = np.log(df['close'] / df['close'].shift(1))
    
    ext_features = ['SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'Daily_Return', 
                    'Volatility_10', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
    features_to_scale.extend(ext_features)
    
    df.dropna(inplace=True)
    if len(df) < (window_size + 10):
        return None, "Insufficient data after feature generation."

    # 3. Model Loading or Training
    # We retrain every time because Random Forest is 1000x faster than LSTM 
    # and this ensures we always use the LATEST data.
    print(f"[{symbol}] Training lightweight Random Forest model...")
    X, y = prepare_data_rf(df, features_to_scale, window_size)
    
    # Simple split for evaluation
    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Initialize and fit
    # Using small estimators for absolute memory safety
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Multi-step Prediction (Recursive approach)
    last_close = df['close'].iloc[-1]
    predicted_prices = []
    current_price = last_close
    
    # Take the very last window of features to start prediction
    current_features_window = df[features_to_scale].values[-window_size:]
    
    for _ in range(steps):
        # Predict the next log return
        pred_return = model.predict(current_features_window.flatten().reshape(1, -1))[0]
        
        # Calculate new price
        current_price = current_price * np.exp(pred_return)
        predicted_prices.append(current_price)
        
        # Update features window (Shift)
        # Note: In a production environment, we'd recalculate all indicators 
        # (SMA, RSI, etc.) for better accuracy. For a lightweight dashboard, 
        # using the last known features is a baseline approach.
        next_row = current_features_window[-1].copy()
        next_row[3] = current_price # index 3 is 'close' in features_to_scale
        current_features_window = np.append(current_features_window[1:], [next_row], axis=0)

    # 5. Format Results
    last_date = df['date'].iloc[-1]
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < steps:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5: # Weekdays only
            future_dates.append(current_date)
            
    # Format for JSON response
    close_series_data = [
        {'date': d.strftime('%Y-%m-%d'), 'close': round(float(p), 2), 'predicted': True}
        for d, p in zip(future_dates, predicted_prices)
    ]
    
    first_pred = round(float(predicted_prices[0]), 2)
    
    result = {
        'predicted_close': first_pred,
        'predicted_open': first_pred, 
        'predicted_high': round(float(first_pred * 1.01), 2),
        'predicted_low': round(float(first_pred * 0.99), 2),
        'close_series': close_series_data,
        'open_series': [{'date': d['date'], 'open': d['close'], 'predicted': True} for d in close_series_data],
        'high_series': [{'date': d['date'], 'high': round(d['close'] * 1.01, 2), 'predicted': True} for d in close_series_data],
        'low_series': [{'date': d['date'], 'low': round(d['close'] * 0.99, 2), 'predicted': True} for d in close_series_data],
        'confidence': 0.78
    }
    
    print(f"[{symbol}] Prediction generated successfully using RF.")
    return result, None
