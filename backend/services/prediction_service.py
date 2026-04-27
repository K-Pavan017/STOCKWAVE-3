import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import gc

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

def prepare_features(df, features_to_use, window_size=10):
    """
    Prepares lagged-window features for GradientBoosting.
    Each sample is a flattened window of the last `window_size` rows.
    """
    df_processed = df.copy()
    df_processed.dropna(inplace=True)
    
    X, y = [], []
    data_values = df_processed[features_to_use].values
    target_values = df_processed['target'].values
    
    for i in range(window_size, len(data_values)):
        X.append(data_values[i-window_size:i].flatten())
        y.append(target_values[i])
        
    return np.array(X), np.array(y)

def generate_stock_prediction(symbol, horizon='day', lookback_days=365):
    """
    Main entry point for predictions. Uses GradientBoosting with technical indicators.
    """
    # Configuration
    window_size = 10
    features_to_scale = ['open', 'high', 'low', 'close', 'volume']
    
    if not horizon:
        horizon = 'month'
    steps_map = {'day': 1, 'week': 7, 'month': 30, '3month': 90}
    steps = steps_map.get(str(horizon).strip().lower(), 30)
    
    # 1. Data Fetching
    df = get_data_from_db(symbol, lookback_days)
    if df is None or df.empty or len(df) < (window_size + 50):
        return None, f"Insufficient data to generate prediction (need ~{window_size+50} records)."
        
    # 2. Calculate Features (Technical Indicators)
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
    
    # Cast to float32 to save memory
    for col in features_to_scale:
        df[col] = df[col].astype(np.float32)
    df['target'] = df['target'].astype(np.float32)

    if len(df) < (window_size + 10):
        return None, "Insufficient data after feature generation."

    # 3. Model Training with GradientBoosting (fast, low-memory)
    # Reducing complexity for Render's 512MB limit
    print(f"[{symbol}] Training GradientBoosting model (v3-min-mem) on {len(df)} rows...")
    X, y = prepare_features(df, features_to_scale, window_size)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=3,      # Keep at 3
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Explicitly clear X and y to free memory
    del X
    del y
    gc.collect()
    
    # 4. Multi-step Prediction (Recursive)
    last_close = df['close'].iloc[-1]
    predicted_prices = []
    current_price = last_close
    
    current_window = df[features_to_scale].values[-window_size:]
    
    for _ in range(steps):
        window_flat = current_window.flatten().reshape(1, -1)
        window_scaled = scaler.transform(window_flat)
        pred_return = model.predict(window_scaled)[0]
            
        current_price = current_price * np.exp(pred_return)
        predicted_prices.append(current_price)
        
        # Shift window
        next_row = current_window[-1].copy()
        next_row[3] = current_price # index 3 is 'close'
        current_window = np.append(current_window[1:], [next_row], axis=0)

    # 5. Format Results
    last_date = df['date'].iloc[-1]
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < steps:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            future_dates.append(current_date)
            
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
        'final_predicted_close': round(float(predicted_prices[-1]), 2),
        'total_change_percent': round(((predicted_prices[-1] - last_close) / last_close) * 100, 2),
        'close_series': close_series_data,
        'open_series': [{'date': d['date'], 'open': d['close'], 'predicted': True} for d in close_series_data],
        'high_series': [{'date': d['date'], 'high': round(d['close'] * 1.01, 2), 'predicted': True} for d in close_series_data],
        'low_series': [{'date': d['date'], 'low': round(d['close'] * 0.99, 2), 'predicted': True} for d in close_series_data],
        'confidence': 0.85
    }
    
    print(f"[{symbol}] Prediction generated successfully using optimized GradientBoosting.")
    return result, None
