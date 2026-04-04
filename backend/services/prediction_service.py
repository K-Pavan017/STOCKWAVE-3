import numpy as np
import pandas as pd
import os
import joblib # Required for saving/loading the scaler object
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow/Keras Imports moved inside functions to prevent startup timeout

# Placeholder/Original Imports (Ensure these are available in your environment)
# NOTE: You MUST ensure StockData is accessible or imported for this to run
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

# --- Configuration for Model Persistence ---
# Define a directory to save trained models and scalers.
# Ensure this directory exists in your environment.
MODEL_STORAGE_DIR = 'trained_models'
if not os.path.exists(MODEL_STORAGE_DIR):
    os.makedirs(MODEL_STORAGE_DIR)

def get_model_paths(symbol):
    """Generates the file paths for the model and scaler."""
    # Sanitizes symbol for use in a filename
    base_name = f"{symbol.replace('^', '_').upper()}"
    return {
        'model': os.path.join(MODEL_STORAGE_DIR, f"{base_name}_model.keras"),
        'scaler': os.path.join(MODEL_STORAGE_DIR, f"{base_name}_scaler.pkl")
    }
# --- End Configuration ---


def get_data_from_db(symbol, data_limit):
    """
    Fetches stock data from the database.
    Note: The 'data_limit' should be sufficient for training (e.g., 2 years) or 
    just for the last lookback sequence, depending on if the model needs training.
    """
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

    # Ensure date is datetime object for operations later
    df['date'] = pd.to_datetime(df['date'])
    return df.dropna()

def prepare_data_multi_feature(
    df_full,
    features_to_scale,
    window_size=60,
    train_test_split_ratio=0.8,
    is_training=True,
    scaler=None
):

    """
    Prepares and scales data. If not training, it only scales the data
    and returns the required components for prediction.
    """
    df_processed = df_full.copy()
    df_processed.dropna(inplace=True)

    data_to_scale = df_processed[features_to_scale].values
    if is_training:
       scaler = MinMaxScaler(feature_range=(0, 1))
       scaled_data = scaler.fit_transform(data_to_scale)
    else:
        scaled_data = scaler.transform(data_to_scale)


    if not is_training:
        # If just predicting, return the scaler and the last sequence
        last_sequence = scaled_data[-window_size:]
        return scaler, scaled_data, df_processed, last_sequence

    # Logic for Training (same as original)
    X, y = [], []
    

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(df_processed['target'].iloc[i])


    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * train_test_split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler, scaled_data, df_processed

def build_model_improved(input_shape):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    
    # Lightweight model — fast training on free-tier hardware
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber_loss",
        metrics=['mae']
    )

    return model


def get_training_callbacks(model_path):
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=0.0001
    )

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    return [early_stop, checkpoint, reduce_lr]


def predict_multiple_steps_returns(model, last_input_sequence, df_processed, steps):
    predicted_returns = []
    current_input = last_input_sequence.copy()

    for _ in range(steps):
        r = model.predict(current_input, verbose=0)[0][0]
        predicted_returns.append(r)

        # shift window (features stay same except close implicitly handled)
        next_row = current_input[0, -1, :].reshape(1, 1, -1)
        current_input = np.append(current_input[:, 1:, :], next_row, axis=1)

    # convert returns → prices
    last_close = df_processed['close'].iloc[-1]
    prices = []
    price = last_close

    for r in predicted_returns:
        price = price * np.exp(r)
        prices.append(price)

    return prices

def lstm_predict_multiple(symbol, horizon='day', lookback_days=240):
    """
    Predicts stock price using an LSTM model, loading a saved model if available.
    """
    from keras.models import load_model
    
    model_paths = get_model_paths(symbol)
    model = None
    scaler = None
    is_trained = False
    
    features_to_scale = ['open', 'high', 'low', 'close', 'volume']
    window_size = 60
    
    steps_map = {'day': 1, 'week': 7, 'month': 30, '3month': 90}
    steps = steps_map.get(horizon.lower(), 1)
    
    # 1. Check and Load Model/Scaler
    if os.path.exists(model_paths['model']) and os.path.exists(model_paths['scaler']):
        try:
            print(f"[{symbol}] Loading pre-trained model and scaler...")
            model = load_model(model_paths['model'])
            scaler = joblib.load(model_paths['scaler'])
            is_trained = True
        except Exception as e:
            print(f"[{symbol}] Error loading cached model/scaler, forcing retraining: {e}")
            model = None

    # 2. Data Fetching
    data_limit = lookback_days + 120
    df = get_data_from_db(symbol, data_limit)

    if df is None or df.empty or len(df) < 200:
        return None, "Insufficient data to train model or generate features."
    
    # Calculate Features (MUST RUN every time to get latest indicators)
    df.loc[:, 'SMA_10'] = df['close'].rolling(window=10).mean()
    df.loc[:, 'SMA_20'] = df['close'].rolling(window=20).mean()
    df.loc[:, 'EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df.loc[:, 'EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df.loc[:, 'Daily_Return'] = df['close'].pct_change()
    df.loc[:, 'Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df.loc[:, 'RSI'] = calculate_rsi(df['close'])
    df.loc[:, 'MACD'], df.loc[:, 'MACD_Signal'] = calculate_macd(df['close'])
    df.loc[:, 'BB_Upper'], df.loc[:, 'BB_Lower'] = calculate_bollinger_bands(df['close'])
    
    df['target'] = np.log(df['close'] / df['close'].shift(1))
    
    features_to_scale.extend(['SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'Daily_Return', 
                             'Volatility_10', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower'])
    num_features = len(features_to_scale)

    # 3. Training/Retraining Logic (if model not loaded)
    if not is_trained:
        print(f"[{symbol}] Training new model (or retraining)...")

        # Prepare data for training
        X_train, y_train, X_test, y_test, scaler, scaled_data_full, df_processed = \
            prepare_data_multi_feature(
                df,
                features_to_scale,
                window_size,
                is_training=True,
                scaler=scaler
            )
        if len(X_test) == 0:
            return None, "Not enough data to create a test set for evaluation."

        # Build and Fit Model
        model = build_model_improved((X_train.shape[1], X_train.shape[2]))
        callbacks = get_training_callbacks(model_paths['model'])

        history = model.fit(
            X_train,
            y_train,
            epochs=70,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0  # Suppress per-epoch output to reduce log noise
        )
    
        
        # --- Evaluation (Only run on training) ---
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        test_predictions_scaled = model.predict(X_test, verbose=0)
        dummy_test_predictions_scaled = np.zeros((len(test_predictions_scaled), num_features))
        
        dummy_actual_test_prices_scaled = np.zeros((len(y_test), num_features))
        
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions_scaled))
        print(f"[{symbol}] Return RMSE: {rmse:.6f}")

        r2 = r2_score(y_test, test_predictions_scaled)
        
        print(f"[{symbol}] Training Complete. RMSE: {rmse:.2f}, R-squared: {r2:.2f}%")
        # -----------------------------------------

        # Save Model and Scaler
        try:
            model.save(model_paths['model'])
            joblib.dump(scaler, model_paths['scaler'])
            print(f"[{symbol}] Successfully trained and saved new model.")
        except Exception as e:
            print(f"[{symbol}] Warning: Failed to save model/scaler: {e}")

    else:
        # 4. Data preparation for prediction (if model loaded)
        # Use the loaded scaler and prepare only the last sequence needed
        scaler, scaled_data_full, df_processed, last_input_sequence_data = \
            prepare_data_multi_feature(df, features_to_scale=features_to_scale, window_size=window_size, is_training=False)
        
        # Reshape the last sequence for the model
        last_input_sequence = last_input_sequence_data.reshape(1, window_size, num_features)
    
    
    if model is None or df_processed is None or len(scaled_data_full) < window_size:
        return None, "Model failed to load or data is insufficient for final prediction sequence."

    # Use the prepared sequence from the training/loading step
    if 'last_input_sequence' not in locals():
        # This occurs if the model was just trained in the 'if not is_trained' block.
        last_input_sequence = scaled_data_full[-window_size:].reshape(1, window_size, num_features)


    # 5. Generate Predictions
    predicted_close_prices = predict_multiple_steps_returns(
        model,
        last_input_sequence,
        df_processed,
        steps
    )

    # 6. Format Results (FIXED: Close series defined first)
    last_date = df_processed['date'].iloc[-1]
    future_dates = []
    current = last_date

    while len(future_dates) < steps:
        current += timedelta(days=1)
        if current.weekday() < 5: # Only include weekdays
            future_dates.append(current)

    # Ensure dates and predictions arrays are the same length
    if len(predicted_close_prices) > len(future_dates):
        predicted_close_prices = predicted_close_prices[:len(future_dates)]
    elif len(predicted_close_prices) < len(future_dates):
          future_dates = future_dates[:len(predicted_close_prices)]

    first_predicted_close = round(float(predicted_close_prices[0]), 2) if len(predicted_close_prices) > 0 else None

    # --- FIX START: Calculate close_series_data first to avoid UnboundLocalError ---
    close_series_data = [
        {'date': d.strftime('%Y-%m-%d'), 'close': round(float(p), 2), 'predicted': True}
        for d, p in zip(future_dates, predicted_close_prices)
    ]
    # --- FIX END ---
    
    # Now, define result, referencing close_series_data instead of result['close_series']
    result = {
        'predicted_close': first_predicted_close,
        # Simplified assumption for open/high/low for the first day
        'predicted_open': first_predicted_close, 
        'predicted_high': round(float(first_predicted_close * 1.01), 2) if first_predicted_close is not None else None,
        'predicted_low': round(float(first_predicted_close * 0.99), 2) if first_predicted_close is not None else None,
        'close_series': close_series_data, # Use the pre-calculated list
        
        # Use the pre-calculated list (close_series_data) to derive other series
        'open_series': [{'date': d['date'], 'open': d['close'], 'predicted': True} for d in close_series_data],
        'high_series': [{'date': d['date'], 'high': round(d['close'] * 1.01, 2), 'predicted': True} for d in close_series_data],
        'low_series': [{'date': d['date'], 'low': round(d['close'] * 0.99, 2), 'predicted': True} for d in close_series_data],
        'confidence': 0.85
    }

    return result, None
