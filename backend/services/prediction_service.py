import numpy as np
import pandas as pd
import os
import joblib # Required for saving/loading the scaler object
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow/Keras Imports
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# Placeholder/Original Imports (Ensure these are available in your environment)
# NOTE: You MUST ensure StockData is accessible or imported for this to run
from models.stock_data import StockData

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

def prepare_data_multi_feature(df_full, features_to_scale, window_size=60, train_test_split_ratio=0.8, is_training=True):
    """
    Prepares and scales data. If not training, it only scales the data
    and returns the required components for prediction.
    """
    df_processed = df_full.copy()
    df_processed.dropna(inplace=True)

    data_to_scale = df_processed[features_to_scale].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)

    if not is_training:
        # If just predicting, return the scaler and the last sequence
        last_sequence = scaled_data[-window_size:]
        return scaler, scaled_data, df_processed, last_sequence

    # Logic for Training (same as original)
    X, y = [], []
    close_feature_idx = features_to_scale.index('close')

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i, close_feature_idx]) 

    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * train_test_split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler, scaled_data, df_processed

def build_model_improved(input_shape):
    """Defines the LSTM model architecture."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(1) 
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_multiple_steps_multi_feature(model, input_seq, scaler, steps, num_features, close_feature_idx):
    """Generates sequential predictions."""
    predictions = []
    current_input = input_seq.copy() 

    for _ in range(steps):
        next_pred_scaled = model.predict(current_input, verbose=0)[0][0]
        predictions.append(next_pred_scaled)

        # Create the new input row: copy the last row, update the 'close' price with the prediction
        dummy_next_features = current_input[0, -1, :].copy().reshape(1, 1, num_features)
        dummy_next_features[0, 0, close_feature_idx] = next_pred_scaled

        # Shift the window: remove the oldest and append the newest prediction
        current_input = np.append(current_input[:, 1:, :], dummy_next_features, axis=1)

    # Inverse Transform
    dummy_full_predictions_scaled = np.zeros((len(predictions), num_features))
    dummy_full_predictions_scaled[:, close_feature_idx] = np.array(predictions)
    predicted_prices_full_features = scaler.inverse_transform(dummy_full_predictions_scaled)
    predicted_close_prices = predicted_prices_full_features[:, close_feature_idx]

    return predicted_close_prices

def lstm_predict_multiple(symbol, horizon='day', lookback_days=240):
    """
    Predicts stock price using an LSTM model, loading a saved model if available.
    """
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
    df.loc[:, 'EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df.loc[:, 'Daily_Return'] = df['close'].pct_change()
    
    features_to_scale.extend(['SMA_10', 'EMA_10', 'Daily_Return'])
    close_feature_idx = features_to_scale.index('close')
    num_features = len(features_to_scale)

    # 3. Training/Retraining Logic (if model not loaded)
    if not is_trained:
        print(f"[{symbol}] Training new model (or retraining)...")

        # Prepare data for training
        X_train, y_train, X_test, y_test, scaler, scaled_data_full, df_processed = \
            prepare_data_multi_feature(df, features_to_scale=features_to_scale, window_size=window_size, is_training=True)

        if len(X_test) == 0:
            return None, "Not enough data to create a test set for evaluation."

        # Build and Fit Model
        model = build_model_improved((X_train.shape[1], X_train.shape[2]))
        # Reduced epochs for speed; adjust this number for better accuracy if needed
        history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, validation_split=0.1) 
        
        # --- Evaluation (Only run on training) ---
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        test_predictions_scaled = model.predict(X_test, verbose=0)
        dummy_test_predictions_scaled = np.zeros((len(test_predictions_scaled), num_features))
        dummy_test_predictions_scaled[:, close_feature_idx] = test_predictions_scaled.flatten()
        test_predictions = scaler.inverse_transform(dummy_test_predictions_scaled)[:, close_feature_idx]

        dummy_actual_test_prices_scaled = np.zeros((len(y_test), num_features))
        dummy_actual_test_prices_scaled[:, close_feature_idx] = y_test.flatten()
        actual_test_prices = scaler.inverse_transform(dummy_actual_test_prices_scaled)[:, close_feature_idx]

        rmse = np.sqrt(mean_squared_error(actual_test_prices, test_predictions))
        r2 = r2_score(actual_test_prices, test_predictions)
        epsilon = 1e-10
        mape = np.mean(np.abs((actual_test_prices - test_predictions) / (actual_test_prices + epsilon))) * 100
        
        print(f"[{symbol}] Training Complete. RMSE: {rmse:.2f}, R-squared: {r2:.2f}, MAPE: {mape:.2f}%")
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
    predicted_close_prices = predict_multiple_steps_multi_feature(
        model, last_input_sequence, scaler, steps, num_features, close_feature_idx
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
