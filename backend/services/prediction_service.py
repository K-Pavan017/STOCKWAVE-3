import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from models.stock_data import StockData
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_data_from_db(symbol, lookback_days):
    records = (
        StockData.query
        .filter(StockData.company_symbol == symbol)
        .order_by(StockData.date.asc())
        .limit(lookback_days)
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

def prepare_data_multi_feature(df_full, features_to_scale, window_size=60, train_test_split_ratio=0.8):
    # Make a copy to avoid SettingWithCopyWarning
    df_processed = df_full.copy()

    df_processed.dropna(inplace=True)

    data_to_scale = df_processed[features_to_scale].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)

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
    predictions = []
    current_input = input_seq.copy() 

    for _ in range(steps):
        next_pred_scaled = model.predict(current_input, verbose=0)[0][0]
        predictions.append(next_pred_scaled)

        dummy_next_features = current_input[0, -1, :].copy().reshape(1, 1, num_features)
        dummy_next_features[0, 0, close_feature_idx] = next_pred_scaled

        current_input = np.append(current_input[:, 1:, :], dummy_next_features, axis=1)

    dummy_full_predictions_scaled = np.zeros((len(predictions), num_features))
    dummy_full_predictions_scaled[:, close_feature_idx] = np.array(predictions)

    predicted_prices_full_features = scaler.inverse_transform(dummy_full_predictions_scaled)
    predicted_close_prices = predicted_prices_full_features[:, close_feature_idx]

    return predicted_close_prices

def lstm_predict_multiple(symbol, horizon='day', lookback_days=240):
    features_to_scale = ['open', 'high', 'low', 'close', 'volume']
    df = get_data_from_db(symbol, lookback_days + 120)

    if df is not None:
        print(f"--- DB records fetched for prediction for {symbol}: {len(df)} days ---")
    else:
        print(f"--- No records fetched from DB for prediction for {symbol} ---")

    if df is None or df.empty or len(df) < 200:
        return None, "Insufficient data to train model or generate features."
    
    df.loc[:, 'SMA_10'] = df['close'].rolling(window=10).mean()
    df.loc[:, 'EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df.loc[:, 'Daily_Return'] = df['close'].pct_change()

    features_to_scale.extend(['SMA_10', 'EMA_10', 'Daily_Return'])
    
    steps_map = {'day': 1, 'week': 7, 'month': 30, '3month': 90}
    steps = steps_map.get(horizon.lower(), 1)

    window_size = 60
    close_feature_idx = features_to_scale.index('close')

    # Pass the full df to prepare_data_multi_feature, which will handle dropping NaNs
    X_train, y_train, X_test, y_test, scaler, scaled_data_full, df_processed = \
        prepare_data_multi_feature(df, features_to_scale=features_to_scale, window_size=window_size)

    if len(X_test) == 0:
        return None, "Not enough data to create a test set for evaluation. Consider increasing lookback_days."

    model = build_model_improved((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1, validation_split=0.1)

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss}")

    test_predictions_scaled = model.predict(X_test, verbose=0)
    dummy_test_predictions_scaled = np.zeros((len(test_predictions_scaled), len(features_to_scale)))
    dummy_test_predictions_scaled[:, close_feature_idx] = test_predictions_scaled.flatten()
    test_predictions = scaler.inverse_transform(dummy_test_predictions_scaled)[:, close_feature_idx]

    dummy_actual_test_prices_scaled = np.zeros((len(y_test), len(features_to_scale)))
    dummy_actual_test_prices_scaled[:, close_feature_idx] = y_test.flatten()
    actual_test_prices = scaler.inverse_transform(dummy_actual_test_prices_scaled)[:, close_feature_idx]

    rmse = np.sqrt(mean_squared_error(actual_test_prices, test_predictions))
    mae = mean_absolute_error(actual_test_prices, test_predictions)
    r2 = r2_score(actual_test_prices, test_predictions)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")

    epsilon = 1e-10
    mape = np.mean(np.abs((actual_test_prices - test_predictions) / (actual_test_prices + epsilon))) * 100
    print(f"MAPE: {mape:.2f}%")

    if len(scaled_data_full) < window_size:
        return None, "Not enough data for generating future predictions (window_size too large for available data)."

    last_input_sequence = scaled_data_full[-window_size:].reshape(1, window_size, len(features_to_scale))

    predicted_close_prices = predict_multiple_steps_multi_feature(
        model, last_input_sequence, scaler, steps, len(features_to_scale), close_feature_idx
    )

    # Use df_processed (which has 'date' column and is cleaned) for last_date
    last_date = df_processed['date'].iloc[-1]
    future_dates = []
    current = last_date

    while len(future_dates) < steps:
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current)

    if len(predicted_close_prices) > len(future_dates):
        predicted_close_prices = predicted_close_prices[:len(future_dates)]
    elif len(predicted_close_prices) < len(future_dates):
          future_dates = future_dates[:len(predicted_close_prices)]

    first_predicted_close = round(float(predicted_close_prices[0]), 2) if len(predicted_close_prices) > 0 else None

    result = {
        'predicted_close': first_predicted_close,
        'predicted_open': first_predicted_close,
        'predicted_high': round(float(first_predicted_close * 1.01), 2) if first_predicted_close is not None else None,
        'predicted_low': round(float(first_predicted_close * 0.99), 2) if first_predicted_close is not None else None,
        'close_series': [
            {'date': d.strftime('%Y-%m-%d'), 'close': round(float(p), 2), 'predicted': True}
            for d, p in zip(future_dates, predicted_close_prices)
        ],
        'open_series': [
            {'date': d.strftime('%Y-%m-%d'), 'open': round(float(p), 2), 'predicted': True}
            for d, p in zip(future_dates, predicted_close_prices)
        ],
        'high_series': [
            {'date': d.strftime('%Y-%m-%d'), 'high': round(float(p), 2), 'predicted': True}
            for d, p in zip(future_dates, predicted_close_prices)
        ],
        'low_series': [
            {'date': d.strftime('%Y-%m-%d'), 'low': round(float(p), 2), 'predicted': True}
            for d, p in zip(future_dates, predicted_close_prices)
        ],
        'confidence': 0.85
    }

    return result, None
