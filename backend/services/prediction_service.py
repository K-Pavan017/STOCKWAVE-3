import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from models.stock_data import StockData

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
        'close': r.close_price
    } for r in records])

    return df.dropna()

def prepare_data(df, feature='close', window_size=60):
    data = df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_multiple_steps(model, input_seq, scaler, steps):
    predictions = []
    current_input = input_seq.copy()

    for _ in range(steps):
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0][0])
        # FIX: Reshape next_pred to (1, 1, 1) to match dimensions for appending
        current_input = np.append(current_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_prices

def lstm_predict_multiple(symbol, horizon='day', lookback_days=365):
    df = get_data_from_db(symbol, lookback_days + 60)
    if df is None or df.empty or len(df) < 100:
        return None, "Insufficient data to train model."

    steps_map = {'day': 1, 'week': 7, 'month': 30, '3month': 90}
    steps = steps_map.get(horizon.lower(), 1)

    window_size = 60
    feature = 'close'

    X, y, scaler = prepare_data(df, feature=feature, window_size=window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    predicted_close_prices = predict_multiple_steps(model, X[-1:].copy(), scaler, steps)

    last_date = df['date'].iloc[-1]
    future_dates = []
    current = last_date

    while len(future_dates) < steps:
        current += timedelta(days=1)
        # Skip weekends for future dates
        if current.weekday() < 5:
            future_dates.append(current)
            
    # Ensure future_dates matches the number of predictions, especially if skipping weekends
    # Trim predictions if more than future_dates (e.g., if steps was large and hit many weekends)
    if len(predicted_close_prices) > len(future_dates):
        predicted_close_prices = predicted_close_prices[:len(future_dates)]
    # Or, if future_dates got truncated by prediction length, vice-versa
    elif len(predicted_close_prices) < len(future_dates):
         future_dates = future_dates[:len(predicted_close_prices)]


    # Construct the result to match what frontend expects
    # For now, open/high/low predictions are just the close prediction for day 1
    # as the model is only trained on 'close'
    first_predicted_close = round(float(predicted_close_prices[0]), 2) if len(predicted_close_prices) > 0 else None

    result = {
        'predicted_close': first_predicted_close,
        'predicted_open': first_predicted_close, # Using close as placeholder
        'predicted_high': round(float(first_predicted_close * 1.01), 2) if first_predicted_close is not None else None, # A slight estimation
        'predicted_low': round(float(first_predicted_close * 0.99), 2) if first_predicted_close is not None else None, # A slight estimation
        'close_series': [ # This is the actual series for charting
            {'date': d.strftime('%Y-%m-%d'), 'close': round(float(p), 2), 'predicted': True}
            for d, p in zip(future_dates, predicted_close_prices)
        ],
        # For open/high/low series for charting, use the close predictions as placeholders.
        # If true predictions for these were available, they would replace these.
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
        'confidence': 0.85 # Placeholder confidence, can be derived from model loss or other metrics
    }

    return result, None