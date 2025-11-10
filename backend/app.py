from flask import Flask, request, jsonify
from flask_cors import CORS
# Assuming config.py exists and defines a Config class
from .config import Config
from waitress import serve
import jwt 
import os
from .database import db # Assuming database.py only exports 'db' (SQLAlchemy instance)
from .models.stock_data import StockData # Ensure this is imported for db.create_all
from .services import auth_service, data_services, prediction_service # Assuming these service modules exist

from datetime import datetime, timedelta
import pandas as pd
from math import ceil # Import ceil for calculating months
# In app.py or equivalent config file
from flask_migrate import Migrate
# ...
# Initialize SQLAlchemy (db) and Flask-Migrate (migrate)

app = Flask(__name__)
app.config.from_object(Config) # Load configuration from Config object
CORS(app, 
     origins=["https://stockwave-3.vercel.app"], # Replace with your actual URL
     supports_credentials=True) # Enable CORS for all routes

# Initialize SQLAlchemy with the Flask app
db.init_app(app)

migrate = Migrate(app, db)


# Create database tables if they don't exist
# This should be done within the application context, preferably once on startup.
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return "StockWave Backend is running!"

@app.route('/signup', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400

    # Assuming auth_service.register_user handles user creation and returns appropriate JSON
    return auth_service.register_user(username, email, password)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'success': False, 'message': 'Missing email or password'}), 400

    # Assuming auth_service.login_user handles login and returns appropriate JSON
    return auth_service.login_user(email, password)

@app.route('/stock/fetch', methods=['POST'])
def fetch_and_store_stock_route():
    data = request.get_json()
    print(f"--- app.py: Received fetch request with data: {data} ---")
    symbol = data.get('symbol')
    months = data.get('months')
    market = data.get('market')

    if not symbol or not months:
        return jsonify({'success': False, 'message': 'Missing symbol or months'}), 400

    try:
        # Call the orchestrator function from data_services
        success, message = data_services.fetch_and_store_stock(symbol, months, market)

        # After fetching and storing, get the latest statistics to return to the frontend
        # This provides immediate feedback including current price and day's change.
        statistics = None
        if success:
            statistics = data_services.get_stock_statistics(symbol)

        if success:
            return jsonify({'success': True, 'message': message, 'statistics': statistics}), 200
        else:
            return jsonify({'success': False, 'message': message}), 500
    except Exception as e:
        print(f"--- app.py: Unhandled Error fetching/storing stock data for {symbol}: {e} ---")
        return jsonify({'success': False, 'message': f'Server error during data fetch: {str(e)}'}), 500

@app.route('/stock/data/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    market = request.args.get('market', 'US')
    formatted_symbol = data_services.format_symbol(symbol, market)
    limit = int(request.args.get('limit', 365)) # Max records to return
    days = int(request.args.get('days', limit)) # Duration for historical fetch if needed, and for stats

    # 1. Try to get data from DB first
    # get_stored_stock_data typically orders by date descending.
    # We want enough data for charting for 'limit' days.
    records = data_services.get_stored_stock_data(company_symbol=formatted_symbol, limit=limit)
    # Ensure chronological order for charts (oldest to newest)
    records = sorted(records, key=lambda r: r.date)

    # 2. If DB data is insufficient, fetch from YFinance and store
    # Consider "insufficient" if we have significantly less data than requested by 'limit'
    if not records or len(records) < limit * 0.9:
        print(f"[GET_STOCK_DATA] Insufficient DB records for {formatted_symbol} ({len(records)}/{limit}). Attempting to fetch and store.")
        # Fetch enough data to cover the requested 'days' (duration) for the chart, plus some buffer.
        # Convert days to months for fetch_and_store_stock if 'days' is substantial.
        months_to_fetch = ceil(days / 30) + 1 if days > 0 else 1 # Fetch at least 1 month, or enough for 'days' + buffer
        success, message = data_services.fetch_and_store_stock(formatted_symbol, months=months_to_fetch, market=market)
        if success:
            print(f"[GET_STOCK_DATA] Successfully fetched and stored new data for {formatted_symbol}.")
            # Re-fetch from DB after storing
            records = data_services.get_stored_stock_data(company_symbol=formatted_symbol, limit=limit)
            records = sorted(records, key=lambda r: r.date)
        else:
            print(f"[GET_STOCK_DATA_ERROR] Failed to fetch and store data: {message}")
            return jsonify({'success': False, 'message': f"Failed to get historical data for {symbol}: {message}"}), 500 # Return 500 if fetch failed

    if not records: # After attempting to fetch and store, if still no records
        return jsonify({'success': False, 'message': f"No historical data available for {symbol}."}), 404

    # Prepare data for frontend
    processed_records = [{
        'date': r.date.strftime('%Y-%m-%d'),
        'open': float(r.open_price),
        'high': float(r.high_price),
        'low': float(r.low_price),
        'close': float(r.close_price),
        'volume': int(r.volume),
    } for r in records]

    # Get statistics using the function that handles DB/YFinance fallback
    # Use the 'days' parameter from the frontend for statistics
    stats = data_services.get_stock_statistics(formatted_symbol, days=days)

    return jsonify({
        'success': True,
        'data': {
            'records': processed_records,
            'statistics': stats
        }
    }), 200

# NEW: API route for fetching real-time stock info (for ticker and dashboard preview)
@app.route('/api/stock_info/<symbol>', methods=['GET'])
def api_stock_info(symbol):
    info = data_services.get_company_info(symbol)
    if info:
        return jsonify({"success": True, "data": info})
    return jsonify({"success": False, "message": "Could not retrieve company info."}), 404

# NEW: API route for fetching stock statistics (for dashboard preview details)
@app.route('/api/stock_statistics/<symbol>', methods=['GET'])
def api_stock_statistics(symbol):
    stats = data_services.get_stock_statistics(symbol, days=1) # Get today's stats for current, open, high, volume
    if stats:
        return jsonify({"success": True, "data": stats})
    return jsonify({"success": False, "message": "Could not retrieve stock statistics."}), 404

@app.route('/stock/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    horizon = request.args.get('horizon', 'month') # Default to 'month' for 30-day prediction
    market = request.args.get('market', 'US')
    formatted_symbol = data_services.format_symbol(symbol, market)

    # Ensure sufficient data is in DB for prediction.
    # The LSTM model typically needs a good amount of historical data (e.g., 240-360 days)
    # Check if we have at least 400 days (approx. 13-14 months) for robust prediction.
    min_prediction_data_days = 400
    db_records_for_pred = data_services.get_stored_stock_data(company_symbol=formatted_symbol, limit=min_prediction_data_days)

    if not db_records_for_pred or len(db_records_for_pred) < min_prediction_data_days * 0.9: # If significantly less data
        print(f"[PREDICT_PREP] Insufficient DB data ({len(db_records_for_pred)} records) for prediction for {formatted_symbol}. Attempting to fetch and store ~18 months.")
        # Fetching ~18 months should generally provide enough data for prediction's lookback_days (e.g., 240+120=360 days)
        success, fetch_message = data_services.fetch_and_store_stock(formatted_symbol, months=18, market=market)
        if not success:
            return jsonify({'success': False, 'message': f"Prediction failed due to insufficient historical data: {fetch_message}"}), 500
        print(f"[PREDICT_PREP] Successfully fetched and stored additional data for {formatted_symbol}.")
        # No need to re-fetch db_records_for_pred here, as prediction_service.lstm_predict_multiple
        # will internally call get_data_from_db which will get the newly stored data.

    predictions_data, error_message = prediction_service.lstm_predict_multiple(formatted_symbol, horizon=horizon)

    if error_message:
        print(f"Prediction Error for {formatted_symbol}: {error_message}")
        return jsonify({"success": False, "message": error_message}), 400

    if predictions_data:
        return jsonify({"success": True, "prediction": predictions_data})

    return jsonify({"success": False, "message": "Prediction could not be generated."}), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)
