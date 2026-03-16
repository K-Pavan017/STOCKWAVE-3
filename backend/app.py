from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import time
from functools import wraps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from waitress import serve
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_handler = RotatingFileHandler('logs/stockwave.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('StockWave backend startup')

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
stock_cache = TTLCache(maxsize=200, ttl=300)
prediction_cache = TTLCache(maxsize=100, ttl=3600)  # Cache predictions for 1 hour

# Simple rate limiting
request_counts = {}

def rate_limit(max_requests=10, window=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            
            # Clean old requests
            request_counts[client_ip] = [t for t in request_counts[client_ip] if current_time - t < window]
            
            if len(request_counts[client_ip]) >= max_requests:
                return jsonify({'success': False, 'message': 'Rate limit exceeded'}), 429
            
            request_counts[client_ip].append(current_time)
            return f(*args, **kwargs)
        return wrapped
    return decorator


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
@rate_limit(max_requests=10, window=60)  # 10 requests per minute for data fetching
def fetch_and_store_stock_route():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No JSON data provided'}), 400
        
    # Debug log removed for production
    symbol = data.get('symbol')
    months = data.get('months')
    market = data.get('market', 'US')

    # Input validation
    if not symbol or len(symbol) > 10 or not symbol.replace('.', '').replace('-', '').isalnum():
        return jsonify({'success': False, 'message': 'Invalid stock symbol'}), 400
    if not months or not isinstance(months, int) or months < 1 or months > 60:
        return jsonify({'success': False, 'message': 'Invalid months parameter (1-60)'}), 400
    if market not in ['US', 'IN']:
        return jsonify({'success': False, 'message': 'Invalid market. Must be US or IN'}), 400

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
        # Debug log removed for production
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
        # Debug log removed for production
        months_to_fetch = ceil(days / 30) + 1 if days > 0 else 1 # Fetch at least 1 month, or enough for 'days' + buffer
        success, message = data_services.fetch_and_store_stock(formatted_symbol, months=months_to_fetch, market=market)
        if success:
            # Debug log removed for production
            records = data_services.get_stored_stock_data(company_symbol=formatted_symbol, limit=limit)
            records = sorted(records, key=lambda r: r.date)
        else:
            # Debug log removed for production
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
    return jsonify({"success": False, "message": "Could not retrieve company info."}), 200

# NEW: API route for fetching stock statistics (for dashboard preview details)
@app.route('/api/stock_statistics/<symbol>', methods=['GET'])
def api_stock_statistics(symbol):
    stats = data_services.get_stock_statistics(symbol, days=1) # Get today's stats for current, open, high, volume
    if stats:
        return jsonify({"success": True, "data": stats})
    return jsonify({"success": False, "message": "Could not retrieve stock statistics."}), 404

@app.route('/stock/predict/<symbol>', methods=['GET'])
@rate_limit(max_requests=5, window=300)  # 5 requests per 5 minutes for predictions
def predict_stock(symbol):
    # Input validation
    if not symbol or len(symbol) > 10 or not symbol.replace('.', '').replace('-', '').isalnum():
        return jsonify({'success': False, 'message': 'Invalid stock symbol'}), 400
        
    horizon = request.args.get('horizon', 'month') # Default to 'month' for 30-day prediction
    if horizon not in ['day', 'week', 'month', '3month']:
        return jsonify({'success': False, 'message': 'Invalid horizon. Must be day, week, month, or 3month'}), 400
        
    market = request.args.get('market', 'US')
    if market not in ['US', 'IN']:
        return jsonify({'success': False, 'message': 'Invalid market. Must be US or IN'}), 400
    formatted_symbol = data_services.format_symbol(symbol, market)
    
    # Create cache key
    cache_key = f"{formatted_symbol}_{horizon}"
    
    # Check cache first
    if cache_key in prediction_cache:
        # Debug log removed for production
        return jsonify({"success": True, "prediction": prediction_cache[cache_key]})
    
    # Ensure sufficient data is in DB for prediction.
    # The LSTM model typically needs a good amount of historical data (e.g., 240-360 days)
    # Check if we have at least 400 days (approx. 13-14 months) for robust prediction.
    min_prediction_data_days = 250
    db_records_for_pred = data_services.get_stored_stock_data(company_symbol=formatted_symbol, limit=min_prediction_data_days)

    if not db_records_for_pred or len(db_records_for_pred) < min_prediction_data_days * 0.9: # If significantly less data
        # Debug log removed for production
        success, fetch_message = data_services.fetch_and_store_stock(formatted_symbol, months=18, market=market)
        if not success:
            return jsonify({'success': False, 'message': f"Prediction failed due to insufficient historical data: {fetch_message}"}), 500
        # Debug log removed for production
        # No need to re-fetch db_records_for_pred here, as prediction_service.lstm_predict_multiple
        # will internally call get_data_from_db which will get the newly stored data.

    predictions_data, error_message = prediction_service.lstm_predict_multiple(formatted_symbol, horizon=horizon)

    if error_message:
        # Debug log removed for production
        return jsonify({"success": False, "message": error_message}), 400

    if predictions_data:
        # Cache the prediction
        prediction_cache[cache_key] = predictions_data
        return jsonify({"success": True, "prediction": predictions_data})

    return jsonify({"success": False, "message": "Prediction could not be generated."}), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)
