from flask import Flask, request, jsonify
from flask_cors import CORS
# Assuming config.py exists and defines a Config class
from config import Config
from database import db # Assuming database.py only exports 'db' (SQLAlchemy instance)
from models.stock_data import StockData # Ensure this is imported for db.create_all
from services import auth_service, data_services, prediction_service # Assuming these service modules exist

from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config) # Load configuration from Config object
CORS(app) # Enable CORS for all routes

# Initialize SQLAlchemy with the Flask app
db.init_app(app)

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
    limit = int(request.args.get('limit', 365))
    days = int(request.args.get('days', 365))
    
    # It's better to pass start and end dates directly to get_stored_stock_data if `days` is used
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days) if days else None

    # Fetch records based on the calculated date range, or just limit if days is not specified meaningfully
    records = data_services.get_stored_stock_data(company_symbol=symbol, start_date=start_date, end_date=end_date, limit=limit)
    stats = data_services.get_stock_statistics(symbol, days=days) # Ensure days is passed to statistics for relevant range

    if not records:
        return jsonify({'success': False, 'message': 'No data found'}), 404

    data = [{
        'date': r.date.strftime('%Y-%m-%d'),
        'open': r.open_price,
        'high': r.high_price,
        'low': r.low_price,
        'close': r.close_price,
        'volume': r.volume
    } for r in records[::-1]]  # Ensure chronological order for charts

    return jsonify({'success': True, 'data': {'records': data, 'statistics': stats}})

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
    if stats: # <--- This is the key line
        return jsonify({"success": True, "data": stats})
    return jsonify({"success": False, "message": "Could not retrieve stock statistics."}), 404 
@app.route('/stock/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    horizon = request.args.get('horizon', 'month') # Default to 'month' for 30-day prediction

    predictions_data, error_message = prediction_service.lstm_predict_multiple(symbol, horizon=horizon)

    if error_message:
        print(f"Prediction Error for {symbol}: {error_message}")
        return jsonify({"success": False, "message": error_message}), 400
    
    if predictions_data:
        return jsonify({"success": True, "prediction": predictions_data})
    
    return jsonify({"success": False, "message": "Prediction could not be generated."}), 500

if __name__ == '__main__':
    # Ensure tables are created when running app.py directly
    with app.app_context():
        db.create_all()
    # Host on '0.0.0.0' to be accessible from other devices on the network, if needed
    app.run(debug=True, host='0.0.0.0', port=5000)