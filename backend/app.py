from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from database import db, init_app
from services import auth_service, data_services
from models import users
from models.stock_data import StockData
from services import prediction_service

from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
init_app(app)

@app.route('/signup', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400

    return auth_service.register_user(username, email, password)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'success': False, 'message': 'Missing email or password'}), 400

    return auth_service.login_user(email, password)

@app.route('/stock/fetch', methods=['POST'])
def fetch_and_store_stock():
    data = request.get_json()
    symbol = data.get('symbol')
    months = int(data.get('months', 3))
    if not symbol:
        return jsonify({'success': False, 'message': 'Missing symbol'}), 400

    df = data_services.get_historical_data(symbol, months)
    if df is None or df.empty:
        return jsonify({'success': False, 'message': 'Failed to fetch data'}), 404

    stored = data_services.store_stock_data(symbol, df)
    stats = data_services.get_stock_statistics(symbol, days=months*30)
    return jsonify({'success': stored, 'data': {'statistics': stats}})

@app.route('/stock/data/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    limit = int(request.args.get('limit', 90))
    days = int(request.args.get('days', 90))
    records = data_services.get_stored_stock_data(symbol, limit=limit)
    stats = data_services.get_stock_statistics(symbol, days=days)
    if not records:
        return jsonify({'success': False, 'message': 'No data found'}), 404

    data = [{
        'date': r.date.strftime('%Y-%m-%d'),
        'open': r.open_price,
        'high': r.high_price,
        'low': r.low_price,
        'close': r.close_price,
        'volume': r.volume
    } for r in records[::-1]]  # chronological order

    return jsonify({'success': True, 'data': {'records': data, 'statistics': stats}})

@app.route('/stock/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    # The 'horizon' parameter is now primarily for consistency, but the prediction_service
    # is modified to always return 30 days for 'month' or default.
    horizon = request.args.get('horizon', 'month') # Default to 'month' for 30-day prediction

    # Call the updated lstm_predict_multiple function
    predictions_data, error_message = prediction_service.lstm_predict_multiple(symbol, horizon=horizon)

    if error_message:
        return jsonify({"success": False, "message": error_message}), 400
    
    if predictions_data:
        return jsonify({"success": True, "prediction": predictions_data})
    
    return jsonify({"success": False, "message": "Prediction could not be generated."}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)