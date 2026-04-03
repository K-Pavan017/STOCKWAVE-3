import sys
import os

# Ensure the backend directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.services import data_services
from backend.app import app

with app.app_context():
    print("Testing get_historical_data...")
    try:
        df = data_services.get_historical_data("AAPL", days=365, market="US")
        print(df.head() if df is not None else "DF is None")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("Testing fetch_and_store_stock...")
    try:
        res = data_services.fetch_and_store_stock("AAPL", months=12, market="US")
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    print("Testing get_stock_statistics...")
    try:
        stats = data_services.get_stock_statistics("AAPL", days=365, market="US")
        print(stats)
    except Exception as e:
        import traceback
        traceback.print_exc()
