import requests
from yahooquery import Ticker
from models.stock_data import StockData
from database import db
from datetime import datetime, timedelta
import pandas as pd
from math import ceil
from config import Config
from dotenv import load_dotenv

from cachetools import TTLCache

# --- Symbol Formatting ---
def format_symbol(symbol, market='US'):
    """Format symbol for Indian or US stocks."""
    if market == 'IN' and not symbol.upper().endswith('.NS'):
        return f"{symbol.upper()}.NS"
    return symbol.upper()

# --- Validate Stock (lightweight check) ---
def validate_stock_symbol(company_symbol, market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        ticker = Ticker(symbol)
        hist = ticker.history(period="1d")
        if isinstance(hist, dict):
            return False
        return not hist.empty
    except Exception as e:
        print(f"[VALIDATION ERROR] {symbol}: {e}")
        return False

def get_historical_data(company_symbol, months=None, days=None, period_type='months', market='US'):
    try:
        symbol = format_symbol(company_symbol, market)

        ticker = Ticker(symbol)

        # Use start/end dates for precise control
        if months is not None:
            start_date = datetime.now() - timedelta(days=months * 30)
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'))
        elif days is not None:
            start_date = datetime.now() - timedelta(days=days)
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'))
        else:
            df = ticker.history(period="2y")

        if isinstance(df, dict) or df is None or df.empty:
            print(f"[YAHOOQUERY] No data found for {symbol}")
            return None

        # Reset index to get date as a column
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            
        # Rename yahooquery columns to match format
        df = df.rename(columns={
            'date': 'Date',
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Remove timezone info from Date column to avoid issues with DB comparisons
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Keep only the columns we need and drop any rows with NaN values (e.g. missing Volume data)
        req_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in req_cols):
             return None
             
        df = df[req_cols].dropna()

        return df

    except Exception as e:
        print(f"[YAHOOQUERY ERROR] {company_symbol}: {e}")
        return None


def get_stored_stock_data(company_symbol, start_date=None, end_date=None, limit=None):
    try:
        query = StockData.query.filter_by(company_symbol=company_symbol)
        
        if start_date:
            query = query.filter(StockData.date >= start_date)
        if end_date:
            query = query.filter(StockData.date <= end_date)

        query = query.order_by(StockData.date.desc()) # Order by date descending (latest first)

        if limit:
            records = query.limit(limit).all()
        else:
            records = query.all()
        
        return records
    except Exception as e:
        print(f"[DB READ ERROR] {company_symbol}: {e}")
        return []

def fetch_and_store_stock(company_symbol, months=18, market='US'):
    symbol = format_symbol(company_symbol, market)

    try:
        df = get_historical_data(company_symbol, months=months, period_type='months', market=market)

        if df is None or df.empty:
            return False, f"No data found for {symbol} from Alpha Vantage."

        # Convert dataframe index to date list
        dates = [date.date() for date in df['Date']]

        # 🔹 Fetch existing records in ONE query
        existing_records = StockData.query.filter(
            StockData.company_symbol == symbol,
            StockData.date.in_(dates)
        ).all()

        # 🔹 Map existing records by date
        existing_map = {record.date: record for record in existing_records}

        new_records = []
        updated_count = 0

        for index, row in df.iterrows():
            date = row['Date'].date()

            if date in existing_map:
                # Update existing record
                record = existing_map[date]

                record.open_price = float(row['Open'])
                record.high_price = float(row['High'])
                record.low_price = float(row['Low'])
                record.close_price = float(row['Close'])
                record.volume = int(row['Volume'])
                record.updated_at = datetime.utcnow()

                updated_count += 1

            else:
                # Create new record
                new_records.append(
                    StockData(
                        company_symbol=symbol,
                        date=date,
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=int(row['Volume']),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )

        # 🔹 Bulk insert new records
        if new_records:
            db.session.bulk_save_objects(new_records)

        db.session.commit()

        print(
            f"[DB WRITE] {symbol}: {len(new_records)} inserted, {updated_count} updated"
        )

        return True, f"Stored {len(new_records)} new records and updated {updated_count} for {symbol}"

    except Exception as e:
        db.session.rollback()
        print(f"[STORAGE ERROR] {symbol}: {e}")
        return False, f"Failed to fetch and store data for {symbol}: {e}"

# --- Get Stock Statistics ---
def get_stock_statistics(company_symbol, days=1, market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        
        records_to_process = []
        source_tag = "N/A"

        # 1. Try to get data from DB first for the specified 'days'
        end_date_for_db_query = datetime.now().date()
        start_date_for_db_query = end_date_for_db_query - timedelta(days=days)
        
        # get_stored_stock_data returns records sorted by date DESCENDING.
        # We need them chronological for daily_changes calculation.
        db_records = get_stored_stock_data(company_symbol=symbol, start_date=start_date_for_db_query, end_date=end_date_for_db_query)
        db_records_chronological = sorted(db_records, key=lambda r: r.date) if db_records else []

        if db_records_chronological:
            records_to_process = db_records_chronological
            source_tag = "DB"
        else:
            # 2. If no recent records in DB, fetch live historical data from yfinance for the period
            print(f"[NO DB DATA FOR STATS] Fetching live historical data for {symbol} for {days} days from yfinance.")
            
            yf_df = get_historical_data(company_symbol, days=days, period_type='days', market=market)
            
            if yf_df is not None and not yf_df.empty:
                # Convert DataFrame rows to a list of dicts that our stat calculation can use
                for index, row in yf_df.iterrows():
                    records_to_process.append({
                        'date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'], 
                        'open_price': row['Open'],
                        'close_price': row['Close'],
                        'high_price': row['High'],
                        'low_price': row['Low'],
                        'volume': row['Volume']
                    })
                # Ensure chronological order for yfinance data if not already (DataFrame is sorted)
                records_to_process = sorted(records_to_process, key=lambda r: r['date'])
                source_tag = "YFINANCE_LIVE"
            else:
                print(f"[NO DATA FOR STATS] No historical data found for {symbol} from DB or yfinance for last {days} days.")
                return None 

        if not records_to_process:
            return None # Safeguard if processing leads to empty list

        # Ensure that data access is compatible with both StockData objects and dictionaries
        prices = []
        opening_prices = []
        volumes = []
        high_prices = []
        low_prices = []

        for r in records_to_process:
            if isinstance(r, StockData):
                if r.close_price is not None: prices.append(r.close_price)
                if r.open_price is not None: opening_prices.append(r.open_price)
                if r.volume is not None: volumes.append(r.volume)
                if r.high_price is not None: high_prices.append(r.high_price)
                if r.low_price is not None: low_prices.append(r.low_price)
            else: # Must be a dictionary from yfinance
                if r.get('close_price') is not None: prices.append(r['close_price'])
                if r.get('open_price') is not None: opening_prices.append(r['open_price'])
                if r.get('volume') is not None: volumes.append(r['volume'])
                if r.get('high_price') is not None: high_prices.append(r['high_price'])
                if r.get('low_price') is not None: low_prices.append(r['low_price'])

        # Get latest day's specific values for current_price and opening_price
        latest_record = records_to_process[-1] # Last element is the most recent due to chronological sort
        
        if isinstance(latest_record, StockData):
            current_price = latest_record.close_price
            opening_price = latest_record.open_price
        else: # Must be a dictionary from yfinance
            current_price = latest_record.get('close_price')
            opening_price = latest_record.get('open_price')


        # Calculate daily changes based on daily close price
        daily_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                daily_change = (prices[i] - prices[i-1]) / prices[i-1] * 100
                daily_changes.append(daily_change)
            else:
                daily_changes.append(0)

        # Ensure that prices list is not empty before attempting max/min
        highest_price_period = max(high_prices) if high_prices else None
        lowest_price_period = min(low_prices) if low_prices else None

        stats = {
            'period_days': len(records_to_process), # Actual number of days for which stats are calculated
            'current_price': current_price,
            'opening_price': opening_price,
            'price_stats': {
                'open': opening_price,
                'current': current_price,
                'highest': highest_price_period,
                'lowest': lowest_price_period,
                'change_value': round(current_price - opening_price, 2) if current_price is not None and opening_price is not None else 0,
                'change_percent': round(((current_price - opening_price) / opening_price) * 100, 2) if current_price is not None and opening_price is not None and opening_price != 0 else 0
            } if current_price is not None else {}, # Ensure this dict is only created if current_price exists
            'volume_stats': {
                'average': int(sum(volumes) / len(volumes)) if volumes else 0,
                'highest': max(volumes) if volumes else 0,
                'total': sum(volumes) if volumes else 0
            },
            'performance_stats': {
                'positive_days': sum(1 for x in daily_changes if x > 0),
                'negative_days': sum(1 for x in daily_changes if x < 0),
                'positive_ratio': round(sum(1 for x in daily_changes if x > 0) / len(daily_changes) * 100, 1) if daily_changes else 0,
                'avg_daily_change': round(sum(daily_changes) / len(daily_changes), 2) if daily_changes else 0
            }
        }
        return stats
    except Exception as e:
        print(f"[STATS ERROR] {company_symbol}: {e}")
        return None

# --- Get Company Info (basic) ---

stock_cache = TTLCache(maxsize=500, ttl=300)  # 5 minutes
# get_company_info removed as requested to avoid rate limiting other requests
