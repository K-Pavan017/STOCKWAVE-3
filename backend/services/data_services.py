import requests
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
        url = f"https://finnhub.io/api/v1/search?q={symbol}&token={Config.FINNHUB_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return 'result' in data and len(data['result']) > 0
    except Exception as e:
        print(f"[VALIDATION ERROR] {symbol}: {e}")
        return False

def get_historical_data(company_symbol, months=None, days=None, period_type='months', market='US'):
    try:
        symbol = format_symbol(company_symbol, market)

        # Alpha Vantage provides daily data
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=365 * 2)).timestamp())
        
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start}&to={end}&token={Config.FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"[ALPHA VANTAGE] API Error: {response.status_code}")
            return None

        data = response.json()

        # Handle Alpha Vantage limits/errors

        if data.get("s") != "ok":
            print(f"[FINNHUB ERROR] {data}")
            return None
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        })
        
        df = df.sort_values('Date').reset_index(drop=True)

        df = pd.DataFrame(df_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Filter by date range if specified
        if months is not None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)
            df = df[df['Date'] >= start_date]
        elif days is not None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = df[df['Date'] >= start_date]

        return df

    except Exception as e:
        print(f"[ALPHA VANTAGE ERROR] {symbol}: {e}")
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
            # 2. If no recent records in DB, fetch live historical data from Alpha Vantage for the period
            print(f"[NO DB DATA FOR STATS] Fetching live historical data for {symbol} for {days} days from Finnhub.")
            
            alphavantage_df = get_historical_data(company_symbol, days=days, period_type='days', market=market)
            
            if alphavantage_df is not None and not alphavantage_df.empty:
                # Convert DataFrame rows to a list of dicts that our stat calculation can use
                for index, row in alphavantage_df.iterrows():
                    records_to_process.append({
                        'date': row['Date'].date(), # Convert timestamp to date object
                        'open_price': row['Open'],
                        'close_price': row['Close'],
                        'high_price': row['High'],
                        'low_price': row['Low'],
                        'volume': row['Volume']
                    })
                # Ensure chronological order for alphavantage data if not already (DataFrame is sorted)
                records_to_process = sorted(records_to_process, key=lambda r: r['date'])
                source_tag = "FINNHUB_LIVE"
            else:
                print(f"[NO DATA FOR STATS] No historical data found for {symbol} from DB or Finnhub for last {days} days.")
                return None # Still no data, return None

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
                prices.append(r.close_price)
                opening_prices.append(r.open_price)
                volumes.append(r.volume)
                high_prices.append(r.high_price)
                low_prices.append(r.low_price)
            else: # Must be a dictionary from yfinance
                prices.append(r['close_price'])
                opening_prices.append(r['open_price'])
                volumes.append(r['volume'])
                high_prices.append(r['high_price'])
                low_prices.append(r['low_price'])

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
def get_company_info(company_symbol, market='US'):
    symbol = format_symbol(company_symbol, market)

    if symbol in stock_cache:
        return stock_cache[symbol]

    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={Config.FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()

        # 🔴 Rate limit
        if "Note" in data:
            print("RATE LIMIT HIT")
            return {"error": "rate_limit"}

        # 🔴 API error
        if "Error Message" in data:
            return None

        quote = data.get("Global Quote")

        # 🔴 Fallback if empty
        if not quote or quote.get("05. price") in [None, "", "0"]:
            print("FALLBACK TRIGGERED")

            # fallback: use your DB latest record
            records = get_stored_stock_data(symbol, limit=1)
            if records:
                r = records[0]
                return {
                    "symbol": symbol,
                    "current_price": r.close_price,
                    "fallback": "db"
                }

            return None

        current_price = data.get("c")   # current price
        previous_close = data.get("pc") # previous close

        result = {
            "symbol": symbol,
            "current_price": current_price,
            "previous_close": previous_close,
            "day_change": round(current_price - previous_close, 2),
            "day_change_percent": round(((current_price - previous_close) / previous_close) * 100 if previous_close else 0, 2)
        }

        stock_cache[symbol] = result
        return result

    except Exception as e:
        print("COMPANY INFO ERROR:", e)
        return None
