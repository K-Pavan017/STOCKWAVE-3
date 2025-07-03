import yfinance as yf
from models.stock_data import StockData
from database import db
from datetime import datetime, timedelta
import pandas as pd

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
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')
        return not hist.empty
    except Exception as e:
        print(f"[VALIDATION ERROR] {symbol}: {e}")
        return False

# --- Fetch Historical Data ---
def get_historical_data(company_symbol, months=24, market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30) # Using 30 days per month for approximation
        
        print(f"\n--- Inside Flask App's get_historical_data ---")
        print(f"Requesting data for symbol: {symbol}")
        print(f"Calculated start_date: {start_date.strftime('%Y-%m-%d')}")
        print(f"Calculated end_date: {end_date.strftime('%Y-%m-%d')}")
        print(f"Months parameter received: {months}")
        print(f"--- Calling yfinance.Ticker to fetch data ---")

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        print(f"[YFINANCE FETCH] {symbol}: Fetched {len(data)} records from yfinance (raw).")
        
        if data.empty:
            print(f"[NO DATA] for {symbol} - yfinance returned an empty DataFrame.")
            return None
        
        # Drop rows with any NaN values
        data.dropna(inplace=True)
        print(f"[YFINANCE FETCH] {symbol}: {len(data)} records after dropping NaN values.")

        # Ensure index is in date format
        data.index = pd.to_datetime(data.index).date
        
        print(f"[YFINANCE FETCH] {symbol}: Data preparation complete. Returning DataFrame with {len(data)} records.")
        return data

    except Exception as e:
        print(f"[FETCH ERROR] {company_symbol}: {e}")
        return None

# --- Store to DB ---
def store_stock_data(company_symbol, data):
    if data is None or data.empty:
        print("[EMPTY DATA] Nothing to store in DB for records.")
        return False

    try:
        records_added, records_updated = 0, 0
        
        # Optimize: Delete existing records for the company symbol BEFORE iterating
        # This prevents issues with duplicate primary keys if dates overlap
        print(f"[DB DELETE] Deleting all existing records for {company_symbol} before fresh insert.")
        StockData.query.filter_by(company_symbol=company_symbol).delete()
        db.session.commit() # Commit the deletion immediately

        for index, row in data.iterrows():
            try:
                # Use .get() with default None to safely access columns,
                # then round or convert only if not None.
                open_price = round(float(row.get('Open')), 2) if pd.notna(row.get('Open')) else None
                high_price = round(float(row.get('High')), 2) if pd.notna(row.get('High')) else None
                low_price = round(float(row.get('Low')), 2) if pd.notna(row.get('Low')) else None
                close_price = round(float(row.get('Close')), 2) if pd.notna(row.get('Close')) else None
                volume = int(row.get('Volume')) if pd.notna(row.get('Volume')) else 0
            except Exception as e:
                print(f"[PARSE ERROR] {company_symbol} @ {index}: Could not parse data row. Error: {e}")
                continue

            # Skip if all essential price data is missing for a row
            if all(x is None for x in [open_price, high_price, low_price, close_price]):
                print(f"[SKIP ROW] Skipping row for {company_symbol} @ {index} due to missing price data.")
                continue

            # Create a new record as we deleted all existing data for this symbol
            record = StockData(
                company_symbol=company_symbol,
                date=index, # Index is already a date object due to data.index = pd.to_datetime(data.index).date
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume
            )
            db.session.add(record)
            records_added += 1

        db.session.commit()
        print(f"[DB COMMIT] Successfully added {records_added} records for {company_symbol}. (No updates in this strategy)")
        return True

    except Exception as e:
        db.session.rollback()
        print(f"[DB ERROR] {company_symbol}: Failed to save stock data to DB. Error: {e}")
        return False

# --- NEW FUNCTION: Fetch and Store Stock Data (Orchestrator) ---
def fetch_and_store_stock(symbol, months=18, market='US'):
    """
    Orchestrates fetching historical stock data and storing it in the database.
    """
    print(f"\n--- fetch_and_store_stock called for {symbol}, {months} months, market {market} ---")
    if not validate_stock_symbol(symbol, market):
        print(f"Validation failed for {symbol}.")
        return False, f"Invalid or unsupported stock symbol: {symbol}"

    print(f"Fetching {months} months data for {symbol} from yfinance...")
    data = get_historical_data(symbol, months=months, market=market)

    if data is None or data.empty:
        print(f"No data returned from get_historical_data for {symbol}.")
        return False, f"Could not fetch historical data for {symbol}."

    print(f"Attempting to store {len(data)} records for {symbol} in the database...")
    success = store_stock_data(symbol, data)

    if success:
        print(f"Successfully fetched and stored {len(data)} records for {symbol}.")
        return True, f"Successfully fetched and stored {len(data)} records for {symbol}."
    else:
        print(f"Failed to store data for {symbol}.")
        return False, f"Failed to store data for {symbol}."


# --- Retrieve Stored Data ---
def get_stored_stock_data(company_symbol, start_date=None, end_date=None, limit=None):
    try:
        query = StockData.query.filter_by(company_symbol=company_symbol)
        if start_date:
            query = query.filter(StockData.date >= start_date)
        if end_date:
            query = query.filter(StockData.date <= end_date)
        query = query.order_by(StockData.date.desc())
        if limit:
            query = query.limit(limit)
        
        records = query.all()
        print(f"[DB READ] Fetched {len(records)} records for {company_symbol} from DB.")
        return records
    except Exception as e:
        print(f"[FETCH ERROR] {company_symbol}: {e}")
        return None

# --- Calculate Statistics ---
def get_stock_statistics(company_symbol, days=30):
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        records = get_stored_stock_data(company_symbol, start_date, end_date)

        if not records:
            print(f"[NO STATS] No records for {company_symbol} in the last {days} days.")
            return None

        prices, volumes, daily_changes = [], [], []

        for record in reversed(records): # Process in chronological order
            if record.close_price is not None:
                prices.append(record.close_price)
                if record.open_price and record.open_price != 0:
                    daily_changes.append((record.close_price - record.open_price) / record.open_price * 100)
            if record.volume:
                volumes.append(record.volume)

        if not prices:
            return None

        stats = {
            'symbol': company_symbol,
            'total_records': len(records),
            'date_range': {
                'start': records[-1].date.strftime('%Y-%m-%d'),
                'end': records[0].date.strftime('%Y-%m-%d')
            },
            'price_stats': {
                'current': round(prices[-1], 2),
                'opening': round(prices[0], 2),
                'highest': round(max(prices), 2),
                'lowest': round(min(prices), 2),
                'average': round(sum(prices) / len(prices), 2),
                'change': round(prices[-1] - prices[0], 2),
                'change_percent': round((prices[-1] - prices[0]) / prices[0] * 100, 2)
            } if prices else {},
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
def get_company_info(company_symbol, market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        ticker = yf.Ticker(symbol)

        # fallback to safe attrs
        fast_info = ticker.fast_info or {}
        return {
            'symbol': symbol,
            'currency': fast_info.get('currency', 'USD'),
            'exchange': fast_info.get('exchange', 'N/A'),
            'last_price': fast_info.get('last_price', None),
            'market': market
        }

    except Exception as e:
        print(f"[INFO ERROR] {company_symbol}: {e}")
        return None