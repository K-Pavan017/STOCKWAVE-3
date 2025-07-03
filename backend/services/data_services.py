import yfinance as yf
from models.stock_data import StockData
from database import db
from datetime import datetime, timedelta
import pandas as pd
from math import ceil

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
# Modified to accept days or months for period, and return a DataFrame
def get_historical_data(company_symbol, months=None, days=None, period_type='months', market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        ticker = yf.Ticker(symbol)

        period_str = "1mo" # Default period
        if period_type == 'days' and days is not None:
            # yfinance periods are limited, for days we request a slightly larger period to ensure enough data
            # e.g., if asking for 1 day, sometimes '1d' period might be empty for weekends/holidays.
            # Requesting '5d' is a common practice to get recent daily data.
            period_str = f"{max(days, 5)}d" # Get at least 5 days to be safe
        elif months is not None:
            period_str = f"{months}mo"
        
        hist = ticker.history(period=period_str, interval="1d") # Always get daily interval

        if hist.empty:
            print(f"[YFINANCE] No historical data found for {symbol} for period {period_str}.")
            return None
        return hist
    except Exception as e:
        print(f"[YFINANCE ERROR] {symbol}: {e}")
        return None

# --- Get Stored Stock Data ---
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
        
        print(f"[DB READ] Fetched {len(records)} records for {company_symbol} from DB.")
        return records
    except Exception as e:
        print(f"[DB READ ERROR] {company_symbol}: {e}")
        return []

# --- Fetch and Store Stock Data (from Hero.jsx) ---
def fetch_and_store_stock(company_symbol, months=18, market='US'):
    symbol = format_symbol(company_symbol, market)
    try:
        df = get_historical_data(company_symbol, months=months, period_type='months', market=market)
        if df is None or df.empty:
            return False, f"No data found for {symbol} from yfinance."

        # Delete existing data for the symbol to prevent duplicates for the same date
        # Note: This approach might be too aggressive for partial updates.
        # Consider a more sophisticated merge if only updating recent data.
        # For simplicity, if we fetch 18 months, we assume we overwrite/update that range.
        # Better: use db.session.merge or check for existence before adding.
        StockData.query.filter_by(company_symbol=symbol).delete()
        db.session.commit()
        print(f"[DB CLEANUP] Deleted existing records for {symbol}.")


        for index, row in df.iterrows():
            date = index.date() # Convert pandas timestamp to Python date
            
            # Check if record for this date already exists to avoid duplicates
            existing_record = StockData.query.filter_by(company_symbol=symbol, date=date).first()
            if existing_record:
                # Update existing record
                existing_record.open_price = row['Open']
                existing_record.high_price = row['High']
                existing_record.low_price = row['Low']
                existing_record.close_price = row['Close']
                existing_record.volume = row['Volume']
            else:
                # Add new record
                stock_data = StockData(
                    company_symbol=symbol,
                    date=date,
                    open_price=row['Open'],
                    high_price=row['High'],
                    low_price=row['Low'],
                    close_price=row['Close'],
                    volume=row['Volume']
                )
                db.session.add(stock_data)
        
        db.session.commit()
        print(f"[DB WRITE] Successfully fetched and stored {len(df)} records for {symbol}.")
        return True, f"Successfully fetched and stored data for {symbol}."
    except Exception as e:
        db.session.rollback() # Rollback on error
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
            print(f"[STATS SOURCE] {symbol}: Data from DB for {len(records_to_process)} days.")
        else:
            # 2. If no recent records in DB, fetch live historical data from yfinance for the period
            print(f"[NO DB DATA FOR STATS] Fetching live historical data for {symbol} for {days} days from YFinance.")
            
            yfinance_df = get_historical_data(company_symbol, days=days, period_type='days', market=market)
            
            if yfinance_df is not None and not yfinance_df.empty:
                # Convert DataFrame rows to a list of dicts that our stat calculation can use
                for index, row in yfinance_df.iterrows():
                    records_to_process.append({
                        'date': index.date(), # Convert timestamp to date object
                        'open_price': row['Open'],
                        'close_price': row['Close'],
                        'high_price': row['High'],
                        'low_price': row['Low'],
                        'volume': row['Volume']
                    })
                # Ensure chronological order for yfinance data if not already (DataFrame index usually is)
                records_to_process = sorted(records_to_process, key=lambda r: r['date'])
                source_tag = "YFINANCE_LIVE"
                print(f"[STATS SOURCE] {symbol}: Data from YFinance for {len(records_to_process)} days.")
            else:
                print(f"[NO DATA FOR STATS] No historical data found for {symbol} from DB or YFinance for last {days} days.")
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
def get_company_info(company_symbol, market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        ticker = yf.Ticker(symbol)

        # Use info from ticker.fast_info for common fields
        fast_info = ticker.fast_info or {}
        
        # Fallback to info for more details if fast_info is insufficient
        info = ticker.info or {} 

        # Current price and day's change from fast_info if available, otherwise fetch explicitly
        current_price = fast_info.get('lastPrice')
        previous_close = fast_info.get('previousClose')

        if current_price is None or previous_close is None:
            # If fast_info doesn't have it, try fetching a small history
            hist = ticker.history(period="1d", interval="1m") # Fetch 1-minute interval for current day
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                # Try to get previous close from 2-day history
                hist_2d = ticker.history(period="2d", interval="1d")
                if len(hist_2d) > 1:
                    previous_close = hist_2d['Close'].iloc[-2] # Second to last closing price
            else:
                # As a last resort, use info.currentPrice or info.previousClose
                current_price = info.get('currentPrice')
                previous_close = info.get('previousClose')
        
        day_change = None
        day_change_percent = None
        if current_price is not None and previous_close is not None and previous_close != 0:
            day_change = round(current_price - previous_close, 2)
            day_change_percent = round((day_change / previous_close) * 100, 2)

        return {
            'symbol': symbol,
            'currency': fast_info.get('currency', info.get('currency', 'USD')),
            'exchange': fast_info.get('exchange', info.get('exchange', 'N/A')),
            'shortName': fast_info.get('shortName', info.get('shortName', company_symbol)),
            'longName': info.get('longName', fast_info.get('longName', company_symbol)),
            'current_price': current_price,
            'previous_close': previous_close,
            'day_change': day_change,
            'day_change_percent': day_change_percent,
            'market_cap': fast_info.get('marketCap', info.get('marketCap')),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'website': info.get('website'),
            'beta': info.get('beta')
        }
    except Exception as e:
        print(f"[GET COMPANY INFO ERROR] {company_symbol}: {e}")
        return None