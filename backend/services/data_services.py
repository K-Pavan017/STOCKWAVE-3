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
def get_historical_data(company_symbol, months=12, market='US'):
    try:
        symbol = format_symbol(company_symbol, market)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        ticker = yf.Ticker(symbol)

        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            print(f"[NO DATA] for {symbol}")
            return None

        data.dropna(inplace=True)
        data.index = pd.to_datetime(data.index).date
        return data

    except Exception as e:
        print(f"[FETCH ERROR] {company_symbol}: {e}")
        return None

# --- Store to DB ---
def store_stock_data(company_symbol, data):
    if data is None or data.empty:
        print("[EMPTY DATA] Nothing to store")
        return False

    try:
        records_added, records_updated = 0, 0
        for index, row in data.iterrows():
            try:
                open_price = round(float(row['Open']), 2) if pd.notna(row['Open']) else None
                high_price = round(float(row['High']), 2) if pd.notna(row['High']) else None
                low_price = round(float(row['Low']), 2) if pd.notna(row['Low']) else None
                close_price = round(float(row['Close']), 2) if pd.notna(row['Close']) else None
                volume = int(row['Volume']) if pd.notna(row['Volume']) else 0
            except Exception as e:
                print(f"[PARSE ERROR] {company_symbol} @ {index}: {e}")
                continue

            if all(x is None for x in [open_price, high_price, low_price, close_price]):
                continue

            existing = StockData.query.filter_by(company_symbol=company_symbol, date=index).first()
            if existing:
                existing.open_price = open_price
                existing.high_price = high_price
                existing.low_price = low_price
                existing.close_price = close_price
                existing.volume = volume
                existing.updated_at = db.func.now()
                records_updated += 1
            else:
                record = StockData(
                    company_symbol=company_symbol,
                    date=index,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume
                )
                db.session.add(record)
                records_added += 1

        db.session.commit()
        print(f"[DB] {records_added} added, {records_updated} updated for {company_symbol}")
        return True

    except Exception as e:
        db.session.rollback()
        print(f"[DB ERROR] {company_symbol}: {e}")
        return False

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
        return query.all()
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
            print(f"[NO STATS] No records for {company_symbol}")
            return None

        prices, volumes, daily_changes = [], [], []

        for record in reversed(records):
            if record.close_price is not None:
                prices.append(record.close_price)
                if record.open_price and record.open_price != 0:
                    daily_changes.append((record.close_price - record.open_price) / record.open_price * 100)
            if record.volume:
                volumes.append(record.volume)

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
