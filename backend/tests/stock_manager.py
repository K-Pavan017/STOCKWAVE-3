import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Date, DateTime, func, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine('sqlite:///stock_data.db', echo=False)
Session = sessionmaker(bind=engine)

class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    company_symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('company_symbol', 'date', name='_company_date_uc'),
    )

    def __repr__(self):
        return f'<StockData {self.company_symbol} - {self.date}>'

def create_db_model():
    """Create database tables"""
    Base.metadata.create_all(engine)

def fetch_stock_data(symbol: str = "AAPL", months: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance for the specified period.
    
    Args:
        symbol (str): Stock symbol to fetch (default: AAPL)
        months (int): Number of months of historical data to fetch (default: 3)
    
    Returns:
        pd.DataFrame: Stock data or None if fetching fails
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        # Download stock data
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Add symbol column
        data['company_symbol'] = symbol
        
        # Rename columns to match database schema
        data.rename(columns={
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume'
        }, inplace=True)
        
        # Convert date to date format (remove time component)
        data['date'] = pd.to_datetime(data['date']).dt.date
        
        # Select only the columns we need
        columns_needed = ['company_symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        data = data[columns_needed]
        
        # Handle NaN values
        data = data.dropna()
        
        return data
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def store_stock_data(data: pd.DataFrame) -> bool:
    """
    Store stock data in database using SQLAlchemy ORM.
    
    Args:
        data (pd.DataFrame): Stock data to store
    
    Returns:
        bool: True if successful, False otherwise
    """
    if data is None or data.empty:
        print("No data to store")
        return False
    
    session = Session()
    
    try:
        records_added = 0
        records_updated = 0
        
        for _, row in data.iterrows():
            # Check if record already exists
            existing_record = session.query(StockData).filter_by(
                company_symbol=row['company_symbol'],
                date=row['date']
            ).first()
            
            if existing_record:
                # Update existing record
                existing_record.open_price = row['open_price']
                existing_record.high_price = row['high_price']
                existing_record.low_price = row['low_price']
                existing_record.close_price = row['close_price']
                existing_record.volume = row['volume']
                existing_record.updated_at = func.now()
                records_updated += 1
            else:
                # Create new record
                stock_record = StockData(
                    company_symbol=row['company_symbol'],
                    date=row['date'],
                    open_price=row['open_price'],
                    high_price=row['high_price'],
                    low_price=row['low_price'],
                    close_price=row['close_price'],
                    volume=row['volume']
                )
                session.add(stock_record)
                records_added += 1
        
        session.commit()
        print(f"Successfully processed {records_added} new records and updated {records_updated} existing records")
        return True
        
    except Exception as e:
        session.rollback()
        print(f"Error storing data: {e}")
        return False
    finally:
        session.close()

def print_stock_data(symbol: str = "AAPL", limit: int = None, sort_by_date: bool = True):
    """
    Print stock data from database in a formatted table.
    
    Args:
        symbol (str): Stock symbol to display (default: AAPL)
        limit (int): Maximum number of records to display (None for all)
        sort_by_date (bool): Sort by date (newest first if True)
    """
    session = Session()
    
    try:
        # Build query
        query = session.query(StockData).filter_by(company_symbol=symbol)
        
        # Sort by date
        if sort_by_date:
            query = query.order_by(StockData.date.desc())
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        records = query.all()
        
        if not records:
            print(f"No data found for symbol: {symbol}")
            return
        
        # Print header
        print(f"\n{'='*100}")
        print(f"📈 STOCK DATA FOR {symbol.upper()}")
        print(f"{'='*100}")
        print(f"{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<15} {'Last Updated':<20}")
        print(f"{'-'*100}")
        
        # Print data rows
        for record in records:
            volume_str = f"{record.volume:,}" if record.volume else "N/A"
            updated_str = record.updated_at.strftime("%Y-%m-%d %H:%M") if record.updated_at else "N/A"
            
            print(f"{record.date} "
                  f"${record.open_price:<9.2f} "
                  f"${record.high_price:<9.2f} "
                  f"${record.low_price:<9.2f} "
                  f"${record.close_price:<9.2f} "
                  f"{volume_str:<15} "
                  f"{updated_str}")
        
        print(f"{'-'*100}")
        print(f"Total records displayed: {len(records)}")
        
        # Show statistics
        if records:
            prices = [r.close_price for r in records if r.close_price]
            volumes = [r.volume for r in records if r.volume]
            
            if prices:
                print(f"\n📊 STATISTICS:")
                print(f"   Highest Close: ${max(prices):.2f}")
                print(f"   Lowest Close:  ${min(prices):.2f}")
                print(f"   Average Close: ${sum(prices)/len(prices):.2f}")
            
            if volumes:
                print(f"   Average Volume: {sum(volumes)//len(volumes):,}")
                print(f"   Highest Volume: {max(volumes):,}")
        
    except Exception as e:
        print(f"Error retrieving data: {e}")
    finally:
        session.close()

def print_all_symbols(limit_per_symbol: int = 5):
    """
    Print data for all symbols in the database.
    
    Args:
        limit_per_symbol (int): Number of recent records to show per symbol
    """
    session = Session()
    
    try:
        # Get all unique symbols
        symbols = session.query(StockData.company_symbol).distinct().all()
        symbols = [s[0] for s in symbols]
        
        if not symbols:
            print("No stock data found in database")
            return
        
        print(f"\n{'='*50}")
        print(f"📋 ALL SYMBOLS IN DATABASE")
        print(f"{'='*50}")
        
        for symbol in symbols:
            print_stock_data(symbol, limit=limit_per_symbol, sort_by_date=True)
            print()
        
    except Exception as e:
        print(f"Error retrieving symbols: {e}")
    finally:
        session.close()

def main():
    """Main function to orchestrate the stock data download and storage process."""
    try:
        # Step 1: Create database and tables
        print("Creating database model...")
        create_db_model()
        
        # Step 2: Fetch stock data
        print("Fetching AAPL stock data...")
        stock_data = fetch_stock_data("AAPL", 3)
        
        if stock_data is None:
            print("Failed to fetch stock data. Exiting.")
            return
        
        print(f"Fetched {len(stock_data)} records")
        
        # Step 3: Store data in database
        print("Storing data in database...")
        success = store_stock_data(stock_data)
        
        if success:
            print("Process completed successfully!")
            
            # Step 4: Display the stored data
            print_stock_data("AAPL", limit=10)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()