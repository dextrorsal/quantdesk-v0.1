"""
Database Loader - Simple interface to load data from your Supabase/Neon database
Makes it as easy as loading CSV files, but with the power of SQL

USE CASES:
- **Simple data loading**: Load data from database as easily as pd.read_csv()
- **Model training**: Load data for ML model training and backtesting
- **Data exploration**: Quick data loading for analysis and exploration
- **Database queries**: Simple interface to query your trading data
- **Data availability check**: See what data is available in your database
- **Time range filtering**: Load data for specific date ranges

DIFFERENCES FROM OTHER LOADERS/FETCHERS:
- DatabaseLoader: Simple database loading interface (like pd.read_csv)
- UltimateDataFetcher: Multi-exchange orchestrator with database storage
- ReliableDataFetcher: CCXT-based, fast exchanges only, database storage
- DailyDataFetcher: Automated daily CSV storage with multiple exchanges
- CSVDataFetcher: Manual CSV storage, no database integration
- DataLoader: CSV-based data loading with file system storage

WHEN TO USE:
- When you want to load data from your database
- For model training and backtesting
- For data analysis and exploration
- When you need simple, CSV-like interface to database
- When you don't need to fetch new data (just load existing)

EXAMPLES:
    # Load data for model training
    loader = DatabaseLoader()
    df = await loader.load_data(
        symbol="BTC/USDT",
        resolution="15m", 
        exchange="binance",
        days=365
    )
    
    # Check available data
    available = await loader.get_available_data()
    
    # Convenience function
    df = await load_trading_data("BTC/USDT", "15m", "binance", 30)
"""

import os
import pandas as pd
import asyncpg
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class DatabaseLoader:
    """Simple loader for your trading data - makes SQL as easy as CSV"""
    
    def __init__(self, db_conn_str: Optional[str] = None):
        """Initialize with database connection"""
        self.db_conn_str = (
            db_conn_str or os.getenv('DATABASE_URL') or os.getenv('NEON_TEST_DB')
        )
        if not self.db_conn_str:
            raise ValueError("No database connection string found")
    
    async def load_data(
        self,
                       symbol: str = "BTC/USDT",
                       resolution: str = "15m",
                       exchange: str = "binance",
        days: int = 365
    ) -> pd.DataFrame:
        """
        Load data from database - as simple as pd.read_csv()
        """
        conn = await asyncpg.connect(self.db_conn_str)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query data
            query = (
                """
                SELECT 
                    c.ts as timestamp,
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume
                FROM market_data.candles c
                JOIN market_data.markets m ON c.market_id = m.id
                JOIN market_data.exchanges e ON m.exchange_id = e.id
                WHERE m.symbol = $1
                    AND c.resolution = $2
                    AND e.name = $3
                    AND c.ts >= $4
                ORDER BY c.ts ASC
            """
            )
            
            rows = await conn.fetch(query, symbol, resolution, exchange, start_date)
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)

            # Debug: print columns and head
            print("Columns returned:", df.columns)
            print(df.head())
            
            if df.empty:
                return df
            
            # Robustly handle time column
            if 'timestamp' not in df.columns and 'ts' in df.columns:
                df['timestamp'] = pd.to_datetime(df['ts'])
                df = df.rename(columns={'ts': 'timestamp'})
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                print(
                    "No 'timestamp' or 'ts' column found in DataFrame! Returning empty."
                )
                return pd.DataFrame()

            # Set timestamp as index (like CSV)
            df.set_index('timestamp', inplace=True)
            
            print(
                f"✅ Loaded {len(df)} candles for {symbol} {resolution} from {exchange}"
            )
            print(
                f"   Date range: {df.index.min()} to {df.index.max()}"
            )
            
            return df
            
        finally:
            await conn.close()
    
    async def get_available_data(self) -> pd.DataFrame:
        """Get summary of all available data"""
        conn = await asyncpg.connect(self.db_conn_str)
        
        try:
            query = (
                """
                SELECT 
                    m.symbol,
                    e.name as exchange,
                    c.resolution,
                    COUNT(*) as candle_count,
                    MIN(c.ts) as earliest_date,
                    MAX(c.ts) as latest_date
                FROM market_data.candles c
                JOIN market_data.markets m ON c.market_id = m.id
                JOIN market_data.exchanges e ON m.exchange_id = e.id
                GROUP BY m.symbol, e.name, c.resolution
                ORDER BY m.symbol, e.name, c.resolution
            """
            )
            
            rows = await conn.fetch(query)
            df = pd.DataFrame(rows)
            
            if not df.empty:
                print("📊 Available data in your database:")
                for _, row in df.iterrows():
                    print(
                        f"   {row['exchange']} {row['symbol']} {row['resolution']}: "
                          f"{row['candle_count']:,} candles "
                          f"({row['earliest_date'].strftime('%Y-%m-%d')} to "
                        f"{row['latest_date'].strftime('%Y-%m-%d')})"
                    )
            
            return df
            
        finally:
            await conn.close()


# Convenience function for easy loading
async def load_trading_data(
    symbol: str = "BTC/USDT",
                           resolution: str = "15m",
                           exchange: str = "binance",
    days: int = 365
) -> pd.DataFrame:
    """
    Simple function to load data - just like pd.read_csv()
    
    Example:
        df = await load_trading_data("BTC/USDT", "15m", "binance", 30)
    """
    loader = DatabaseLoader()
    return await loader.load_data(symbol, resolution, exchange, days) 