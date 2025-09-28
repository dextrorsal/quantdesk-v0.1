#!/usr/bin/env python3
"""
Reliable Data Fetcher - Uses CCXT to fetch data from fast, reliable exchanges
Avoids Bitget due to slow API and 200 candle limit

USE CASES:
- **Fast, reliable data fetching**: Uses only fast exchanges (Binance, Coinbase)
- **Database storage**: Stores data directly to PostgreSQL/Neon database
- **Bulk historical data**: Fetches large amounts of historical data efficiently
- **Rate limit handling**: Built-in rate limiting and error handling
- **Progress tracking**: Visual progress bars for long-running fetches
- **Data deduplication**: Handles duplicate data with ON CONFLICT DO NOTHING


DIFFERENCES FROM OTHER FETCHERS:
- ReliableDataFetcher: CCXT-based, fast exchanges only, database storage
- UltimateDataFetcher: Multi-exchange orchestrator with WebSocket support
- DailyDataFetcher: Automated daily CSV storage with multiple exchanges
- CSVDataFetcher: CSV-only storage, no database integration
- DatabaseLoader: Simple database loading interface (like pd.read_csv)


WHEN TO USE:
- When you need fast, reliable data from major exchanges
- For bulk historical data fetching (1+ year of data)
- When you want database storage with deduplication
- When you don't need real-time streaming or Bitget data

EXAMPLES:
    # Initialize and fetch 1 year of data
    fetcher = ReliableDataFetcher(db_conn_str)
    await fetcher.init_db()
    fetcher.init_exchanges()
    await fetcher.fetch_all_data()
    
    # Fetch specific symbol and timeframe
    await fetcher.fetch_symbol_data(
        exchange_id="binance",
        symbol="BTC/USDT", 
        timeframe="15m",
        days=365
    )
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ccxt
import pandas as pd
import asyncpg
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
# Fast, reliable exchanges
EXCHANGES = ["binance", "coinbase"]
DAYS_TO_FETCH = 365  # 1 year of data

class ReliableDataFetcher:
    def __init__(self, db_conn_str: str):
        self.db_conn_str = db_conn_str
        self.exchanges = {}
        self.db_pool = None
        
    async def init_db(self):
        """Initialize database connection pool"""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(self.db_conn_str)
            logger.info("Database connection pool initialized")
    
    def init_exchanges(self):
        """Initialize exchange connections"""
        for exchange_id in EXCHANGES:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                exchange.load_markets()
                self.exchanges[exchange_id] = exchange
                logger.info(f"✅ Initialized {exchange_id}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
    
    async def get_or_create_exchange(self, exchange_name: str) -> int:
        """Get or create exchange record"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM market_data.exchanges WHERE name = $1", 
                exchange_name
            )
            if row:
                return row["id"]
            else:
                row = await conn.fetchrow(
                    "INSERT INTO market_data.exchanges (name) VALUES ($1) RETURNING id",
                    exchange_name
                )
                return row["id"]
    
    async def get_or_create_market(self, exchange_id: int, symbol: str) -> int:
        """Get or create market record"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM market_data.markets WHERE exchange_id = $1 AND symbol = $2",
                exchange_id, symbol
            )
            if row:
                return row["id"]
            else:
                # Parse symbol for base/quote assets
                if "/" in symbol:
                    base_asset, quote_asset = symbol.split("/")
                else:
                    base_asset = symbol[:-4] if symbol.endswith("USDT") else symbol
                    quote_asset = "USDT"
                
                row = await conn.fetchrow(
                    """INSERT INTO market_data.markets 
                       (exchange_id, symbol, base_asset, quote_asset, type) 
                       VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                    exchange_id, symbol, base_asset, quote_asset, "SPOT"
                )
                return row["id"]
    
    async def store_candles(self, market_id: int, resolution: str, df: pd.DataFrame):
        """Store candles in database"""
        if df.empty:
            return
        # Ensure timestamps are timezone-aware (UTC)
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        records = []
        for _, row in df.iterrows():
            records.append((
                market_id,
                resolution,
                row['timestamp'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        async with self.db_pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO market_data.candles 
                   (market_id, resolution, ts, open, high, low, close, volume)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (market_id, resolution, ts) DO NOTHING""",
                records
            )
    
    async def fetch_symbol_data(self, exchange_id: str, symbol: str, timeframe: str, days: int):
        """Fetch data for a specific symbol and timeframe"""
        exchange = self.exchanges[exchange_id]
        
        try:
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            logger.info(f"Fetching {symbol} {timeframe} data from {exchange_id}...")
            
            # Get exchange and market IDs
            db_exchange_id = await self.get_or_create_exchange(exchange_id)
            db_market_id = await self.get_or_create_market(db_exchange_id, symbol)
            
            # Fetch data in chunks to avoid rate limits
            all_candles = []
            since = int(start_time.timestamp() * 1000)
            
            with tqdm(desc=f"{exchange_id}:{symbol}:{timeframe}", unit="candles") as pbar:
                while since < int(end_time.timestamp() * 1000):
                    try:
                        ohlcv = exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=since,
                            limit=1000
                        )
                        
                        if not ohlcv:
                            break
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(
                            ohlcv,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Store in database
                        await self.store_candles(db_market_id, timeframe, df)
                        
                        all_candles.extend(ohlcv)
                        pbar.update(len(ohlcv))
                        
                        # Update since timestamp
                        if len(ohlcv) > 0:
                            since = ohlcv[-1][0] + 1
                        else:
                            break
                        
                        # Rate limiting
                        time.sleep(exchange.rateLimit / 1000)
                        
                    except Exception as e:
                        logger.error(f"Error fetching {symbol} {timeframe} from {exchange_id}: {e}")
                        break
            
            logger.info(f"✅ Fetched {len(all_candles)} {timeframe} candles for {symbol} from {exchange_id}")
            return len(all_candles)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe} from {exchange_id}: {e}")
            return 0
    
    async def fetch_all_data(self):
        """Fetch all data for all symbols and timeframes"""
        await self.init_db()
        self.init_exchanges()
        
        total_candles = 0
        
        for exchange_id in EXCHANGES:
            if exchange_id not in self.exchanges:
                continue
                
            for symbol in SYMBOLS:
                # Check if symbol is supported by this exchange
                if symbol not in self.exchanges[exchange_id].markets:
                    logger.warning(f"Symbol {symbol} not supported by {exchange_id}")
                    continue
                
                for timeframe in TIMEFRAMES:
                    candles_fetched = await self.fetch_symbol_data(
                        exchange_id, symbol, timeframe, DAYS_TO_FETCH
                    )
                    total_candles += candles_fetched
        
        logger.info(f"🎉 Data fetching completed! Total candles fetched: {total_candles}")
        return total_candles

async def main():
    """Main function"""
    # Get database connection string
    db_conn_str = os.getenv('DATABASE_URL') or os.getenv('NEON_TEST_DB')
    if not db_conn_str:
        logger.error("No database connection string found in environment variables")
        sys.exit(1)
    
    logger.info("🚀 Starting reliable data fetching...")
    logger.info(f"Target exchanges: {EXCHANGES}")
    logger.info(f"Target symbols: {SYMBOLS}")
    logger.info(f"Target timeframes: {TIMEFRAMES}")
    logger.info(f"Days to fetch: {DAYS_TO_FETCH}")
    
    fetcher = ReliableDataFetcher(db_conn_str)
    
    try:
        total_candles = await fetcher.fetch_all_data()
        logger.info(f"✅ Successfully fetched {total_candles} candles")
    except Exception as e:
        logger.error(f"❌ Data fetching failed: {e}")
        sys.exit(1)
    finally:
        if fetcher.db_pool:
            await fetcher.db_pool.close()

if __name__ == "__main__":
    asyncio.run(main()) 