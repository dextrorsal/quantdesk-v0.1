#!/usr/bin/env python3
"""
CSV Data Fetcher - Fetches market data and stores it in CSV format

This script fetches candlestick data from various exchanges and stores it
in the new CSV storage system with the structure:
data/ohlcv/{exchange}/{pair}/{interval}/{YYYY}/{MM}/{YYYY-MM-DD}.csv

USE CASES:
- **Manual CSV data fetching**: One-time or on-demand data collection
- **CSV storage system**: Uses the organized CSV storage structure
- **Multiple exchanges**: Supports Binance, Bitget, and other exchanges
- **Flexible time ranges**: Configurable number of days to fetch
- **Data verification**: Built-in verification of fetched data
- **Progress tracking**: Visual progress bars for long-running fetches

DIFFERENCES FROM OTHER FETCHERS:
- CSVDataFetcher: Manual CSV storage, flexible time ranges, data verification
- DailyDataFetcher: Automated daily CSV storage, cron job integration
- UltimateDataFetcher: Multi-exchange orchestrator with database storage
- ReliableDataFetcher: CCXT-based, fast exchanges only, database storage
- DatabaseLoader: Simple database loading interface (like pd.read_csv)

WHEN TO USE:
- For one-time or manual data collection
- When you want CSV storage (not database)
- When you need flexible time ranges (not just daily)
- For data verification and quality checks
- When you don't need automation

FEATURES:
- Supports multiple exchanges (Binance, Bitget, etc.)
- Supports multiple symbols and timeframes
- Automatic folder creation
- Data deduplication and merging
- Progress tracking with tqdm
- Rate limiting to respect exchange limits
- Error handling and retry logic

Usage:
    python scripts/fetch_to_csv.py
"""

import sys
import time
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import ccxt
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.csv_storage import CSVStorage, StorageConfig


# Configuration
EXCHANGES = ["binance", "bitget"]  # Add more exchanges as needed
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DAYS_TO_FETCH = 7  # Start with a smaller number for testing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVDataFetcher:
    """Fetches market data and stores it in CSV format."""
    
    def __init__(self, data_path: str = "data/ohlcv"):
        """
        Initialize the CSV data fetcher.
        
        Args:
            data_path: Base path for storing CSV files
        """
        self.config = StorageConfig(data_path=data_path)
        self.storage = CSVStorage(self.config)
        self.exchanges = {}
        
        logger.info(f"Initialized CSV data fetcher with storage at {data_path}")
    
    def init_exchanges(self):
        """Initialize exchange connections."""
        for exchange_id in EXCHANGES:
            try:
                exchange = getattr(ccxt, exchange_id)()
                exchange.load_markets()
                self.exchanges[exchange_id] = exchange
                logger.info(f"✅ Initialized {exchange_id} exchange")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
    
    async def fetch_symbol_data(self, exchange_id: str, symbol: str, timeframe: str, days: int) -> int:
        """
        Fetch historical data for a specific symbol and timeframe.
        
        Args:
            exchange_id: Exchange name (e.g., "binance")
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Candle interval (e.g., "1h")
            days: Number of days to fetch
            
        Returns:
            int: Number of candles fetched
        """
        if exchange_id not in self.exchanges:
            logger.warning(f"Exchange {exchange_id} not available")
            return 0
        
        exchange = self.exchanges[exchange_id]
        
        # Check if symbol is supported
        if symbol not in exchange.markets:
            logger.warning(f"Symbol {symbol} not supported by {exchange_id}")
            return 0
        
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Convert to timestamp for exchange API
            since = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            all_candles = []
            
            with tqdm(desc=f"{exchange_id}:{symbol}:{timeframe}", unit="candles") as pbar:
                while since < end_ts:
                    try:
                        # Fetch batch of candles
                        ohlcv = exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=since,
                            limit=1000
                        )
                        
                        if not ohlcv:
                            logger.debug(f"No more data for {symbol} {timeframe} from {exchange_id}")
                            break
                        
                        # Convert to list of dictionaries
                        candles = []
                        for candle in ohlcv:
                            candles.append({
                                "timestamp": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[5])
                            })
                        
                        # Store in CSV
                        success = await self.storage.store_candles(exchange_id, symbol, timeframe, candles)
                        
                        if success:
                            all_candles.extend(candles)
                            pbar.update(len(candles))
                            logger.debug(f"Stored {len(candles)} candles for {symbol} {timeframe}")
                        else:
                            logger.error(f"Failed to store candles for {symbol} {timeframe}")
                        
                        # Update since timestamp for next batch
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
    
    async def fetch_all_data(self) -> Dict[str, int]:
        """
        Fetch all data for all symbols and timeframes.
        
        Returns:
            Dict with summary of fetched data
        """
        self.init_exchanges()
        
        if not self.exchanges:
            logger.error("No exchanges available")
            return {}
        
        summary = {}
        total_candles = 0
        
        for exchange_id in self.exchanges:
            summary[exchange_id] = {}
            
            for symbol in SYMBOLS:
                summary[exchange_id][symbol] = {}
                
                for timeframe in TIMEFRAMES:
                    logger.info(f"🔄 Fetching {symbol} {timeframe} from {exchange_id}...")
                    
                    candles_fetched = await self.fetch_symbol_data(
                        exchange_id, symbol, timeframe, DAYS_TO_FETCH
                    )
                    
                    summary[exchange_id][symbol][timeframe] = candles_fetched
                    total_candles += candles_fetched
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
        
        logger.info(f"🎉 Data fetching completed! Total candles fetched: {total_candles}")
        return summary
    
    async def verify_fetched_data(self) -> Dict[str, Any]:
        """
        Verify the integrity of fetched data.
        
        Returns:
            Dict with verification results
        """
        logger.info("🔍 Verifying fetched data...")
        
        verification_results = {}
        
        for exchange_id in self.exchanges:
            verification_results[exchange_id] = {}
            
            for symbol in SYMBOLS:
                verification_results[exchange_id][symbol] = {}
                
                for timeframe in TIMEFRAMES:
                    # Check last 7 days of data
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=7)
                    
                    try:
                        integrity = await self.storage.verify_data_integrity(
                            exchange_id, symbol, timeframe, start_time, end_time
                        )
                        verification_results[exchange_id][symbol][timeframe] = integrity
                        
                        if integrity['data_integrity_ok']:
                            logger.info(f"✅ {exchange_id}/{symbol}/{timeframe}: {integrity['total_candles']} candles")
                        else:
                            logger.warning(f"⚠️ {exchange_id}/{symbol}/{timeframe}: {integrity}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error verifying {exchange_id}/{symbol}/{timeframe}: {e}")
                        verification_results[exchange_id][symbol][timeframe] = {"error": str(e)}
        
        return verification_results
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a summary of fetched data."""
        print("\n" + "="*60)
        print("📊 FETCH SUMMARY")
        print("="*60)
        
        total_candles = 0
        
        for exchange_id, exchange_data in summary.items():
            print(f"\n🏢 {exchange_id.upper()}:")
            
            for symbol, symbol_data in exchange_data.items():
                print(f"  📈 {symbol}:")
                
                for timeframe, count in symbol_data.items():
                    print(f"    {timeframe}: {count:,} candles")
                    total_candles += count
        
        print(f"\n🎯 TOTAL: {total_candles:,} candles fetched")
        print("="*60)


async def main():
    """Main function to run the CSV data fetcher."""
    logger.info("🚀 Starting CSV data fetching...")
    logger.info(f"Target exchanges: {EXCHANGES}")
    logger.info(f"Target symbols: {SYMBOLS}")
    logger.info(f"Target timeframes: {TIMEFRAMES}")
    logger.info(f"Days to fetch: {DAYS_TO_FETCH}")
    
    fetcher = CSVDataFetcher()
    
    try:
        # Fetch all data
        summary = await fetcher.fetch_all_data()
        
        # Print summary
        fetcher.print_summary(summary)
        
        # Verify data integrity
        verification = await fetcher.verify_fetched_data()
        
        logger.info("✅ CSV data fetching completed successfully!")
        
        # Show available data
        available = fetcher.storage.list_available_data()
        logger.info(f"📁 Total data files created: {len(available)}")
        
    except Exception as e:
        logger.error(f"❌ CSV data fetching failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 