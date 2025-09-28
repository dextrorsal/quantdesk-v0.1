#!/usr/bin/env python3
"""
Small test fetch for last month of data.
Uses the existing CSV storage system with a subset of pairs.
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

# Test configuration - small subset for testing
TEST_EXCHANGES = ["bitget", "binance"]  # Start with most reliable
TEST_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT", 
    "SOL/USDT",
    "XRP/USDT",
    "PEPE/USDT"
]
TEST_TIMEFRAMES = ["1h", "4h", "1d"]  # Start with common timeframes
DAYS_TO_FETCH = 30  # Last month

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDataFetcher:
    """Simple test fetcher for last month of data."""
    
    def __init__(self, data_path: str = "data/historical/processed"):
        """Initialize the test data fetcher."""
        self.config = StorageConfig(data_path=data_path)
        self.storage = CSVStorage(self.config)
        self.exchanges = {}
        
        logger.info(f"🚀 Initialized test fetcher with storage at {data_path}")
    
    def init_exchanges(self):
        """Initialize exchange connections."""
        for exchange_id in TEST_EXCHANGES:
            try:
                exchange = getattr(ccxt, exchange_id)()
                exchange.load_markets()
                self.exchanges[exchange_id] = exchange
                logger.info(f"✅ Initialized {exchange_id} exchange")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
    
    async def fetch_symbol_data(self, exchange_id: str, symbol: str, timeframe: str) -> int:
        """Fetch last month of data for a specific symbol and timeframe."""
        if exchange_id not in self.exchanges:
            logger.warning(f"Exchange {exchange_id} not available")
            return 0
        
        exchange = self.exchanges[exchange_id]
        
        # Check if symbol is supported
        if symbol not in exchange.markets:
            logger.warning(f"Symbol {symbol} not supported by {exchange_id}")
            return 0
        
        try:
            # Calculate time range (last month)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=DAYS_TO_FETCH)
            
            # Convert to timestamp for exchange API
            since = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            all_candles = []
            
            print(f"  📈 Fetching {symbol} {timeframe} from {exchange_id}...")
            
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
                        logger.debug(f"No more data for {symbol} {timeframe}")
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
                    success = await self.storage.store_candles(
                        exchange_id, symbol, timeframe, candles
                    )
                    
                    if success:
                        all_candles.extend(candles)
                        print(f"    ✅ {timeframe}: {len(candles)} candles stored")
                    else:
                        print(f"    ❌ {timeframe}: Failed to store candles")
                    
                    # Update since timestamp for next batch
                    if len(ohlcv) > 0:
                        since = ohlcv[-1][0] + 1
                    else:
                        break
                    
                    # Rate limiting
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    logger.error(f"Error fetching {symbol} {timeframe}: {e}")
                    break
            
            logger.info(f"✅ Fetched {len(all_candles)} {timeframe} candles for {symbol} from {exchange_id}")
            return len(all_candles)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe} from {exchange_id}: {e}")
            return 0
    
    async def fetch_all_data(self) -> Dict[str, int]:
        """Fetch all test data."""
        self.init_exchanges()
        
        total_candles = 0
        successful_fetches = 0
        failed_fetches = 0
        
        print(f"\n📊 Starting test fetch for last {DAYS_TO_FETCH} days")
        print(f"🏢 Exchanges: {', '.join(TEST_EXCHANGES)}")
        print(f"📈 Pairs: {', '.join(TEST_SYMBOLS)}")
        print(f"⏰ Timeframes: {', '.join(TEST_TIMEFRAMES)}")
        print("-" * 60)
        
        for exchange_id in TEST_EXCHANGES:
            if exchange_id not in self.exchanges:
                continue
                
            print(f"\n🏢 Fetching from {exchange_id.upper()}...")
            
            for symbol in TEST_SYMBOLS:
                for timeframe in TEST_TIMEFRAMES:
                    try:
                        candles_fetched = await self.fetch_symbol_data(
                            exchange_id, symbol, timeframe
                        )
                        
                        if candles_fetched > 0:
                            total_candles += candles_fetched
                            successful_fetches += 1
                        else:
                            failed_fetches += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to fetch {symbol} {timeframe} from {exchange_id}: {e}")
                        failed_fetches += 1
        
        return {
            "total_candles": total_candles,
            "successful_fetches": successful_fetches,
            "failed_fetches": failed_fetches
        }


async def main():
    """Main function to run the test fetch."""
    fetcher = TestDataFetcher()
    
    try:
        summary = await fetcher.fetch_all_data()
        
        print("\n" + "=" * 60)
        print("📊 TEST FETCH SUMMARY")
        print("=" * 60)
        print(f"✅ Successful fetches: {summary['successful_fetches']}")
        print(f"❌ Failed fetches: {summary['failed_fetches']}")
        print(f"📈 Total candles stored: {summary['total_candles']:,}")
        
        # Show storage location
        data_dir = Path("data/historical/processed")
        if data_dir.exists():
            print(f"\n📁 Data stored in: {data_dir.absolute()}")
            
            # Show what was created
            for exchange_dir in data_dir.iterdir():
                if exchange_dir.is_dir():
                    print(f"  📂 {exchange_dir.name}/")
                    for pair_dir in exchange_dir.iterdir():
                        if pair_dir.is_dir():
                            print(f"    📂 {pair_dir.name}/")
                            for timeframe_dir in pair_dir.iterdir():
                                if timeframe_dir.is_dir():
                                    csv_files = list(timeframe_dir.glob("*.csv"))
                                    if csv_files:
                                        print(f"      📄 {timeframe_dir.name}: {len(csv_files)} files")
        
        print(f"\n🎉 Test completed! Check the data directory to see your fetched data.")
        
    except Exception as e:
        logger.error(f"Test fetch failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 