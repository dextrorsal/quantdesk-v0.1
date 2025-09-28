#!/usr/bin/env python3
"""
Daily Automated Data Fetcher

This script automatically fetches candlestick data daily for the user's preferred
exchanges and pairs, storing everything in CSV format.

USE CASES:
- **Automated daily data collection**: Runs daily to collect fresh market data
- **CSV storage**: Stores data in organized CSV files by date/exchange/pair
- **Multiple exchanges**: Supports 6+ exchanges (Bitget, Binance, Coinbase, etc.)
- **Wide pair coverage**: Fetches 40+ trading pairs including memecoins
- **Cron job integration**: Designed to run automatically via cron/scheduler
- **Incremental updates**: Only fetches the last day of data each run
- **Error recovery**: Built-in retry logic and error handling

DIFFERENCES FROM OTHER FETCHERS:
- DailyDataFetcher: Automated daily CSV storage, wide exchange/pair coverage
- UltimateDataFetcher: Multi-exchange orchestrator with database storage
- ReliableDataFetcher: CCXT-based, fast exchanges only, database storage
- CSVDataFetcher: Manual CSV storage, no automation
- DatabaseLoader: Simple database loading interface (like pd.read_csv)

WHEN TO USE:
- For automated daily data collection
- When you want CSV storage (not database)
- When you need data from many exchanges and pairs
- For cron job automation
- When you want incremental daily updates

FEATURES:
- Automated daily fetching
- Support for multiple exchanges and pairs
- Configurable timeframes
- Error handling and retry logic
- Progress tracking
- Logging to file
- Rate limiting to respect exchange limits

Usage:
    python scripts/daily_fetch.py
    # Or set up as a cron job for daily execution
"""

import sys
import time
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import ccxt
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.csv_storage import CSVStorage, StorageConfig


# Configuration - User's preferred exchanges and pairs
EXCHANGES = ["bitget", "binance", "coinbase", "mexc", "kucoin", "kraken"]

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "SPX/USDT",
    "FARTCOIN/USDT", "HYPE/USDT", "WIF/USDT", "POPCAT/USDT", "TRUMP/USDT",
    "PENGU/USDT", "PNUT/USDT", "MOODENG/USDT", "GIGA/USDT", "MEW/USDT",
    "GOAT/USDT", "AI16z/USDT", "PONKE/USDT", "BOME/USDT", "FWOG/USDT",
    "PEPE/USDT", "PEPECOIN/USDT", "DODGE/USDT", "FLOKI/USDT", "BRETT/USDT",
    "SHIB/USDT", "BGB/USDT", "CAKE/USDT", "ORCA/USDT", "GRASS/USDT",
    "UNI/USDT", "JUP/USDT", "DRIFT/USDT", "CHILLGUY/USDT", "PI/USDT",
    "BIGTIME/USDT", "FET/USDT", "AAVE/USDT", "AVAX/USDT", "LINK/USDT",
    "SUI/USDT", "ADA/USDT", "IMX/USDT", "BNB/USDT"
]

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Fetching configuration
DAYS_TO_FETCH = 1  # Fetch last day of data
RATE_LIMIT_DELAY = 0.1  # Delay between requests (seconds)
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds to wait between retries


class DailyDataFetcher:
    """
    Automated daily data fetcher for multiple exchanges and pairs.
    """
    
    def __init__(self, data_path: str = "data/ohlcv"):
        """
        Initialize the daily data fetcher.
        
        Args:
            data_path: Path to store CSV data
        """
        self.data_path = Path(data_path)
        self.storage = CSVStorage(StorageConfig(data_path=data_path))
        
        # Setup logging
        self._setup_logging()
        
        # Initialize exchanges
        self.exchanges = self._initialize_exchanges()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"daily_fetch_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_exchanges(self) -> Dict[str, Any]:
        """
        Initialize exchange connections.
        
        Returns:
            Dictionary of exchange instances
        """
        exchanges = {}
        
        for exchange_name in EXCHANGES:
            try:
                # Create exchange instance
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'rateLimit': int(RATE_LIMIT_DELAY * 1000)
                })
                
                # Load markets
                exchange.load_markets()
                exchanges[exchange_name] = exchange
                
                self.logger.info(f"✅ Initialized {exchange_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
                continue
        
        return exchanges
    
    def _get_available_pairs(self, exchange_name: str) -> List[str]:
        """
        Get available pairs for an exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            List of available pairs
        """
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return []
        
        available_pairs = []
        for pair in PAIRS:
            if pair in exchange.markets:
                available_pairs.append(pair)
            else:
                # Try alternative formats
                alt_pair = pair.replace("/", "")
                if alt_pair in exchange.markets:
                    available_pairs.append(pair)
        
        return available_pairs
    
    async def fetch_data_for_pair(
        self,
        exchange_name: str,
        pair: str,
        timeframe: str,
        days: int = 1
    ) -> bool:
        """
        Fetch data for a specific exchange, pair, and timeframe.
        
        Args:
            exchange_name: Name of the exchange
            pair: Trading pair
            timeframe: Timeframe
            days: Number of days to fetch
            
        Returns:
            True if successful, False otherwise
        """
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return False
        
        # Check if pair is available
        if pair not in exchange.markets:
            self.logger.warning(f"Pair {pair} not available on {exchange_name}")
            return False
        
        # Check if timeframe is supported
        if timeframe not in exchange.timeframes:
            self.logger.warning(
                f"Timeframe {timeframe} not supported on {exchange_name}"
            )
            return False
        
        for attempt in range(MAX_RETRIES):
            try:
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
                
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    since=int(start_time.timestamp() * 1000),
                    limit=1000  # Adjust based on exchange limits
                )
                
                if not ohlcv:
                    self.logger.warning(
                        f"No data returned for {exchange_name}/{pair}/{timeframe}"
                    )
                    return False
                
                # Convert to DataFrame format
                candles = []
                for candle in ohlcv:
                    candles.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                
                # Store data
                await self.storage.store_candles(
                    exchange=exchange_name,
                    pair=pair,
                    interval=timeframe,
                    candles=candles
                )
                
                self.logger.info(
                    f"✅ Fetched {len(candles)} candles for "
                    f"{exchange_name}/{pair}/{timeframe}"
                )
                
                return True
                
            except Exception as e:
                self.logger.error(
                    f"Attempt {attempt + 1} failed for {exchange_name}/{pair}/"
                    f"{timeframe}: {e}"
                )
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    self.logger.error(
                        f"Failed to fetch {exchange_name}/{pair}/{timeframe} "
                        f"after {MAX_RETRIES} attempts"
                    )
                    return False
    
    async def run_daily_fetch(self):
        """
        Run the daily data fetching process.
        """
        self.logger.info("🚀 Starting daily data fetch")
        self.logger.info(f"📊 Exchanges: {', '.join(EXCHANGES)}")
        self.logger.info(f"🪙 Pairs: {len(PAIRS)} pairs")
        self.logger.info(f"⏰ Timeframes: {', '.join(TIMEFRAMES)}")
        
        total_tasks = 0
        successful_tasks = 0
        
        # Create all tasks
        tasks = []
        for exchange_name in EXCHANGES:
            if exchange_name not in self.exchanges:
                continue
                
            available_pairs = self._get_available_pairs(exchange_name)
            self.logger.info(
                f"📈 {exchange_name}: {len(available_pairs)} available pairs"
            )
            
            for pair in available_pairs:
                for timeframe in TIMEFRAMES:
                    task = self.fetch_data_for_pair(
                        exchange_name, pair, timeframe, DAYS_TO_FETCH
                    )
                    tasks.append(task)
                    total_tasks += 1
        
        # Execute tasks with progress bar
        self.logger.info(f"🔄 Executing {total_tasks} fetch tasks...")
        
        for i, task in enumerate(tqdm(tasks, desc="Fetching data")):
            success = await task
            if success:
                successful_tasks += 1
            
            # Small delay to respect rate limits
            await asyncio.sleep(RATE_LIMIT_DELAY)
        
        # Summary
        self.logger.info("🎉 Daily fetch completed!")
        self.logger.info(f"✅ Successful: {successful_tasks}/{total_tasks}")
        self.logger.info(f"❌ Failed: {total_tasks - successful_tasks}/{total_tasks}")
        
        if successful_tasks > 0:
            self.logger.info("📁 Data saved to CSV files in data/ohlcv/")
        
        return successful_tasks, total_tasks


async def main():
    """Main function to run the daily fetch."""
    fetcher = DailyDataFetcher()
    
    try:
        successful, total = await fetcher.run_daily_fetch()
        
        if successful > 0:
            print(f"\n🎉 Successfully fetched data for {successful}/{total} tasks!")
        else:
            print("\n❌ No data was fetched successfully.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Fetch interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 