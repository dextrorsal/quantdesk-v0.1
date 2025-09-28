#!/usr/bin/env python3
"""
Fetch data for all pairs from the user's list.
Uses the CSV storage system to fetch the last month of data.
"""

import sys
import time
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import ccxt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.csv_storage import CSVStorage, StorageConfig

# All pairs from user's list
ALL_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "SPX/USDT", 
    "FARTCOIN/USDT", "HYPE/USDT", "WIF/USDT", "POPCAT/USDT", 
    "TRUMP/USDT", "PENGU/USDT", "PNUT/USDT", "MOODENG/USDT", 
    "GIGA/USDT", "MEW/USDT", "GOAT/USDT", "AI16z/USDT", 
    "PONKE/USDT", "BOME/USDT", "FWOG/USDT", "PEPE/USDT", 
    "PEPECOIN/USDT", "DODGE/USDT", "FLOKI/USDT", "BRETT/USDT", 
    "SHIB/USDT", "BGB/USDT", "CAKE/USDT", "ORCA/USDT", 
    "GRASS/USDT", "UNI/USDT", "JUP/USDT", "DRIFT/USDT", 
    "CHILLGUY/USDT", "PI/USDT", "BIGTIME/USDT", "FET/USDT", 
    "AAVE/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT", 
    "ADA/USDT", "IMX/USDT", "BNB/USDT"
]

# Exchanges to fetch from
EXCHANGES = ["bitget", "binance", "coinbase", "mexc", "kucoin", "kraken"]

# Timeframes to fetch - complete range from 1m to 1d
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Days to fetch - scaled for 1 year with appropriate limits
# Balancing data completeness with storage efficiency
DAYS_TO_FETCH = {
    "1m": 30,   # 30 days of 1m data = 43,200 candles per pair
    "5m": 90,   # 90 days of 5m data = 25,920 candles per pair  
    "15m": 180, # 180 days of 15m data = 17,280 candles per pair
    "1h": 365,  # 1 year of 1h data = 8,760 candles per pair
    "4h": 365,  # 1 year of 4h data = 2,190 candles per pair
    "1d": 365   # 1 year of 1d data = 365 candles per pair
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AllPairsFetcher:
    """Fetches data for all pairs from the user's list."""
    
    def __init__(self, data_path: str = "data/historical/processed"):
        """Initialize the all pairs fetcher."""
        self.config = StorageConfig(data_path=data_path)
        self.storage = CSVStorage(self.config)
        self.exchanges = {}
        
        logger.info(f"🚀 Initialized all pairs fetcher with storage at {data_path}")
    
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
    
    def get_available_pairs(self, exchange_id: str) -> List[str]:
        """Get list of pairs that are available on the exchange."""
        if exchange_id not in self.exchanges:
            return []
        
        exchange = self.exchanges[exchange_id]
        available_pairs = []
        
        for pair in ALL_PAIRS:
            if pair in exchange.markets:
                available_pairs.append(pair)
        
        return available_pairs
    
    async def fetch_symbol_data(self, exchange_id: str, symbol: str, timeframe: str) -> int:
        """Fetch last month of data for a specific symbol and timeframe."""
        if exchange_id not in self.exchanges:
            return 0
        
        exchange = self.exchanges[exchange_id]
        
        try:
            # Calculate time range based on timeframe
            end_time = datetime.now()
            days_to_fetch = DAYS_TO_FETCH.get(timeframe, 30)
            start_time = end_time - timedelta(days=days_to_fetch)
            
            # Convert to timestamp for exchange API
            since = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            all_candles = []
            
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
            
            return len(all_candles)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe} from {exchange_id}: {e}")
            return 0
    
    async def fetch_all_data(self) -> Dict[str, int]:
        """Fetch all data for all available pairs."""
        self.init_exchanges()
        
        total_candles = 0
        successful_fetches = 0
        failed_fetches = 0
        
        print(f"\n📊 Starting fetch for all pairs with timeframe-specific ranges:")
        for tf, days in DAYS_TO_FETCH.items():
            print(f"   {tf}: {days} days")
        print(f"🏢 Exchanges: {', '.join(EXCHANGES)}")
        print(f"📈 Total pairs to check: {len(ALL_PAIRS)}")
        print(f"⏰ Timeframes: {', '.join(TIMEFRAMES)}")
        print("-" * 80)
        
        for exchange_id in EXCHANGES:
            if exchange_id not in self.exchanges:
                continue
            
            # Get available pairs for this exchange
            available_pairs = self.get_available_pairs(exchange_id)
            
            print(f"\n🏢 {exchange_id.upper()}: {len(available_pairs)} pairs available")
            
            for symbol in available_pairs:
                print(f"  📈 {symbol}...")
                
                for timeframe in TIMEFRAMES:
                    try:
                        candles_fetched = await self.fetch_symbol_data(
                            exchange_id, symbol, timeframe
                        )
                        
                        if candles_fetched > 0:
                            total_candles += candles_fetched
                            successful_fetches += 1
                            print(f"    ✅ {timeframe}: {candles_fetched} candles")
                        else:
                            failed_fetches += 1
                            print(f"    ⚠️  {timeframe}: No data")
                            
                    except Exception as e:
                        logger.error(f"Failed to fetch {symbol} {timeframe} from {exchange_id}: {e}")
                        failed_fetches += 1
                        print(f"    ❌ {timeframe}: Error")
        
        return {
            "total_candles": total_candles,
            "successful_fetches": successful_fetches,
            "failed_fetches": failed_fetches
        }


async def main():
    """Main function to run the all pairs fetch."""
    fetcher = AllPairsFetcher()
    
    try:
        summary = await fetcher.fetch_all_data()
        
        print("\n" + "=" * 80)
        print("📊 ALL PAIRS FETCH SUMMARY")
        print("=" * 80)
        print(f"✅ Successful fetches: {summary['successful_fetches']}")
        print(f"❌ Failed fetches: {summary['failed_fetches']}")
        print(f"📈 Total candles stored: {summary['total_candles']:,}")
        
        # Show storage location
        data_dir = Path("data/historical/processed")
        if data_dir.exists():
            print(f"\n📁 Data stored in: {data_dir.absolute()}")
            
            # Count total CSV files
            csv_files = list(data_dir.rglob("*.csv"))
            print(f"📄 Total CSV files created: {len(csv_files)}")
            
            # Show disk usage
            total_size = sum(f.stat().st_size for f in csv_files)
            print(f"💾 Total disk usage: {total_size / (1024*1024):.2f} MB")
        
        print(f"\n🎉 All pairs fetch completed!")
        
    except Exception as e:
        logger.error(f"All pairs fetch failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 