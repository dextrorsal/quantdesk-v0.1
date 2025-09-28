#!/usr/bin/env python3
"""
Test script to fetch the last month of data for a subset of pairs.
This will test our CSV storage system with a small dataset first.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.csv_storage import CSVStorage
from exchanges.bitget.bitget import BitgetHandler
from exchanges.binance.binance import BinanceHandler
from exchanges.coinbase.coinbase import CoinbaseHandler


async def test_fetch_last_month():
    """Fetch last month of data for a few test pairs."""
    
    # Test with a small subset of pairs
    test_pairs = [
        "BTC/USDT",
        "ETH/USDT", 
        "SOL/USDT",
        "XRP/USDT",
        "PEPE/USDT"
    ]
    
    # Test exchanges (start with the most reliable ones)
    test_exchanges = [
        ("bitget", BitgetHandler),
        ("binance", BinanceHandler),
        ("coinbase", CoinbaseHandler)
    ]
    
    # Timeframes to test
    timeframes = ["1h", "4h", "1d"]
    
    # Calculate date range (last month)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"🚀 Starting test fetch for last month "
          f"({start_date.date()} to {end_date.date()})")
    print(f"📊 Pairs: {', '.join(test_pairs)}")
    print(f"🏢 Exchanges: {', '.join([ex[0] for ex in test_exchanges])}")
    print(f"⏰ Timeframes: {', '.join(timeframes)}")
    print("-" * 60)
    
    # Initialize CSV storage
    storage = CSVStorage()
    
    total_candles = 0
    successful_fetches = 0
    failed_fetches = 0
    
    for exchange_name, exchange_handler in test_exchanges:
        print(f"\n🏢 Fetching from {exchange_name.upper()}...")
        
        try:
            # Initialize exchange handler
            exchange = exchange_handler()
            
            for pair in test_pairs:
                print(f"  📈 {pair}...")
                
                for timeframe in timeframes:
                    try:
                        # Fetch data
                        since_timestamp = int(start_date.timestamp() * 1000)
                        candles = await exchange.fetch_ohlcv(
                            symbol=pair,
                            timeframe=timeframe,
                            since=since_timestamp,
                            limit=1000  # Get as much as possible
                        )
                        
                        if candles:
                            # Store in CSV
                            await storage.store_candles(
                                exchange=exchange_name,
                                pair=pair,
                                timeframe=timeframe,
                                candles=candles
                            )
                            
                            print(f"    ✅ {timeframe}: {len(candles)} candles stored")
                            total_candles += len(candles)
                            successful_fetches += 1
                        else:
                            print(f"    ⚠️  {timeframe}: No data available")
                            
                    except Exception as e:
                        print(f"    ❌ {timeframe}: Error - {str(e)}")
                        failed_fetches += 1
                        
        except Exception as e:
            print(f"  ❌ Failed to initialize {exchange_name}: {str(e)}")
            failed_fetches += 1
    
    print("\n" + "=" * 60)
    print("📊 FETCH SUMMARY")
    print("=" * 60)
    print(f"✅ Successful fetches: {successful_fetches}")
    print(f"❌ Failed fetches: {failed_fetches}")
    print(f"📈 Total candles stored: {total_candles:,}")
    
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

if __name__ == "__main__":
    asyncio.run(test_fetch_last_month()) 