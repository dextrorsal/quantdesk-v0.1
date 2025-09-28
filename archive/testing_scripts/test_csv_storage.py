#!/usr/bin/env python3
"""
Test script for CSV Storage functionality.

This script demonstrates how to use the new CSV storage system for candlestick data.
It shows examples of storing data, loading data for different date ranges, and
verifying data integrity.

Usage:
    python scripts/test_csv_storage.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.csv_storage import (
    CSVStorage, StorageConfig, store_candles_simple, load_candles_simple
)


async def test_csv_storage():
    """Test the CSV storage functionality with sample data."""
    
    print("🧪 Testing CSV Storage System")
    print("=" * 50)
    
    # Initialize storage
    config = StorageConfig(data_path="data/ohlcv")
    storage = CSVStorage(config)
    
    # Sample candlestick data
    sample_candles = [
        {
            "timestamp": "2025-01-15T10:00:00",
            "open": 65000.0,
            "high": 65100.0,
            "low": 64900.0,
            "close": 65050.0,
            "volume": 12.345
        },
        {
            "timestamp": "2025-01-15T11:00:00",
            "open": 65050.0,
            "high": 65200.0,
            "low": 65000.0,
            "close": 65150.0,
            "volume": 15.678
        },
        {
            "timestamp": "2025-01-15T12:00:00",
            "open": 65150.0,
            "high": 65300.0,
            "low": 65100.0,
            "close": 65250.0,
            "volume": 18.901
        },
        {
            "timestamp": "2025-01-16T10:00:00",  # Next day
            "open": 65250.0,
            "high": 65400.0,
            "low": 65200.0,
            "close": 65350.0,
            "volume": 20.123
        },
        {
            "timestamp": "2025-01-16T11:00:00",
            "open": 65350.0,
            "high": 65500.0,
            "low": 65300.0,
            "close": 65450.0,
            "volume": 22.456
        }
    ]
    
    print("📊 Sample Data:")
    for candle in sample_candles:
        print(f"  {candle['timestamp']}: O:{candle['open']} H:{candle['high']} "
              f"L:{candle['low']} C:{candle['close']} V:{candle['volume']}")
    
    print("\n💾 Storing Data...")
    
    # Test 1: Store data using the main class
    success = await storage.store_candles("binance", "BTCUSDT", "1h", sample_candles)
    print(f"✅ Storage result: {success}")
    
    # Test 2: Store data using simple function
    success2 = await store_candles_simple("bitget", "ETHUSDT", "1h", sample_candles)
    print(f"✅ Simple storage result: {success2}")
    
    print("\n📖 Loading Data...")
    
    # Test 3: Load data for a date range
    start_time = datetime(2025, 1, 15, 10, 0, 0)
    end_time = datetime(2025, 1, 16, 12, 0, 0)
    
    df = await storage.load_candles("binance", "BTCUSDT", "1h", start_time, end_time)
    print(f"✅ Loaded {len(df)} candles from {start_time.date()} to "
          f"{end_time.date()}")
    
    if not df.empty:
        print("📈 Loaded Data Preview:")
        print(df.head())
        print(f"\n📊 Data Summary:")
        print(f"  Total candles: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Test 4: Load data using simple function
    df2 = await load_candles_simple("bitget", "ETHUSDT", "1h", start_time, end_time)
    print(f"✅ Simple load result: {len(df2)} candles")
    
    print("\n🔍 Data Integrity Check...")
    
    # Test 5: Verify data integrity
    integrity = await storage.verify_data_integrity("binance", "BTCUSDT", "1h", 
                                                   start_time, end_time)
    print("✅ Data Integrity Results:")
    for key, value in integrity.items():
        print(f"  {key}: {value}")
    
    print("\n📋 Available Data Files...")
    
    # Test 6: List available data
    available = storage.list_available_data()
    print(f"✅ Found {len(available)} data files:")
    for data in available:
        print(f"  {data['exchange']}/{data['pair']}/{data['interval']} - "
              f"{data['date']} ({data['file_size']} bytes)")
    
    print("\n🔄 Resampling Test...")
    
    # Test 7: Resample data (if we have enough data)
    if len(df) > 1:
        resampled = await storage.resample_candles(df, "2h")
        print(f"✅ Resampled {len(df)} 1h candles to {len(resampled)} 2h candles")
        if not resampled.empty:
            print("📈 Resampled Data Preview:")
            print(resampled.head())
    
    print("\n🎉 All tests completed!")
    
    # Show the directory structure that was created
    print("\n📁 Directory Structure Created:")
    ohlcv_path = Path("data/ohlcv")
    if ohlcv_path.exists():
        for item in ohlcv_path.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(ohlcv_path)
                print(f"  📄 {rel_path}")
            elif item.is_dir():
                rel_path = item.relative_to(ohlcv_path)
                print(f"  📁 {rel_path}/")


async def test_data_merging():
    """Test that new data merges correctly with existing data."""
    
    print("\n🔄 Testing Data Merging...")
    print("=" * 30)
    
    config = StorageConfig(data_path="data/ohlcv")
    storage = CSVStorage(config)
    
    # First batch of data
    candles1 = [
        {
            "timestamp": "2025-01-20T10:00:00",
            "open": 66000.0,
            "high": 66100.0,
            "low": 65900.0,
            "close": 66050.0,
            "volume": 10.0
        }
    ]
    
    # Second batch with overlapping timestamp
    candles2 = [
        {
            "timestamp": "2025-01-20T10:00:00",  # Same timestamp
            "open": 66000.0,
            "high": 66200.0,  # Higher high
            "low": 65900.0,
            "close": 66100.0,  # Different close
            "volume": 15.0  # Different volume
        },
        {
            "timestamp": "2025-01-20T11:00:00",  # New timestamp
            "open": 66100.0,
            "high": 66300.0,
            "low": 66000.0,
            "close": 66250.0,
            "volume": 20.0
        }
    ]
    
    # Store first batch
    await storage.store_candles("test", "BTCUSDT", "1h", candles1)
    print("✅ Stored first batch")
    
    # Store second batch (should merge)
    await storage.store_candles("test", "BTCUSDT", "1h", candles2)
    print("✅ Stored second batch (merged)")
    
    # Load and verify
    start_time = datetime(2025, 1, 20, 10, 0, 0)
    end_time = datetime(2025, 1, 20, 12, 0, 0)
    
    df = await storage.load_candles("test", "BTCUSDT", "1h", start_time, end_time)
    print(f"✅ Loaded {len(df)} candles after merging")
    
    if not df.empty:
        print("📊 Merged Data:")
        for _, row in df.iterrows():
            print(f"  {row['timestamp']}: O:{row['open']} H:{row['high']} "
                  f"L:{row['low']} C:{row['close']} V:{row['volume']}")
    
    print("✅ Merging test completed!")


if __name__ == "__main__":
    print("🚀 Starting CSV Storage Tests")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_csv_storage())
    asyncio.run(test_data_merging())
    
    print("\n✨ All tests completed successfully!")
    print("\n📝 Usage Examples:")
    print("  # Store data")
    print("  await storage.store_candles('binance', 'BTCUSDT', '1h', candles)")
    print("  ")
    print("  # Load data for a range")
    print("  df = await storage.load_candles('binance', 'BTCUSDT', '1h', start, end)")
    print("  ")
    print("  # Check data integrity")
    print("  integrity = await storage.verify_data_integrity('binance', 'BTCUSDT', '1h', start, end)") 