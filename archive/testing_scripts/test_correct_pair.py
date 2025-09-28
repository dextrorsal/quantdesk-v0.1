#!/usr/bin/env python3
"""
Test data loader with correct pair format.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader


async def test_correct_pair():
    """Test with correct pair format."""
    
    print("🧪 Testing with Correct Pair Format")
    print("=" * 40)
    
    # Initialize loader
    loader = DataLoader("data/ohlcv")
    
    # Test with the correct pair format that matches the file structure
    print("📊 Testing: binance/BTC/USDT/1m")
    
    df = await loader.load_data(
        exchange="binance",
        pair="BTC/USDT",  # This should match the directory structure
        interval="1m",
        start_date="2025-07-15",
        end_date="2025-07-15"
    )
    
    if not df.empty:
        print(f"✅ Successfully loaded {len(df)} candles")
        print(f"   📅 Range: {df.index.min()} to {df.index.max()}")
        print(f"   💰 Price: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"   📊 Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\n📋 First 3 candles:")
        print(df.head(3).to_string())
    else:
        print("❌ Failed to load data")
    
    # Test with BTCUSDT format (the old format)
    print("\n📊 Testing: binance/BTCUSDT/1h")
    
    df2 = await loader.load_data(
        exchange="binance",
        pair="BTCUSDT",  # This should work with the old format
        interval="1h",
        start_date="2025-01-15",
        end_date="2025-01-16"
    )
    
    if not df2.empty:
        print(f"✅ Successfully loaded {len(df2)} candles")
        print(f"   📅 Range: {df2.index.min()} to {df2.index.max()}")
        print(f"   💰 Price: ${df2['low'].min():.2f} - ${df2['high'].max():.2f}")
    else:
        print("❌ Failed to load data")


if __name__ == "__main__":
    asyncio.run(test_correct_pair()) 