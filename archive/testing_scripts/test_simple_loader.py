#!/usr/bin/env python3
"""
Simple test for data loader with actual data structure.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader


async def test_simple_loader():
    """Test the data loader with actual data."""
    
    print("🧪 Simple Data Loader Test")
    print("=" * 40)
    
    # Initialize loader
    loader = DataLoader("data/ohlcv")
    
    # Get available data
    available = loader.get_available_data()
    
    if not available:
        print("❌ No data found")
        return
    
    # Find a dataset with actual data
    for exchange, pairs in available.items():
        for pair, intervals in pairs.items():
            for interval, info in intervals.items():
                if info['total_candles'] > 0:
                    print(f"📊 Testing: {exchange}/{pair}/{interval}")
                    print(f"   📅 Date range: {info['start_date']} to {info['end_date']}")
                    print(f"   📁 Files: {info['file_count']}")
                    print(f"   🕯️  Candles: {info['total_candles']}")
                    
                    # Load data for this combination
                    df = await loader.load_data(
                        exchange=exchange,
                        pair=pair,
                        interval=interval,
                        start_date=info['start_date'],
                        end_date=info['end_date']
                    )
                    
                    if not df.empty:
                        print(f"✅ Successfully loaded {len(df)} candles")
                        print(f"   📅 Range: {df.index.min()} to {df.index.max()}")
                        print(f"   💰 Price: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                        print(f"   📊 Columns: {list(df.columns)}")
                        
                        # Show first few rows
                        print("\n📋 First 3 candles:")
                        print(df.head(3).to_string())
                        
                        return  # Success, exit
                    else:
                        print("❌ Failed to load data")
    
    print("❌ No valid data found to test")


if __name__ == "__main__":
    asyncio.run(test_simple_loader()) 