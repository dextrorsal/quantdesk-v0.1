#!/usr/bin/env python3
"""
Test script for Data Loader functionality.

This script demonstrates how to use the new data loader system to load
candlestick data from CSV files with various filtering options.

Usage:
    python scripts/test_data_loader.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader, load_data, get_available_data


async def test_data_loader():
    """Test the data loader functionality."""
    
    print("🧪 Testing Data Loader System")
    print("=" * 50)
    
    # Initialize loader
    loader = DataLoader("data/ohlcv")
    
    # 1. Get available data summary
    print("\n📊 Available Data Summary:")
    print("-" * 30)
    
    available = get_available_data()
    if not available:
        print("❌ No data found. Please run fetch_to_csv.py first.")
        return
    
    for exchange, pairs in available.items():
        print(f"\n🏢 {exchange.upper()}:")
        for pair, intervals in pairs.items():
            print(f"  📈 {pair}:")
            for interval, info in intervals.items():
                print(f"    ⏰ {interval}: {info['file_count']} files, "
                      f"{info['total_candles']} candles")
                print(f"      📅 {info['start_date']} to {info['end_date']}")
    
    # 2. Get data summary as DataFrame
    print("\n📋 Data Summary DataFrame:")
    print("-" * 30)
    
    summary_df = loader.get_data_summary()
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("❌ No data summary available")
    
    # 3. Load specific data
    print("\n📈 Loading Specific Data:")
    print("-" * 30)
    
    # Try to load data for a specific exchange/pair/interval
    exchanges = list(available.keys())
    if exchanges:
        exchange = exchanges[0]
        pairs = list(available[exchange].keys())
        if pairs:
            pair = pairs[0]
            intervals = list(available[exchange][pair].keys())
            if intervals:
                interval = intervals[0]
                
                print(f"Loading {exchange}/{pair}/{interval}...")
                
                df = await loader.load_data(
                    exchange=exchange,
                    pair=pair,
                    interval=interval,
                    start_date="2025-07-15",
                    end_date="2025-07-16"
                )
                
                if not df.empty:
                    print(f"✅ Loaded {len(df)} candles")
                    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
                    print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                    print(f"📊 Columns: {list(df.columns)}")
                    
                    # Show first few rows
                    print("\n📋 First 5 candles:")
                    print(df.head().to_string())
                else:
                    print("❌ No data loaded")
    
    # 4. Load recent data
    print("\n🕒 Loading Recent Data:")
    print("-" * 30)
    
    if exchanges and pairs:
        exchange = exchanges[0]
        pair = pairs[0]
        intervals = list(available[exchange][pair].keys())
        if intervals:
            interval = intervals[0]
            
            print(f"Loading recent 2 days of {exchange}/{pair}/{interval}...")
            
            recent_df = await loader.load_recent_data(
                exchange=exchange,
                pair=pair,
                interval=interval,
                days=2
            )
            
            if not recent_df.empty:
                print(f"✅ Loaded {len(recent_df)} recent candles")
                print(f"📅 Date range: {recent_df.index.min()} to {recent_df.index.max()}")
            else:
                print("❌ No recent data loaded")
    
    # 5. Load multiple pairs
    print("\n🔄 Loading Multiple Pairs:")
    print("-" * 30)
    
    if len(exchanges) >= 2 and len(pairs) >= 2:
        test_exchanges = exchanges[:2]
        test_pairs = pairs[:2]
        
        print(f"Loading data for {len(test_exchanges)} exchanges and {len(test_pairs)} pairs...")
        
        multi_data = await loader.load_multiple_pairs(
            exchanges=test_exchanges,
            pairs=test_pairs,
            interval="1h",
            start_date="2025-07-15",
            end_date="2025-07-16",
            show_progress=True
        )
        
        print(f"✅ Loaded data for {len(multi_data)} combinations:")
        for key, df in multi_data.items():
            print(f"  📊 {key}: {len(df)} candles")
    
    # 6. Test convenience functions
    print("\n⚡ Testing Convenience Functions:")
    print("-" * 30)
    
    if exchanges and pairs:
        exchange = exchanges[0]
        pair = pairs[0]
        intervals = list(available[exchange][pair].keys())
        if intervals:
            interval = intervals[0]
            
            print(f"Using convenience function for {exchange}/{pair}/{interval}...")
            
            # Test convenience function
            df = await load_data(
                exchange=exchange,
                pair=pair,
                interval=interval,
                start_date="2025-07-15",
                end_date="2025-07-16"
            )
            
            if not df.empty:
                print(f"✅ Convenience function loaded {len(df)} candles")
            else:
                print("❌ Convenience function failed")
    
    print("\n🎉 Data Loader Test Completed!")
    print("=" * 50)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_data_loader()) 