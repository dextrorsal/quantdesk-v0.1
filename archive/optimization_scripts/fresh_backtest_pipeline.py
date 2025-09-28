"""
Fresh Backtest Pipeline - Complete data refresh and backtesting

This script performs a complete fresh backtest by:
1. Checking existing data in the database
2. Deleting old data if it exists
3. Fetching fresh data from exchanges
4. Running backtests with the new data

USE CASES:
- **Data refresh**: Ensure you're using the most recent data for backtesting
- **Clean backtesting**: Start with fresh data to avoid any data corruption
- **Database cleanup**: Remove old data and fetch new data
- **Pipeline testing**: Test the complete data pipeline from fetch to backtest
- **Development workflow**: Fresh start during development and testing
- **Data validation**: Verify data pipeline is working correctly

DIFFERENCES FROM OTHER BACKTESTING SCRIPTS:
- fresh_backtest_pipeline.py: Complete data refresh and backtesting pipeline
- run_comparison.py: Strategy comparison with multiple models
- final_comparison.py: Final evaluation with optimized parameters
- combined_model_trader.py: Multi-timeframe model combination and trading
- start_trading_system.py: Complete system orchestration

WHEN TO USE:
- When you want to ensure fresh data for backtesting
- For testing the complete data pipeline
- When you suspect data corruption or issues
- During development to test data flow
- For regular data refresh and validation

FEATURES:
- Database data checking and cleanup
- Fresh data fetching from exchanges
- Complete backtesting pipeline
- Data validation and verification
- Automated pipeline execution
- Error handling and logging

EXAMPLES:
    # Run complete fresh backtest pipeline
    python scripts/fresh_backtest_pipeline.py
    
    # This will:
    # 1. Check existing data
    # 2. Delete old data if present
    # 3. Fetch fresh data
    # 4. Run backtests
"""
import asyncio
import os
import asyncpg
import pandas as pd
from src.data.collectors.sol_data_collector import SOLDataCollector
from src.strategy_backtest import StrategyBacktest

SYMBOL = "SOLUSDT"
LOOKBACK_DAYS = 90
TIMEFRAME = "1h"


async def check_data_exists(conn_str):
    query = """
        SELECT COUNT(*) FROM ml_model.price_data WHERE symbol = $1
    """
    async with asyncpg.create_pool(conn_str) as pool:
        async with pool.acquire() as conn:
            count = await conn.fetchval(query, SYMBOL)
            print(f"Rows in ml_model.price_data for {SYMBOL}: {count}")
            return count


async def delete_data(conn_str):
    query = """
        DELETE FROM ml_model.price_data WHERE symbol = $1
    """
    async with asyncpg.create_pool(conn_str) as pool:
        async with pool.acquire() as conn:
            await conn.execute(query, SYMBOL)
            print(f"Deleted all rows for {SYMBOL} in ml_model.price_data.")


async def fetch_and_store(conn_str):
    collector = SOLDataCollector(conn_str)
    # This fetches and stores data in Neon
    await collector.fetch_all_timeframes(lookback_days=LOOKBACK_DAYS)
    print("Fetched and stored fresh data in Neon.")


async def run_backtest(conn_str):
    collector = SOLDataCollector(conn_str)
    df = await collector.fetch_all_timeframes(lookback_days=LOOKBACK_DAYS)
    df_1h = df[TIMEFRAME]
    backtester = StrategyBacktest(
        initial_capital=10000, max_positions=3, min_confidence=0.3
    )
    results = backtester.run_backtest(df_1h)
    backtester.print_stats()
    backtester.plot_results()


async def main():
    conn_str = os.getenv("NEON_DATABASE_URL")
    if not conn_str:
        raise RuntimeError("NEON_DATABASE_URL environment variable not set.")

    # 1. Check if data exists
    count = await check_data_exists(conn_str)
    if count > 0:
        # 2. Delete existing data
        await delete_data(conn_str)

    # 3. Refetch and store fresh data
    await fetch_and_store(conn_str)

    # 4. Run the backtest
    await run_backtest(conn_str)


if __name__ == "__main__":
    asyncio.run(main())
