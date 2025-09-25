import os
import pandas as pd
from datetime import datetime
from shared.PostgresMarketDataAdapter import PostgresMarketDataAdapter
import asyncio

# Load Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

assert SUPABASE_URL and SUPABASE_KEY, (
    "Please set SUPABASE_URL and SUPABASE_KEY in your environment."
)

# Create the adapter
adapter = PostgresMarketDataAdapter()


async def smoke_test():
    # Create a dummy candle DataFrame
    df = pd.DataFrame([
        {
            "timestamp": datetime.utcnow(),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 10.5,
        }
    ])
    try:
        # Try to store the candle
        print("Storing dummy candle...")
        success = await adapter.store_candles(
            exchange="bitget",
            market="BTC/USDT",
            resolution="1m",
            df=df
        )
        print("Store result:", success)

        # Try to load the candle back
        print("Loading candles...")
        start_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end_time = datetime.utcnow()
        loaded = await adapter.load_candles(
            exchange="bitget",
            market="BTC/USDT",
            resolution="1m",
            start_time=start_time,
            end_time=end_time
        )
        print("Loaded candles:")
        print(loaded)
    except Exception as e:
        print("Smoke test failed:", e)


if __name__ == "__main__":
    asyncio.run(smoke_test()) 