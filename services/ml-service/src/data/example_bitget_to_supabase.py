# No longer need to modify sys.path
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../d3x7-algo/src'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

import asyncio
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
from dotenv import load_dotenv
from core.config import Config
from core.models import TimeRange
from exchanges import get_exchange_handler
from data.PostgresMarketDataAdapter import PostgresMarketDataAdapter
from data.SupabaseAdapter import SupabaseAdapter
from data.bitget_ws_to_supabase import BitgetWsToSupabase


# Set up logging
# ... existing code ...

async def main():
    """
    An example of using the BitgetWsToSupabase class to fetch live trade data
    and store it in Supabase.
    """
    load_dotenv()
    # Load configuration
    config = Config()
    supabase_url = config.get_supabase_url()
    supabase_key = config.get_supabase_key()

    # Define parameters
    symbol = "BTCUSDT"
    table_name = "bitget_trades"
    # Define time range for fetching data (e.g., last 1 minute)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=1)
    time_range = TimeRange(start=start_time, end=end_time)

    # Initialize Supabase adapter
    supabase_adapter = SupabaseAdapter(supabase_url, supabase_key)

    # Initialize data adapter
    data_adapter = PostgresMarketDataAdapter(supabase_adapter)

    # Initialize exchange handler
    exchange_handler = get_exchange_handler("bitget", "ws", {"symbol": symbol})

    # Initialize BitgetWsToSupabase
    service = BitgetWsToSupabase(exchange_handler, data_adapter, table_name)

    # Start fetching and storing data
    await service.run()

    print("Example run finished.")


if __name__ == "__main__":
    asyncio.run(main()) 