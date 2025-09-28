import logging
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

from src.ultimate_fetcher import UltimateDataFetcher
from src.core.models import TimeRange

# Load environment variables from .env file
# The script will look for a .env file in the current working directory.
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_and_store")

# Configuration for the data fetch
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 5, 1, tzinfo=timezone.utc)
RESOLUTION = "1h"

async def main():
    """
    Main function to initialize and run the data fetcher.
    """
    logger.info("Starting data fetch process...")

    fetcher = UltimateDataFetcher()
    await fetcher.start()

    time_range = TimeRange(start=START_DATE, end=END_DATE)

    try:
        await fetcher.fetch_historical_data(
            markets=SYMBOLS,
            time_range=time_range,
            resolution=RESOLUTION
        )
        logger.info("Successfully fetched and stored data.")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}", exc_info=True)
    finally:
        await fetcher.stop()

    logger.info("Data fetch process completed.")

if __name__ == "__main__":
    asyncio.run(main())
