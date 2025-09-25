"""
Ultimate Data Fetcher - Main orchestrator for fetching and managing crypto data.

USE CASES:
- **Multi-exchange orchestration**: Manages multiple exchanges 
  (Bitget, Binance, Coinbase) simultaneously
- **Database integration**: Stores data directly to Supabase/Neon databases
- **Symbol mapping**: Handles exchange-specific symbol formats automatically
- **Live data streaming**: Supports real-time data fetching with WebSocket 
  connections
- **Historical data fetching**: Batch historical data retrieval with time range 
  support
- **ML pipeline integration**: Provides PyTorch/TensorFlow dataloaders for 
  model training

DIFFERENCES FROM OTHER FETCHERS:
- UltimateDataFetcher: Multi-exchange orchestrator with database storage
- ReliableDataFetcher: CCXT-based, fast, reliable exchanges only (Binance/Coinbase)
- DailyDataFetcher: Automated daily CSV storage with multiple exchanges
- CSVDataFetcher: CSV-only storage, no database integration
- DatabaseLoader: Simple database loading interface (like pd.read_csv)

EXAMPLES:
    # Multi-exchange historical data fetching
    async with UltimateDataFetcher() as fetcher:
        await fetcher.fetch_historical_data(
            markets=["BTC/USDT", "ETH/USDT"],
            time_range=TimeRange(
                start=datetime(2024, 1, 1), 
                end=datetime(2024, 12, 31)
            ),
            resolution="15m",
            exchanges=["binance", "coinbase"]
        )
    
    # Live data streaming
    await fetcher.start_live_fetching(
        markets=["BTC/USDT"],
        resolution="1m",
        exchanges=["bitget"]
    )
    
    # ML dataloader creation
    dataloader = fetcher.get_pytorch_dataloader(
        market="BTC/USDT",
        resolution="15m", 
        exchange="binance",
        batch_size=32
    )
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional
import argparse
from dotenv import load_dotenv
import pandas as pd

from .core.models import TimeRange
from .core.exceptions import DataFetcherError
from .utils.log_setup import setup_logging
from src.core.symbol_mapper import SymbolMapper
from src.data.PostgresMarketDataAdapter import PostgresMarketDataAdapter
from src.data.NeonAdapter import NeonAdapter
from .core.config import Config

logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)


class UltimateDataFetcher:
    """Main orchestrator for fetching and managing crypto data."""

    def __init__(self, config_path: str = ".env"):
        """Initialize the data fetcher with configuration from .env."""
        # Load environment variables from .env
        load_dotenv(dotenv_path=os.path.abspath(config_path))
        # Load Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        print(f"DEBUG: UltimateDataFetcher Supabase URL: {supabase_url}")
        print(
            f"DEBUG: UltimateDataFetcher Supabase Key: {'REDACTED' if supabase_key else None}"
        )
        if not (supabase_url and supabase_key):
            raise ValueError(
                "Supabase URL and Key must be provided in .env for Supabase integration."
            )
        self.data_manager = PostgresMarketDataAdapter()
        # Neon fallback
        neon_conn_str = os.getenv("NEON_DATABASE_URL")
        if not neon_conn_str:
            raise ValueError("NEON_DATABASE_URL must be set in .env for Neon fallback.")
        self.neon_manager = NeonAdapter(neon_conn_str)
        self.exchange_handlers = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the data fetcher and initialize all enabled exchanges from .env."""
        logger.info("Starting Ultimate Data Fetcher")
        self.initialize_symbol_mapper()
        # Example: EXCHANGES=bitget,binance
        exchanges = os.getenv("EXCHANGES", "bitget").split(",")
        for exchange_name in exchanges:
            exchange_name = exchange_name.strip()
            enabled = os.getenv(f"{exchange_name.upper()}_ENABLED", "true").lower() == "true"
            if not enabled:
                continue
            try:
                handler = self.get_exchange_handler_from_env(exchange_name)
                await handler.start()
                self.exchange_handlers[exchange_name] = handler
                logger.info(f"Initialized {exchange_name} handler")
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name} handler: {e}")
                self.initialize_symbol_mapper()

    def get_exchange_handler_from_env(self, exchange_name):
        from src.exchanges import get_exchange_handler

        config = Config()
        exchange_config = config.exchanges.get(exchange_name)
        if not exchange_config:
            raise ValueError(f"No config found for exchange: {exchange_name}")
        return get_exchange_handler(exchange_config)

    async def stop(self):
        """Stop all exchange handlers and cleanup."""
        logger.info("Stopping Ultimate Data Fetcher")
        for name, handler in self.exchange_handlers.items():
            try:
                await handler.stop()
                logger.info(f"Stopped {name} handler")
            except Exception as e:
                logger.error(f"Error stopping {name} handler: {e}")

    def initialize_symbol_mapper(self):
        """Initialize the symbol mapper with all available markets."""
        self.symbol_mapper = SymbolMapper()

        # Register markets for each exchange
        for exchange_name, handler in self.exchange_handlers.items():
            try:
                if hasattr(handler, "markets") and handler.markets:
                    self.symbol_mapper.register_exchange(exchange_name, handler.markets)
                elif hasattr(handler, "config") and hasattr(handler.config, "markets"):
                    # Use markets from config if available
                    self.symbol_mapper.register_exchange(
                        exchange_name, handler.config.markets
                    )
                else:
                    logger.warning(
                        f"No markets available for {exchange_name} during symbol mapper initialization"
                    )
            except Exception as e:
                logger.warning(
                    f"Error registering markets for {exchange_name}: {str(e)}"
                )

        # Count the number of exchanges with registered symbols
        registered_exchanges = len(self.symbol_mapper.supported_symbols)
        logger.info(f"Initialized symbol mapper with {registered_exchanges} exchanges")

    async def fetch_historical_data(
        self,
        markets: List[str],
        time_range: TimeRange,
        resolution: str,
        exchanges: Optional[List[str]] = None,
    ):
        """Fetch historical data for specified markets and time range."""
        if not exchanges:
            exchanges = list(self.exchange_handlers.keys())

        # Initialize symbol mapper if not already done
        if not hasattr(self, "symbol_mapper") or self.symbol_mapper is None:
            self.initialize_symbol_mapper()

        for exchange_name in exchanges:
            handler = self.exchange_handlers.get(exchange_name)
            if not handler:
                logger.warning(f"Exchange {exchange_name} not initialized, skipping")
                continue

            for market_symbol in markets:
                try:
                    # First, check if the market is directly in the handler's markets list
                    is_valid_market = False

                    # Safely check if markets attribute exists
                    if hasattr(handler, "markets") and handler.markets:
                        is_valid_market = market_symbol in handler.markets

                    # If not found directly, try validating as a standard symbol
                    if not is_valid_market:
                        try:
                            result = handler.validate_standard_symbol(market_symbol)
                            if asyncio.iscoroutine(result):
                                is_valid_market = await result
                            else:
                                is_valid_market = result
                        except Exception as e:
                            logger.warning(
                                f"Error validating {market_symbol} on {exchange_name}: {str(e)}"
                            )
                            continue

                    if not is_valid_market:
                        logger.info(
                            f"Market {market_symbol} not supported by {exchange_name}, skipping"
                        )
                        continue

                    # Get the exchange-specific format if needed
                    try:
                        if (
                            hasattr(handler, "markets")
                            and handler.markets
                            and market_symbol in handler.markets
                        ):
                            exchange_market = market_symbol
                        elif hasattr(self, "symbol_mapper") and self.symbol_mapper:
                            try:
                                exchange_market = self.symbol_mapper.to_exchange_symbol(
                                    exchange_name, market_symbol
                                )
                            except Exception:
                                # If symbol mapping fails, try using the original symbol
                                exchange_market = market_symbol
                        else:
                            exchange_market = market_symbol
                    except Exception as e:
                        logger.warning(
                            f"Symbol mapping error for {market_symbol} on {exchange_name}: {str(e)}"
                        )
                        exchange_market = market_symbol

                    logger.info(
                        f"Fetching historical data for {market_symbol} (exchange format: {exchange_market}) from {exchange_name}"
                    )
                    debug_msg_1 = (
                        "DEBUG: About to fetch candles for market_symbol="
                        f"{market_symbol}, "
                    )
                    debug_msg_2 = (
                        "exchange_market="
                        f"{exchange_market}, "
                    )
                    debug_msg_3 = (
                        "exchange="
                        f"{exchange_name}"
                    )
                    print(debug_msg_1 + debug_msg_2 + debug_msg_3)
                    try:
                        candles = await handler.fetch_historical_candles(
                            exchange_market,
                            time_range,
                            resolution,
                        )
                        print(f"DEBUG: Fetched {len(candles)} candles for {exchange_market} from {exchange_name}")
                        if candles:
                            df = pd.DataFrame([c.__dict__ for c in candles])
                            print(f"DEBUG: DataFrame shape before store: {df.shape}")
                            print(f"DEBUG: First 5 rows before store:\n{df.head()}")
                            if 'ts' in df.columns:
                                print(f"DEBUG: Unique ts count: {df['ts'].nunique()} / {len(df)}")
                            elif 'timestamp' in df.columns:
                                print(f"DEBUG: Unique timestamp count: {df['timestamp'].nunique()} / {len(df)}")
                            # --- Fallback Storage Logic ---
                            store_result = False
                            try:
                                store_result = self.data_manager.store_candles(
                                    exchange_name,
                                    market_symbol,
                                    resolution,
                                    df
                                )
                                print(f"DEBUG: Supabase store result: {store_result}")
                            except Exception as e:
                                logger.error(f"Supabase store failed: {e}")
                                store_result = False
                            if not store_result:
                                try:
                                    neon_result = await self.neon_manager.store_candles(
                                        exchange_name,
                                        market_symbol,
                                        resolution,
                                        df
                                    )
                                    print(f"DEBUG: Neon store result: {neon_result}")
                                    if not neon_result:
                                        print("DEBUG: Both Supabase and Neon failed, data will be written to CSV by SupabaseAdapter fallback.")
                                except Exception as e:
                                    logger.error(f"Neon store failed: {e}")
                                    print("DEBUG: Both Supabase and Neon failed, data will be written to CSV by SupabaseAdapter fallback.")
                        else:
                            print(f"DEBUG: No candles fetched for {exchange_market} from {exchange_name}")
                    except Exception as e:
                        logger.error(f"Error fetching/storing data for {market_symbol} on {exchange_name}: {e}")

                except Exception as e:
                    logger.error(
                        f"Error processing market {market_symbol} from {exchange_name}: {e}"
                    )

    async def start_live_fetching(
        self, markets: List[str], resolution: str, exchanges: Optional[List[str]] = None
    ):
        """Start live data fetching for specified markets."""
        if not exchanges:
            exchanges = list(self.exchange_handlers.keys())

        logger.info(
            f"Starting live data fetching for {markets} with resolution {resolution}"
        )
        logger.info(f"Using exchanges: {exchanges}")

        try:
            while True:
                for exchange_name in exchanges:
                    handler = self.exchange_handlers.get(exchange_name)
                    if not handler:
                        continue

                    for standard_market in markets:
                        try:
                            exchange_market = self.symbol_mapper.to_exchange_symbol(
                                exchange_name, standard_market
                            )

                            # Validate market symbol; await if necessary.
                            result = handler.validate_standard_symbol(standard_market)
                            if asyncio.iscoroutine(result):
                                valid = await result
                            else:
                                valid = result
                            if not valid:
                                continue

                            logger.debug(
                                f"Fetching live data for {standard_market} from {exchange_name}"
                            )
                            candle = await handler.fetch_live_candles(
                                market=exchange_market, resolution=resolution
                            )

                            # Store using PostgresMarketDataAdapter
                            self.data_manager.store_candles(
                                [candle],
                                exchange=exchange_name,
                                market=standard_market,
                                resolution=resolution,
                            )

                            logger.debug(
                                f"Stored live candle for {standard_market} from {exchange_name}"
                            )

                        except Exception as e:
                            logger.error(
                                f"Error fetching live data for {standard_market} from {exchange_name}: {e}"
                            )

                await asyncio.sleep(60)  # Adjust based on resolution

        except KeyboardInterrupt:
            logger.info("Live data fetching interrupted by user")
        except Exception as e:
            logger.error(f"Error in live data fetching: {e}")
            raise

    async def convert_historical_to_tfrecord(
        self,
        markets: List[str],
        time_range: TimeRange,
        resolution: str,
        exchanges: Optional[List[str]] = None,
    ):
        """
        Convert historical data to TFRecord format for machine learning.

        Args:
            markets: List of market symbols
            time_range: Time range to convert
            resolution: Candle resolution
            exchanges: Optional list of exchanges to use
        """
        if not exchanges:
            exchanges = list(self.exchange_handlers.keys())

        logger.info(
            f"Converting historical data to TFRecord format for {markets} with resolution {resolution}"
        )

        for exchange_name in exchanges:
            for market_symbol in markets:
                try:
                    # Convert market symbol if needed
                    if exchange_name in self.symbol_mapper.exchanges:
                        exchange_market = self.symbol_mapper.to_exchange_symbol(
                            exchange_name, market_symbol
                        )
                    else:
                        exchange_market = market_symbol

                    logger.info(
                        f"Converting {exchange_name}/{exchange_market} data to TFRecord format"
                    )

                    # Convert to TFRecord
                    self.data_manager.convert_to_tfrecord(
                        exchange_name,
                        exchange_market,
                        resolution,
                        time_range.start,
                        time_range.end,
                    )

                    logger.info(
                        f"Converted {exchange_name}/{exchange_market} data to TFRecord format"
                    )

                except Exception as e:
                    logger.error(
                        f"Error converting {market_symbol} from {exchange_name} to TFRecord: {e}"
                    )

        logger.info("TFRecord conversion completed")

    def get_pytorch_dataloader(
        self,
        market: str,
        resolution: str,
        exchange: str,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        """
        Get a PyTorch DataLoader for the specified data.

        Args:
            market: Market symbol
            resolution: Candle resolution
            exchange: Exchange name
            batch_size: Batch size
            shuffle: Whether to shuffle the data

        Returns:
            torch.utils.data.DataLoader: DataLoader for the data
        """
        # Convert market symbol if needed
        if exchange in self.symbol_mapper.exchanges:
            exchange_market = self.symbol_mapper.to_exchange_symbol(exchange, market)
        else:
            exchange_market = market

        return self.data_manager.get_pytorch_dataloader(
            exchange, exchange_market, resolution, batch_size, shuffle
        )

    def get_tensorflow_dataset(
        self, market: str, resolution: str, exchange: str, batch_size: int = 32
    ):
        """
        Get a TensorFlow Dataset for the specified data.

        Args:
            market: Market symbol
            resolution: Candle resolution
            exchange: Exchange name
            batch_size: Batch size

        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Convert market symbol if needed
        if exchange in self.symbol_mapper.exchanges:
            exchange_market = self.symbol_mapper.to_exchange_symbol(exchange, market)
        else:
            exchange_market = market

        return self.data_manager.get_tensorflow_dataset(
            exchange, exchange_market, resolution, batch_size
        )

# Move main() and script entry point to module level

async def main():
    """Main entry point for the Ultimate Data Fetcher."""
    parser = argparse.ArgumentParser(description="Ultimate Crypto Data Fetcher")
    parser.add_argument(
        "--config", default=".env", help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live"],
        required=True,
        help="Fetching mode: historical or live",
    )
    parser.add_argument(
        "--markets", nargs="+", help="Markets to fetch (e.g., BTC-PERP ETH-PERP)"
    )
    parser.add_argument("--exchanges", nargs="+", help="Exchanges to use")
    parser.add_argument(
        "--resolution",
        default="1",
        choices=["1", "15", "60", "240", "1D", "1W", "5", "30"],
        help="Candle resolution (e.g., 1, 5, 15)",
    )
    parser.add_argument(
        "--start-date", help="Start date for historical data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", help="End date for historical data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--convert-tfrecord",
        action="store_true",
        help="Convert historical data to TFRecord format",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for TFRecord datasets",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Initialize fetcher
    fetcher = UltimateDataFetcher(args.config)
    await fetcher.start()

    try:
        if args.mode == "historical":
            if not args.start_date or not args.end_date:
                raise DataFetcherError(
                    "Start and end dates required for historical mode"
                )

            time_range = TimeRange(
                start=datetime.strptime(args.start_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ),
                end=datetime.strptime(args.end_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ),
            )

            await fetcher.fetch_historical_data(
                args.markets, time_range, args.resolution, args.exchanges
            )
        elif args.mode == "live":
            await fetcher.start_live_fetching(
                markets=args.markets,
                resolution=args.resolution,
                exchanges=args.exchanges,
            )
        elif args.convert_tfrecord:
            # Convert historical data to TFRecord format
            time_range = TimeRange(
                start=datetime.fromisoformat(args.start_date),
                end=datetime.fromisoformat(args.end_date),
            )

            await fetcher.convert_historical_to_tfrecord(
                markets=args.markets,
                time_range=time_range,
                resolution=args.resolution,
                exchanges=args.exchanges,
            )
    finally:
        await fetcher.stop()

if __name__ == "__main__":
    asyncio.run(main())
