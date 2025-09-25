"""
Mock Binance exchange handler for testing purposes.
"""

import logging
from typing import List, Dict
from datetime import datetime, timezone

from src.exchanges.base import BaseExchangeHandler
from src.core.models import StandardizedCandle, TimeRange
from src.core.exceptions import ValidationError
from src.core.symbol_mapper import SymbolMapper

logger = logging.getLogger(__name__)


class MockBinanceHandler(BaseExchangeHandler):
    """
    Mock handler for Binance exchange data, supporting test mode for unit testing.
    When test mode is enabled, all data-fetching methods return static mock data.
    """

    def __init__(self, config):
        """Initialize mock Binance handler with configuration."""
        super().__init__(config)
        self.base_url = "https://api.binance.com"
        self.symbol_mapper = SymbolMapper()
        self._is_test_mode = False  # Add test mode flag
        # Define mock data for test mode
        self._mock_markets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._mock_candles = {
            "BTCUSDT": [
                StandardizedCandle(
                    timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    open=1,
                    high=2,
                    low=0.5,
                    close=1.5,
                    volume=100,
                    source="binance",
                    resolution="1D",
                    market="BTCUSDT",
                    raw_data={"mock": True},
                )
            ],
            "ETHUSDT": [
                StandardizedCandle(
                    timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    open=10,
                    high=12,
                    low=9,
                    close=11,
                    volume=200,
                    source="binance",
                    resolution="1D",
                    market="ETHUSDT",
                    raw_data={"mock": True},
                )
            ],
            "SOLUSDT": [
                StandardizedCandle(
                    timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    open=20,
                    high=22,
                    low=19,
                    close=21,
                    volume=300,
                    source="binance",
                    resolution="1D",
                    market="SOLUSDT",
                    raw_data={"mock": True},
                )
            ],
        }

        # Define markets in standard format for the symbol mapper
        self._standard_markets = [
            "BTC-USDT",
            "ETH-USDT",
            "SOL-USDT",
            "BNB-USDT",
            "XRP-USDT",
        ]

        # Register markets with symbol mapper
        for market in self._standard_markets:
            base, quote = market.split("-")
            # Register standard format
            self.symbol_mapper.register_symbol(
                exchange="binance",
                symbol=market,
                base_asset=base,
                quote_asset=quote,
                is_perpetual=False,
            )
            # Register Binance format
            binance_symbol = f"{base}{quote}"
            self.symbol_mapper.register_symbol(
                exchange="binance",
                symbol=binance_symbol,
                base_asset=base,
                quote_asset=quote,
                is_perpetual=False,
            )

        # Store available markets in Binance's native format
        self._available_markets = set(
            f"{market.split('-')[0]}{market.split('-')[1]}"  # Convert to Binance format
            for market in self._standard_markets
        )

        self.timeframe_map = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1D": "1d",
        }
        logger.info("Initialized mock Binance handler")

    async def start(self):
        """Start the mock handler."""
        await super().start()
        logger.info("Started mock Binance handler")

    async def stop(self):
        """Stop the mock handler."""
        await super().stop()
        logger.info("Stopped mock Binance handler")

    def validate_market(self, market: str) -> bool:
        """
        Validate if a market symbol is available.

        Args:
            market (str): Market symbol to validate (e.g., "BTCUSDT" or "BTC-USDT")

        Returns:
            bool: True if market is valid, False otherwise
        """
        if not isinstance(market, str):
            raise ValidationError("Market must be a string")

        try:
            # Convert market to Binance format
            binance_market = self._convert_market_symbol(market)
            return binance_market in self._available_markets or any(
                self._convert_market_symbol(m) == binance_market
                for m in self.config.markets
            )
        except ValueError:
            return False

    def _convert_market_symbol(self, market: str) -> str:
        """
        Convert internal market symbol to Binance format.
        Example:
            - BTC-USDT -> BTCUSDT
            - BTC-PERP -> BTCUSDT (Drift format to Binance)
            - BTC-USD -> BTCUSDT (Coinbase format to Binance)
        """
        if not isinstance(market, str):
            raise ValidationError("Market must be a string")

        try:
            # Always try the symbol mapper first
            result = self.symbol_mapper.to_exchange_symbol("binance", market)
            # Ensure the result is in Binance format (no hyphens)
            if "-" in result:
                # If we got a hyphenated result, convert it to Binance format
                base, quote = result.split("-")
                return f"{base}{quote}"
            return result

        except ValueError:
            logger.warning(
                f"Symbol mapper failed to convert {market}, "
                "falling back to basic conversion"
            )
            # Fallback to basic conversion if symbol mapper fails

            # First convert to uppercase
            market = market.upper()

            # If already in Binance format (no hyphen), return as is
            if "-" not in market:
                return market

            # Extract base and quote assets
            parts = market.split("-")
            base_asset = parts[0]

            # Always convert to USDT pair for Binance
            if len(parts) > 1 and parts[1] in ["PERP", "USD"]:
                return f"{base_asset}USDT"
            elif len(parts) > 1:
                return f"{base_asset}{parts[1]}"
            else:
                return f"{base_asset}USDT"

    def enable_test_mode(self):
        """
        Enable test mode. All data-fetching methods will return mock data.
        """
        self._is_test_mode = True

    def _setup_test_mode(self):
        """
        (Legacy) Enable test mode for backward compatibility with existing tests.
        """
        self.enable_test_mode()

    async def fetch_historical_candles(
        self, market: str, time_range: TimeRange, resolution: str
    ) -> List[StandardizedCandle]:
        """Fetch mock historical candle data."""
        if self._is_test_mode:
            # Always call _generate_mock_candles so it can be patched in tests
            return self._generate_mock_candles(market, time_range, resolution)
        self.validate_market(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        # Generate mock candles
        candles = []
        current_time = time_range.start
        while current_time <= time_range.end:
            candle = StandardizedCandle(
                timestamp=current_time,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
                source="binance",
                resolution=resolution,
                market=market,
                raw_data={"mock": True, "interval": interval},
            )
            candles.append(candle)
            # Increment time based on resolution
            seconds = int(resolution) * 60 if resolution.isdigit() else 86400
            current_time = datetime.fromtimestamp(
                current_time.timestamp() + seconds, tz=timezone.utc
            )

        return candles

    async def fetch_live_candles(
        self, market: str, resolution: str
    ) -> StandardizedCandle:
        """Fetch mock live candle data."""
        if self._is_test_mode:
            binance_market = self._convert_market_symbol(market)
            candles = self._mock_candles.get(binance_market, [])
            if candles:
                return candles[0]
            # Return a default mock candle if none found
            return StandardizedCandle(
                timestamp=datetime.now(timezone.utc),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
                source="binance",
                resolution=resolution,
                market=market,
                raw_data={
                    "mock": True,
                    "interval": self.timeframe_map.get(resolution, "1D"),
                },
            )
        self.validate_market(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        return StandardizedCandle(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            source="binance",
            resolution=resolution,
            market=market,
            raw_data={"mock": True, "interval": interval},
        )

    async def get_markets(self) -> List[str]:
        """Get available markets in Binance's native format."""
        if self._is_test_mode:
            return self._mock_markets
        return list(self._available_markets)

    async def get_exchange_info(self) -> Dict:
        """Get mock exchange information."""
        return {
            "timezone": "UTC",
            "serverTime": int(datetime.now(timezone.utc).timestamp() * 1000),
            "symbols": [
                {
                    "symbol": market,  # Already in Binance format
                    "status": "TRADING",
                    "baseAsset": self.symbol_mapper.from_exchange_symbol(
                        "binance", market
                    ).split("-")[0],
                    "quoteAsset": "USDT",
                    "isSpotTradingAllowed": True,
                }
                for market in self._available_markets
            ],
        }

    def _generate_mock_candles(
        self, market, time_range=None, resolution="1", start_time=None, end_time=None
    ):
        """Return a list of mock StandardizedCandle objects for test compatibility."""
        # Use the mock candle if available, else return a generic one
        if market in self._mock_candles:
            return self._mock_candles[market]
        from datetime import datetime, timezone

        return [
            StandardizedCandle(
                timestamp=datetime.now(timezone.utc),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
                source="binance",
                resolution=resolution,
                market=market,
                raw_data={"mock": True},
            )
        ]
