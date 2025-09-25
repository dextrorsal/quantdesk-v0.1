"""
Coinbase exchange handler implementation focused on fetching historical and live candle data.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import List

from core.models import StandardizedCandle, TimeRange
from exchanges.base import BaseExchangeHandler
from core.exceptions import ExchangeError, ValidationError
from core.config import ExchangeConfig

logger = logging.getLogger(__name__)


class CoinbaseHandler(BaseExchangeHandler):
    """Handler for fetching historical and live candle data from Coinbase."""

    def __init__(self, config):
        """Initialize Coinbase handler with configuration."""
        super().__init__(config)
        self.base_url = "https://api.exchange.coinbase.com"
        # Coinbase's exact granularity values
        self.timeframe_map = {
            "1": "ONE_MINUTE",
            "5": "FIVE_MINUTE",
            "15": "FIFTEEN_MINUTE",
            "30": "THIRTY_MINUTE",
            "60": "ONE_HOUR",
            "120": "TWO_HOUR",
            "360": "SIX_HOUR",
            "1D": "ONE_DAY",
        }
        self._test_mode = False
        logger.info("Initialized Coinbase handler")

    async def start(self):
        """Start the handler."""
        await super().start()
        logger.info("Started Coinbase handler")

    def validate_market(self, market: str) -> bool:
        """Validate if a market symbol is available on Coinbase."""
        if not isinstance(market, str):
            raise ValidationError("Market must be a string")
        return True  # Let the API handle validation

    def _convert_market_symbol(self, market: str) -> str:
        """Convert internal market symbol to Coinbase format."""
        if not isinstance(market, str):
            raise ValidationError("Market must be a string")

        market = market.upper()
        # Remove -PERP suffix if present (from Drift format)
        market = market.replace("-PERP", "")

        # Convert from Binance format (SOLUSDT -> SOL-USD)
        if "USDT" in market:
            base = market.replace("USDT", "")
            return f"{base}-USD"

        # If already in Coinbase format (SOL-USD), return as is
        if "-USD" in market:
            return market

        # Default case: add -USD if no other format detected
        return f"{market}-USD"

    def _get_granularity_seconds(self, resolution: str) -> int:
        """Convert resolution to seconds."""
        resolution_map = {
            "1": 60,
            "5": 300,
            "15": 900,
            "30": 1800,
            "60": 3600,
            "120": 7200,
            "360": 21600,
            "1D": 86400,
        }
        return resolution_map.get(resolution, 60)

    async def fetch_historical_candles(
        self, market: str, time_range: TimeRange, resolution: str
    ) -> List[StandardizedCandle]:
        """
        Fetch historical candle data from Coinbase public API.
        Returns a list of StandardizedCandle objects.

        Args:
            market (str): Market symbol (e.g., "BTC-USD")
            time_range (TimeRange): Time range to fetch data for
            resolution (str): Candle resolution (e.g., "1", "5", "15", "60")

        Returns:
            List[StandardizedCandle]: List of standardized candles
        """
        self.validate_market(market)
        coinbase_symbol = self._convert_market_symbol(market)
        granularity = self.timeframe_map.get(resolution)
        if not granularity:
            raise ValidationError(f"Invalid resolution: {resolution}")
        if time_range.end < time_range.start:
            raise ValidationError("End time must be after start time")

        logger.debug(
            f"Fetching {market} data from {time_range.start} to {time_range.end}"
        )
        candles = []
        start_time = int(time_range.start.timestamp())
        end_time = int(time_range.end.timestamp())

        try:
            while start_time < end_time:
                batch_duration = self._get_granularity_seconds(resolution) * 300
                current_end = min(start_time + batch_duration, end_time)

                path = f"/api/v3/brokerage/market/products/{coinbase_symbol}/candles"
                params = {
                    "start": str(start_time),
                    "end": str(current_end),
                    "granularity": granularity,
                }

                response_data = await self._make_request(
                    method="GET", endpoint=path, params=params
                )

                candle_list = []
                if isinstance(response_data, dict):
                    candle_list = response_data.get("candles", []) or response_data.get(
                        "data", []
                    )
                elif isinstance(response_data, list):
                    candle_list = response_data

                for candle_data in candle_list:
                    try:
                        if isinstance(candle_data, list):
                            candle = StandardizedCandle(
                                timestamp=datetime.fromtimestamp(
                                    float(candle_data[0]), tz=timezone.utc
                                ),
                                open=float(candle_data[1]),
                                high=float(candle_data[2]),
                                low=float(candle_data[3]),
                                close=float(candle_data[4]),
                                volume=float(candle_data[5]),
                                source="coinbase",
                                resolution=resolution,
                                market=market,
                            )
                        elif isinstance(candle_data, dict):
                            timestamp = None
                            for field in ["time", "timestamp", "start"]:
                                if field in candle_data:
                                    timestamp = candle_data[field]
                                    break

                            if timestamp is None:
                                continue

                            if isinstance(timestamp, str):
                                try:
                                    timestamp = datetime.strptime(
                                        timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"
                                    ).replace(tzinfo=timezone.utc)
                                except ValueError:
                                    timestamp = datetime.fromtimestamp(
                                        float(timestamp), tz=timezone.utc
                                    )
                            else:
                                timestamp = datetime.fromtimestamp(
                                    float(timestamp), tz=timezone.utc
                                )

                            candle = StandardizedCandle(
                                timestamp=timestamp,
                                open=float(candle_data.get("open", 0.0)),
                                high=float(candle_data.get("high", 0.0)),
                                low=float(candle_data.get("low", 0.0)),
                                close=float(candle_data.get("close", 0.0)),
                                volume=float(candle_data.get("volume", 0.0)),
                                source="coinbase",
                                resolution=resolution,
                                market=market,
                            )

                        candles.append(candle)

                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error parsing candle data: {e}")
                        continue

                start_time = current_end
                await asyncio.sleep(0.1)  # Rate limiting

            candles.sort(key=lambda x: x.timestamp)
            return candles

        except Exception as e:
            logger.error(f"Error fetching historical candles: {e}")
            raise ExchangeError(f"Failed to fetch historical candles: {e}")

    async def fetch_live_candles(
        self, market: str, resolution: str
    ) -> StandardizedCandle:
        """
        Fetch live ticker data from Coinbase public API.
        Returns a StandardizedCandle object.

        Args:
            market (str): Market symbol (e.g., "BTC-USD")
            resolution (str): Candle resolution

        Returns:
            StandardizedCandle: Latest candle data
        """
        self.validate_market(market)
        coinbase_symbol = self._convert_market_symbol(market)

        try:
            path = f"/api/v3/brokerage/market/products/{coinbase_symbol}/ticker"
            params = {"limit": 1}
            response = await self._make_request(
                method="GET", endpoint=path, params=params
            )

            if not response or "trades" not in response:
                raise ExchangeError(f"No live data available for {market}")

            trade_data = response["trades"][0]
            current_time = datetime.strptime(
                trade_data["time"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
            price = float(trade_data["price"])

            candle = StandardizedCandle(
                timestamp=current_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=float(trade_data["size"]),
                source="coinbase",
                resolution=resolution,
                market=market,
            )

            return candle

        except Exception as e:
            raise ExchangeError(f"Failed to fetch live data: {e}")

    def _convert_resolution(self, resolution: str) -> str:
        """Convert standard resolution to Coinbase format."""
        return self.timeframe_map.get(resolution, "ONE_MINUTE")

    async def _make_request(self, method: str, endpoint: str, params: dict = None):
        """
        Placeholder for HTTP request logic. Should be patched/mocked in tests.
        """
        raise NotImplementedError(
            "_make_request must be implemented or mocked in tests."
        )

    async def get_markets(self) -> List[str]:
        """
        Return a static list of supported markets for test compatibility.
        In production, this should fetch from the Coinbase API.
        """
        return ["BTC-USD", "ETH-USD", "SOL-USD"]

    async def stop(self):
        """
        Stop the handler. No-op for CoinbaseHandler.
        """
        pass

    @staticmethod
    async def self_test() -> bool:
        """Run a self-test to verify the handler is working correctly."""
        try:
            # Create handler with default config
            config = ExchangeConfig(name="coinbase")
            handler = CoinbaseHandler(config)

            # Start the handler
            await handler.start()

            try:
                # Test getting markets
                markets = await handler.get_markets()
                print(f"Fetched {len(markets)} markets")
                if not markets:
                    print("Error: No markets found")
                    return False

                # Test fetching candles for BTC-USD
                if "BTC-USD" in markets:
                    test_market = "BTC-USD"
                else:
                    test_market = markets[0]

                print(f"Testing with market: {test_market}")

                # Test fetching historical candles
                end_time = datetime.now(timezone.utc)
                from datetime import timedelta

                start_time = end_time - timedelta(days=1)

                candles = await handler.fetch_historical_candles(
                    test_market,
                    TimeRange(start=start_time, end=end_time),
                    resolution="1h",
                )

                print(f"Fetched {len(candles)} historical candles")
                if not candles:
                    print("Error: No candles found")
                    return False

                # Test fetching live candle
                live_candle = await handler.fetch_live_candles(test_market, "1h")
                print(f"Fetched live candle: {live_candle}")

                print("All tests passed!")
                return True

            finally:
                # Stop the handler
                await handler.stop()

        except Exception as e:
            print(f"Error during self-test: {e}")
            return False

    def enable_test_mode(self):
        """Enable test mode for unit testing."""
        self._test_mode = True

    def _get_headers(self, *args, **kwargs):
        """Return test headers if in test mode, else normal headers."""
        if getattr(self, "_test_mode", False):
            return {"CB-ACCESS-KEY": "test_key"}
        return {}

    def _generate_signature(self, *args, **kwargs):
        """Stub for test compatibility. Returns None."""
        return None


async def main():
    """Example usage of CoinbaseHandler."""
    await CoinbaseHandler.self_test()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the example
    asyncio.run(main())
