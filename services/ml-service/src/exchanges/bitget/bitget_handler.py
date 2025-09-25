"""
Bitget exchange handler implementation.
"""

import logging
import hmac
import hashlib
from typing import List, Dict
from datetime import datetime
from core.models import StandardizedCandle, TimeRange
from core.exceptions import ExchangeError, ValidationError, RateLimitError
from exchanges.base import BaseExchangeHandler
from core.symbol_mapper import SymbolMapper

logger = logging.getLogger(__name__)


class BitgetHandler(BaseExchangeHandler):
    """Handler for Bitget exchange data, supporting both demo and live trading modes."""

    def __init__(self, config, mode=None):
        """
        Initialize Bitget handler with configuration.
        Args:
            config: Configuration object (should include API keys, etc.)
            mode: 'demo' for demo trading, 'live' for real trading. If None, will infer from config.name.
        """
        super().__init__(config)
        # Determine mode
        if mode is not None:
            self.mode = mode.lower()
        else:
            # Fallback to config.name if mode not explicitly set
            self.mode = getattr(config, 'name', '').lower()
            if 'demo' in self.mode:
                self.mode = 'demo'
            else:
                self.mode = 'live'

        # Set product type and symbol prefix based on mode
        if self.mode == 'demo':
            self.product_type = 'susdt-futures'
            # e.g., SBTCSUSDT
            self.symbol_prefix = 'S'
        else:
            self.product_type = 'usdt-futures'
            # e.g., BTCUSDT
            self.symbol_prefix = ''

        self.timeframe_map = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1D": "1D",
            "1W": "1W",
            "1M": "1M"
        }
        self.inverse_timeframe_map = {
            v: k for k, v in self.timeframe_map.items()
        }

    def _get_headers(self) -> Dict:
        """Get headers for API requests."""
        headers = {'Accept': 'application/json'}
        if self.mode == 'demo':
            # Required for demo trading per Bitget docs
            headers['paptrading'] = '1'
        return headers

    def _generate_signature(self, timestamp: str, method: str, endpoint: str, params: Dict = None, body: str = None) -> str:
        """Generate signed message for authenticated requests."""
        if not self.credentials or not self.credentials.api_secret:
            return None

        message = timestamp + method + endpoint
        if params:
            # Ensure consistent parameter order (if needed)
            message += '?' + '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        if body:
            message += body

        signature = hmac.new(
            self.credentials.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _convert_market_symbol(self, market: str) -> str:
        """
        Convert a market string to the correct Bitget symbol for the current mode.
        For demo: SBTCSUSDT (do not double-prepend S)
        For live: BTCUSDT_UMCBL (do not double-append _UMCBL)
        Best practice: Only append _UMCBL if the symbol is exactly 6 or 7 uppercase letters (e.g., BTCUSDT).
        If the symbol is already in the correct format, return as-is. Otherwise, raise a ValidationError.
        """
        base_quote = market.replace('-', '').upper()
        if self.mode == 'demo':
            # Only prepend 'S' if not already present
            if not base_quote.startswith('S'):
                return f"S{base_quote}"
            return base_quote
        else:
            if base_quote.endswith('_UMCBL'):
                return base_quote
            # Only allow appending if symbol is exactly 6 or 7 uppercase letters (e.g., BTCUSDT, ETHUSDT)
            if base_quote.isupper() and (6 <= len(base_quote) <= 7):
                return f"{base_quote}_UMCBL"
            # If the symbol is not in a recognized format, raise an error
            raise ValidationError(
                f"BitgetHandler: Invalid market symbol '{market}'. "
                "Expected format: 'BTCUSDT_UMCBL' or 'BTCUSDT'. "
                "Check your config/env for correct Bitget symbol formatting."
            )

    def _parse_raw_candle(self, raw_data: List, market: str, resolution: str) -> StandardizedCandle:
        """Parse raw candle data into StandardizedCandle format."""
        # Bitget Candle Data Format (verify from docs):
        # [ts, open, close, high, low, volume, ...] - Verify order and data types
        try:
            candle = StandardizedCandle(
                timestamp=self.standardize_timestamp(int(raw_data[0])), # Millisecond timestamp
                open=float(raw_data[1]),
                high=float(raw_data[2]),
                low=float(raw_data[3]),
                close=float(raw_data[4]),
                volume=float(raw_data[5]), # Verify volume is base or quote currency
                source='bitget',
                resolution=resolution,
                market=market,
                raw_data=raw_data
            )
            self.validate_candle(candle)
            return candle
        except (IndexError, ValueError) as e:
            raise ValidationError(f"Error parsing Bitget candle data: {str(e)}, Raw Data: {raw_data}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error parsing Bitget candle: {str(e)}, Raw Data: {raw_data}")


    async def fetch_historical_candles(self, market: str, time_range: TimeRange, resolution: str) -> List[StandardizedCandle]:
        """Fetch historical candle data from Bitget."""
        self.validate_market(market)
        bitget_symbol = self._convert_market_symbol(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        candles: List[StandardizedCandle] = []
        start_timestamp_ms = int(time_range.start.timestamp() * 1000)
        end_timestamp_ms = int(time_range.end.timestamp() * 1000)
        limit = 200  # Bitget API allows a maximum of 200 per request
        current_end_ms = end_timestamp_ms

        try:
            while current_end_ms > start_timestamp_ms:
                logger.info(
                    f"Bitget fetch loop: start_timestamp_ms={start_timestamp_ms}, "
                    f"current_end_ms={current_end_ms}"
                )
                if self.mode == 'demo':
                    endpoint = '/api/mix/v1/market/candles'
                    params = {
                        'symbol': bitget_symbol,
                        'granularity': interval,
                        'limit': limit,
                        'startTime': start_timestamp_ms,
                        'endTime': current_end_ms,
                        'productType': self.product_type
                    }
                else:
                    endpoint = '/api/mix/v1/market/history-candles'
                    params = {
                        'symbol': bitget_symbol,
                        'granularity': interval,
                        'limit': limit,
                        'startTime': start_timestamp_ms,
                        'endTime': current_end_ms,
                        'productType': self.product_type
                    }
                print(
                    f"Debug Bitget: fetch_historical_candles - Endpoint: {endpoint} "
                    f"Params: {params}"
                )

                response_data = await self._make_request(
                    method='GET',
                    endpoint=endpoint,
                    params=params,
                    headers=self._get_headers()
                )
                print(
                    f"Debug Bitget: fetch_historical_candles - Response Data: "
                    f"{response_data}"
                )

                # Handle both dict and list responses
                if isinstance(response_data, list):
                    raw_candles = response_data
                elif isinstance(response_data, dict) and 'data' in response_data:
                    raw_candles = response_data['data']
                else:
                    logger.info(
                        f"No more historical data from "
                        f"{datetime.fromtimestamp(start_timestamp_ms/1000)} to "
                        f"{datetime.fromtimestamp(current_end_ms/1000)} for "
                        f"{market} {resolution}"
                    )
                    break

                if not raw_candles:
                    break

                # Bitget returns candles in reverse order (newest first)
                for raw_candle in raw_candles:
                    try:
                        candle = self._parse_raw_candle(
                            raw_candle, market, resolution
                        )
                        candles.append(candle)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid candle: {e}")

                if len(raw_candles) < limit:
                    logger.info(
                        f"Less than limit ({limit}) candles received, "
                        f"assuming end of data for this chunk."
                    )
                    break

                # Find the oldest candle timestamp in the batch (last in list)
                oldest_candle_timestamp = int(raw_candles[-1][0])
                logger.info(
                    f"Bitget fetch loop: advancing current_end_ms from "
                    f"{current_end_ms} to {oldest_candle_timestamp - 1}"
                )
                new_end_ms = oldest_candle_timestamp - 1
                if new_end_ms >= current_end_ms:
                    logger.warning(
                        f"Bitget fetch loop: Detected stuck loop at {new_end_ms}, "
                        f"breaking to avoid infinite loop."
                    )
                    break
                current_end_ms = new_end_ms

                await self._handle_rate_limit()

        except RateLimitError:
            logger.warning("Bitget rate limit hit during historical data fetch.")
            raise
        except ExchangeError as e:
            raise ExchangeError(f"Bitget historical data fetch failed: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error fetching Bitget historical candles: {e}")

        # Deduplicate by timestamp as a safety net
        unique_candles = {}
        for candle in candles:
            ts = candle.timestamp
            if ts not in unique_candles:
                unique_candles[ts] = candle
        # Return sorted by timestamp (oldest to newest)
        return [unique_candles[ts] for ts in sorted(unique_candles)]


    async def fetch_live_candles(self, market: str, resolution: str) -> StandardizedCandle:
        """Fetch live candle data from Bitget."""
        self.validate_market(market)
        bitget_symbol = self._convert_market_symbol(market)
        interval = self.timeframe_map.get(resolution)
        if not interval:
            raise ValidationError(f"Invalid resolution: {resolution}")

        try:
            params = {
                'symbol': bitget_symbol,
                'period': interval, # Correct parameter name? Verify Bitget docs
                'limit': 1 # Get only the latest candle
            }
            print(f"Debug Bitget: fetch_live_candles - Params: {params}") # DEBUG

            response_data = await self._make_request(
                method='GET',
                endpoint='/api/spot/v1/market/candles', # Or /api/spot/v1/ticker ? Verify endpoint
                params=params,
                headers=self._get_headers()
            )
            print(f"Debug Bitget: fetch_live_candles - Response Data: {response_data}") # DEBUG

            if not response_data or not response_data['data']:
                raise ExchangeError(f"No live data available for {market} from Bitget")

            raw_candles = response_data['data'] # Assuming data is list of candles
            if raw_candles:
                return self._parse_raw_candle(raw_candles[0], market, resolution) # Parse the first (and should be only) candle
            else:
                raise ExchangeError(f"No candle data returned in live response for {market}")


        except ExchangeError as e:
            raise ExchangeError(f"Bitget live data fetch failed: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error fetching Bitget live candles: {e}")


    async def get_markets(self, market_type: str = 'usdtm') -> List[str]:
        """Get available markets from Bitget. market_type: 'spot' or 'usdtm'"""
        try:
            if market_type == 'spot':
                # Spot markets endpoint
                response_data = await self._make_request(
                    method='GET',
                    endpoint='/api/spot/v1/public/symbols',
                    headers=self._get_headers()
                )
                print(f"Debug Bitget: get_markets (spot) - Response Data: {response_data}")
                if not response_data or not response_data.get('data'):
                    raise ExchangeError("Could not retrieve spot market list from Bitget")
                markets = [item['symbol'] for item in response_data['data']]
                return markets
            else:
                # USDT-M futures endpoint
                response_data = await self._make_request(
                    method='GET',
                    endpoint='/api/mix/v1/market/contracts',
                    params={'productType': 'umcbl'},
                    headers=self._get_headers()
                )
                print(f"Debug Bitget: get_markets (usdtm) - Response Data: {response_data}")
                if not response_data or not response_data.get('data'):
                    raise ExchangeError("Could not retrieve USDT-M market list from Bitget")
                markets = [item['symbol'] for item in response_data['data']]
                return markets
        except ExchangeError as e:
            raise ExchangeError(f"Failed to fetch Bitget markets: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error getting Bitget markets: {e}")


    async def get_exchange_info(self) -> Dict:
        """Get exchange information from Bitget."""
        try:
            response_data = await self._make_request(
                method='GET',
                endpoint='/api/spot/v1/common/timestamp', # Or exchange info endpoint? Verify Bitget API for general info
                headers=self._get_headers()
            )
            server_time = response_data['serverTime'] if response_data and 'serverTime' in response_data else None # Adjust key if needed
            print(f"Debug Bitget: get_exchange_info - Timestamp Response: {response_data}") # DEBUG


            exchange_info = {
                "name": self.name,
                "markets": await self.get_markets(),
                "timeframes": list(self.timeframe_map.values()),
                "has_live_data": True, # Assume live data is supported
                "rate_limit": self.rate_limit, # Use default rate limit from base class for now
                "server_time": server_time, # Server time from endpoint
                "exchange_filters": [] # Bitget might not have exchange filters like Binance in same format - adjust as needed.
            }
            return exchange_info

        except ExchangeError as e:
            raise ExchangeError(f"Failed to fetch Bitget exchange info: {e}")
        except Exception as e:
            raise ExchangeError(f"Unexpected error getting Bitget exchange info: {e}")