"""
Base exchange handler providing common functionality for all exchanges.
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Dict, Optional, Union
import aiohttp
import asyncio

from core.models import StandardizedCandle, ExchangeCredentials, TimeRange
from core.exceptions import ValidationError, ExchangeError, RateLimitError, ApiError
from core.config import ExchangeConfig
from core.symbol_mapper import SymbolMapper

logger = logging.getLogger(__name__)

class BaseExchangeHandler(ABC):
    """Abstract base class for all exchange handlers."""

    def __init__(self, config: ExchangeConfig, auth_handler=None):
        """
        Initialize the exchange handler with configuration.
        
        Args:
            config: Exchange configuration
            auth_handler: Authentication handler (optional)
        """
        self.config = config
        self.name = config.name
        self.credentials = config.credentials
        self.rate_limit = config.rate_limit
        self.markets = config.markets
        self.base_url = config.base_url
        
        # Authentication
        self._auth_handler = auth_handler
        
        # Rate limiting
        self._last_request_time = 0
        self._request_interval = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start the exchange handler."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        logger.info(f"Started {self.name} exchange handler")

    async def stop(self):
        """Stop the exchange handler."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info(f"Stopped {self.name} exchange handler")

    @abstractmethod
    async def fetch_historical_candles(
        self,
        market: str,
        time_range: TimeRange,
        resolution: str
    ) -> List[StandardizedCandle]:
        """Fetch historical candle data."""
        pass

    @abstractmethod
    async def fetch_live_candles(
        self,
        market: str,
        resolution: str
    ) -> StandardizedCandle:
        """Fetch live candle data."""
        pass

    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: int = 30,
        as_json: bool = True,
        authenticated: bool = False
    ) -> any:
        """
        Make an HTTP request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            headers: HTTP headers
            data: Request body data
            timeout: Request timeout in seconds
            as_json: Whether to parse response as JSON
            authenticated: Whether this is an authenticated request
            
        Returns:
            Response data (JSON or text)
            
        Raises:
            ApiError: If API returns an error
            ExchangeError: If request fails
        """
        if self._session is None:
            await self.start()
            
        # Handle rate limiting
        await self._handle_rate_limit()
        
        # Get authentication headers if needed
        if authenticated and self._auth_handler:
            auth_headers = self._auth_handler.get_auth_headers(
                method=method,
                endpoint=endpoint,
                params=params,
                data=data
            )
            
            # Merge with existing headers
            if headers:
                headers.update(auth_headers)
            else:
                headers = auth_headers

        try:
            url = f"{self.base_url}{endpoint}"
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=data,
                timeout=timeout
            ) as response:
                # Update last request time
                self._last_request_time = datetime.now().timestamp()

                if response.status == 401:
                    error_text = await response.text()
                    raise ApiError(f"Unauthorized: {error_text}")

                if response.status != 200:
                    error_text = await response.text()
                    raise ApiError(f"{self.name} API error: {response.status} - {error_text}")

                if as_json:
                    try:
                        return await response.json()
                    except Exception as e:
                        raise ApiError(f"Failed to parse JSON: {e}")
                else:
                    return await response.text()

        except aiohttp.ClientError as e:
            raise ExchangeError(f"Request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise ExchangeError("Request timed out")
        except Exception as e:
            raise ExchangeError(f"Unexpected error in {self.name}: {str(e)}")

    async def _handle_rate_limit(self):
        """Handle rate limiting between requests."""
        if self._last_request_time == 0:
            return

        current_time = datetime.now().timestamp()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self._request_interval:
            delay = self._request_interval - time_since_last_request
            logger.debug(f"Rate limiting {self.name}: waiting {delay:.2f} seconds")
            await asyncio.sleep(delay)

    def validate_market(self, market: str):
        """
        Validate market symbol.
        
        Args:
            market: Market symbol (can be in exchange-specific or standard format)
        
        Raises:
            ValidationError: If market is not supported by this exchange
        """
        if market in self.markets:
            return
        if self.validate_standard_symbol(market):
            return
        raise ValidationError(f"Invalid market {market} for {self.name}")

    def validate_standard_symbol(self, standard_symbol: str) -> bool:
        """
        Validate if a standard symbol is supported by this exchange.
        
        Args:
            standard_symbol: Symbol in standard format (e.g., BTC-USD) or exchange format (e.g., BTCUSDT)
            
        Returns:
            True if supported, False otherwise
        """
        self._init_symbol_mapper()
        if standard_symbol in self.markets:
            return True
        try:
            exchange_symbol = self._symbol_mapper.to_exchange_symbol(self.name, standard_symbol)
            return exchange_symbol in self.markets
        except ValueError:
            return False

    def standardize_timestamp(self, ts: Union[int, float, str, datetime]) -> datetime:
        try:
            if isinstance(ts, datetime):
                return ts
            elif isinstance(ts, str):
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            elif isinstance(ts, (int, float)):
                if len(str(int(ts))) > 10:
                    return datetime.fromtimestamp(ts / 1000)
                return datetime.fromtimestamp(ts)
            raise ValidationError(f"Unsupported timestamp format: {ts}")
        except Exception as e:
            raise ValidationError(f"Failed to standardize timestamp: {str(e)}")

    def validate_candle(self, candle: StandardizedCandle):
        try:
            candle.validate()
        except ValidationError as e:
            raise ValidationError(f"Invalid candle data for {self.name}: {str(e)}")

    def format_timeframe(self, resolution: str) -> str:
        return resolution

    def _init_symbol_mapper(self):
        if not hasattr(self, '_symbol_mapper'):
            self._symbol_mapper = SymbolMapper()
            if self.markets:
                self._symbol_mapper.register_exchange(self.name, self.markets)

    def convert_to_exchange_symbol(self, standard_symbol: str) -> str:
        self._init_symbol_mapper()
        return self._symbol_mapper.to_exchange_symbol(self.name, standard_symbol)

    def convert_from_exchange_symbol(self, exchange_symbol: str) -> str:
        self._init_symbol_mapper()
        return self._symbol_mapper.from_exchange_symbol(self.name, exchange_symbol)
