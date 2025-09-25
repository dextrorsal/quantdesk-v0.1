"""
JupiterHandler for interacting with Jupiter's new API endpoints.
Uses the Ultra API for price data and trading.
"""

import logging
from datetime import datetime, timezone
from typing import List
import asyncio
import pandas as pd
from io import StringIO
import aiohttp

from core.models import StandardizedCandle, TimeRange
from core.exceptions import ExchangeError, ValidationError
from exchanges.base import BaseExchangeHandler

logger = logging.getLogger(__name__)

class JupiterHandler(BaseExchangeHandler):
    """Handler for Jupiter DEX using the Ultra API endpoints."""

    def __init__(self, config):
        """
        Initialize the Jupiter handler with the given configuration.
        """
        super().__init__(config)
        # Update to use Ultra API endpoints
        self.ultra_api_url = "https://api.jup.ag/ultra/v1"
        self.price_url = "https://price.jup.ag/v4/price"  # Price API v4
        logger.info(f"JupiterHandler initialized with Ultra API URL: {self.ultra_api_url}")
        
        # Token addresses (same as in jup_adapter.py)
        self.tokens = {
            "SOL": "So11111111111111111111111111111111111111112",
            "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        }

    async def fetch_historical_candles(
        self,
        market: str,
        time_range: TimeRange,
        resolution: str
    ) -> List[StandardizedCandle]:
        """
        Jupiter does not provide historical candle data via this API.
        """
        logger.warning("Historical candle fetching is not supported for Jupiter.")
        raise NotImplementedError("Historical candle fetching is not implemented for Jupiter.")

    async def fetch_live_candles(
        self,
        market: str,
        resolution: str
    ) -> StandardizedCandle:
        """
        Fetch the latest price for a given market from Jupiter's price endpoint,
        and simulate a candlestick with open=high=low=close equal to the current price.
        
        For demonstration, this implementation supports "SOL-USDC" only.
        """
        # For "SOL-USDC", use the well-known mint addresses:
        if market.upper() == "SOL-USDC":
            input_mint = self.tokens["SOL"]
            output_mint = self.tokens["USDC"]
        else:
            raise ValidationError(f"Market {market} not supported by JupiterHandler.")

        try:
            # Use the price API for efficient price data
            params = {
                "ids": input_mint,
                "vsToken": output_mint
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.price_url, params=params) as response:
                    if response.status != 200:
                        raise ExchangeError(f"Failed to fetch price data: {await response.text()}")
                    
                    data = await response.json()
                    
                    if "data" not in data or input_mint not in data["data"]:
                        raise ExchangeError("Invalid price response format")
                    
                    price_data = data["data"][input_mint]
                    price = float(price_data["price"])

            # Construct a standardized candle using the current price
            candle = StandardizedCandle(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,  # Volume not provided by the price API
                source="jupiter",
                resolution=resolution,
                market=market,
                raw_data=data
            )
            
            return candle
                
        except Exception as e:
            logger.error(f"Error fetching live price from Jupiter: {e}")
            raise ExchangeError(f"Failed to fetch live price from Jupiter: {e}")
