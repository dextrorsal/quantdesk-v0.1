"""
Price fetching service for cryptocurrency conversions.
Supports USD and CAD price lookups using CoinGecko API.
"""

import aiohttp
import logging
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PriceService:
    """Service for fetching cryptocurrency prices in various fiat currencies."""
    
    def __init__(self):
        """Initialize the price service with caching."""
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)  # Cache prices for 5 minutes
        self.base_url = "https://api.coingecko.com/api/v3"
        
    async def get_token_price(self, token_id: str = "solana") -> Optional[Tuple[float, float]]:
        """
        Get token price in USD and CAD.
        
        Args:
            token_id: CoinGecko token ID (default: "solana")
            
        Returns:
            Tuple of (USD price, CAD price) or None if fetch fails
        """
        # Check cache first
        if token_id in self.cache:
            timestamp, (usd_price, cad_price) = self.cache[token_id]
            if datetime.now() - timestamp < self.cache_duration:
                return usd_price, cad_price
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/simple/price"
                params = {
                    "ids": token_id,
                    "vs_currencies": "usd,cad"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if token_id in data:
                            usd_price = data[token_id]["usd"]
                            cad_price = data[token_id]["cad"]
                            
                            # Update cache
                            self.cache[token_id] = (datetime.now(), (usd_price, cad_price))
                            
                            return usd_price, cad_price
                    elif response.status == 429:  # Rate limit
                        logger.warning("Rate limit hit for CoinGecko API")
                        # Return cached data if available
                        if token_id in self.cache:
                            return self.cache[token_id][1]
                    else:
                        logger.error(f"Failed to fetch price. Status: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            # Return cached data if available
            if token_id in self.cache:
                return self.cache[token_id][1]
        
        return None

    async def get_token_prices(self, token_ids: list[str]) -> Dict[str, Tuple[float, float]]:
        """
        Get prices for multiple tokens in USD and CAD.
        
        Args:
            token_ids: List of CoinGecko token IDs
            
        Returns:
            Dictionary of token_id -> (USD price, CAD price)
        """
        results = {}
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/simple/price"
                params = {
                    "ids": ",".join(token_ids),
                    "vs_currencies": "usd,cad"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for token_id in token_ids:
                            if token_id in data:
                                usd_price = data[token_id]["usd"]
                                cad_price = data[token_id]["cad"]
                                results[token_id] = (usd_price, cad_price)
                                # Update cache
                                self.cache[token_id] = (datetime.now(), (usd_price, cad_price))
                    else:
                        logger.error(f"Failed to fetch prices. Status: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            
        # Fill in any missing results from cache
        for token_id in token_ids:
            if token_id not in results and token_id in self.cache:
                results[token_id] = self.cache[token_id][1]
                
        return results

# Create a singleton instance
_price_service = None

def get_price_service() -> PriceService:
    """Get or create the singleton price service instance."""
    global _price_service
    if _price_service is None:
        _price_service = PriceService()
    return _price_service 