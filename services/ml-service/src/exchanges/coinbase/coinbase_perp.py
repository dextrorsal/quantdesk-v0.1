"""
Coinbase Perpetual Futures trading implementation.
Handles perpetual futures trading operations including portfolio management,
order placement, and position tracking.
"""

import logging
import hmac
import hashlib
import time
import base64
import json
from typing import Dict, Optional
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class CoinbasePerp:
    """Handler for Coinbase Perpetual Futures trading."""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False):
        """
        Initialize Coinbase Perpetual Futures handler.
        
        Args:
            api_key (str): Your Coinbase API key
            api_secret (str): Your Coinbase API secret
            sandbox (bool): Whether to use sandbox environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com"
        self.sandbox = sandbox
        if sandbox:
            self.base_url = "https://api-public.sandbox.exchange.coinbase.com"

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate signature for API authentication."""
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')

    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict:
        """Generate headers for API request."""
        timestamp = str(int(time.time()))
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': self._generate_signature(timestamp, method, request_path, body),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }

    def get_perpetuals_portfolio(self) -> Dict:
        """Get perpetuals portfolio information."""
        endpoint = "/api/v3/brokerage/portfolios"
        headers = self._get_headers("GET", endpoint)
        response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
        return response.json()

    def move_funds_to_perp_portfolio(self, amount: float, source_portfolio_uuid: str, 
                                   target_portfolio_uuid: str) -> Dict:
        """
        Move funds to perpetuals portfolio.
        
        Args:
            amount (float): Amount of USDC to move
            source_portfolio_uuid (str): Source portfolio UUID
            target_portfolio_uuid (str): Target perpetuals portfolio UUID
        """
        endpoint = "/api/v3/brokerage/portfolios/move_funds"
        body = {
            "funds": {
                "value": str(amount),
                "currency": "USDC"
            },
            "source_portfolio_uuid": source_portfolio_uuid,
            "target_portfolio_uuid": target_portfolio_uuid
        }
        headers = self._get_headers("POST", endpoint, json.dumps(body))
        response = requests.post(
            f"{self.base_url}{endpoint}",
            headers=headers,
            json=body
        )
        return response.json()

    def create_perp_order(self, 
                         product_id: str,
                         side: str,
                         leverage: float,
                         size: float,
                         price: Optional[float] = None,
                         order_type: str = "MARKET") -> Dict:
        """
        Create a perpetual futures order.
        
        Args:
            product_id (str): Product ID (e.g., 'SOL-PERP')
            side (str): 'BUY' or 'SELL'
            leverage (float): Leverage amount (max 20x)
            size (float): Order size in base currency
            price (float, optional): Limit price (required for LIMIT orders)
            order_type (str): 'MARKET' or 'LIMIT'
        """
        if leverage > 20:
            raise ValueError("Maximum leverage is 20x")

        endpoint = "/api/v3/brokerage/orders"
        body = {
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                order_type.lower(): {
                    "size": str(size),
                    "leverage": str(leverage)
                }
            }
        }
        
        if order_type == "LIMIT" and price is not None:
            body["order_configuration"]["limit"]["price"] = str(price)

        headers = self._get_headers("POST", endpoint, json.dumps(body))
        response = requests.post(
            f"{self.base_url}{endpoint}",
            headers=headers,
            json=body
        )
        return response.json()

    def get_position(self, product_id: str) -> Dict:
        """
        Get current position information.
        
        Args:
            product_id (str): Product ID (e.g., 'SOL-PERP')
        """
        endpoint = f"/api/v3/brokerage/positions/{product_id}"
        headers = self._get_headers("GET", endpoint)
        response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
        return response.json()

    def get_account_summary(self) -> Dict:
        """Get perpetuals account summary including margin information."""
        endpoint = "/api/v3/brokerage/accounts"
        headers = self._get_headers("GET", endpoint)
        response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
        return response.json()

    def opt_in_multi_asset_collateral(self) -> Dict:
        """Opt-in to use multi-asset collateral (BTC and ETH)."""
        endpoint = "/api/v3/brokerage/multi_asset_collateral/opt_in"
        headers = self._get_headers("POST", endpoint)
        response = requests.post(f"{self.base_url}{endpoint}", headers=headers)
        return response.json()