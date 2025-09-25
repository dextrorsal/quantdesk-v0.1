"""
Bitget exchange client wrapper with common trading operations.
"""

import logging
from typing import Dict, List, Optional, Union
from decimal import Decimal

from ...exchanges.bitget.bitget.exceptions import BitgetAPIException
from ...exchanges.bitget.bitget.bitget_api import BitgetApi
from ..bitget.config import USDT_FUTURES

logger = logging.getLogger(__name__)


class BitgetClient:
    """
    A wrapper around the Bitget API client with common trading operations
    and error handling.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        testnet: bool = False,
        demo: bool = False,
        base_url: str = None
    ):
        """Initialize the Bitget client."""
        self.client = BitgetApi(
            api_key=api_key,
            api_secret_key=api_secret,
            passphrase=passphrase,
            use_server_time=True,
            first=False,
            demo=demo,
            base_url=base_url
        )

    def get_balance(
        self, coin: str = 'USDT', product_type: str = 'usdt-futures',
        margin_coin: str = None
    ) -> Decimal:
        """Get available balance for a specific coin and product type (demo/real, 
        futures/spot/margin)."""
        try:
            assets = self.client.get_account_assets(
                product_type=product_type, margin_coin=margin_coin or coin
            )
            for asset in assets.get('data', []):
                if asset.get('marginCoin') == (margin_coin or coin):
                    return Decimal(asset.get('available', '0'))
            return Decimal('0')
        except Exception as e:
            logger.error(
                f"Error getting balance for {coin}: {str(e)}"
            )
            raise

    def get_position(
        self, symbol: str = None, product_type: str = 'usdt-futures',
        margin_coin: str = None
    ) -> Optional[Dict]:
        """Get current position for a symbol (if symbol is provided) or all positions
        (if symbol is None)."""
        try:
            if symbol:
                positions = self.client.get_single_position(
                    symbol, product_type=product_type, margin_coin=margin_coin
                )
                if isinstance(positions, dict):
                    positions = positions.get('data', [])
                return positions[0] if positions else None
            else:
                positions = self.client.get_positions(
                    product_type=product_type, margin_coin=margin_coin
                )
                if isinstance(positions, dict):
                    positions = positions.get('data', [])
                return positions if positions else None
        except Exception as e:
            logger.error(
                f"Error getting position for {symbol}: {str(e)}"
            )
            raise

    def get_leverage(
        self,
        symbol: str,
        product_type: str = USDT_FUTURES
    ) -> int:
        """Get current leverage for a symbol."""
        try:
            position = self.get_position(symbol, product_type)
            return int(position['leverage']) if position else 1
        except BitgetAPIException as e:
            logger.error(f"Error getting leverage for {symbol}: {str(e)}")
            raise
    
    def set_leverage(self, symbol: str, leverage: int, product_type: str, margin_coin: str, margin_mode: str):
        """Set leverage for a symbol (demo/real, isolated/cross)."""
        return self.client.set_leverage(symbol, leverage, product_type, margin_coin, margin_mode)
    
    def set_margin_mode(self, symbol: str, margin_mode: str, product_type: str, margin_coin: str):
        """Set margin mode for a symbol (isolated/cross)."""
        return self.client.set_margin_mode(symbol, margin_mode, product_type, margin_coin)
    
    def set_asset_mode(self, asset_mode: str):
        """Switch asset mode (single/multi-asset) if supported."""
        return self.client.set_asset_mode(asset_mode)
    
    def place_order(self, symbol: str, side: str, order_type: str, size: Union[int, float], price: Optional[Union[int, float]] = None, product_type: str = None, margin_coin: str = None, margin_mode: str = None, leverage: int = None, asset_mode: str = None, reduce_only: bool = False, **kwargs) -> Dict:
        """Place an order with all relevant params (spot, margin, futures, demo/real)."""
        return self.client.place_order(symbol, side, order_type, size, price, product_type, margin_coin, margin_mode, leverage, asset_mode, reduce_only, **kwargs)
    
    def cancel_order(self, symbol: str, order_id: str, product_type: str = None, margin_coin: str = None) -> Dict:
        """Cancel an order (spot, margin, futures, demo/real)."""
        return self.client.cancel_order(symbol, order_id, product_type, margin_coin)
    
    def get_open_orders(
        self,
        symbol: str,
        product_type: str = USDT_FUTURES
    ) -> List[Dict]:
        """Get all open orders for a symbol."""
        try:
            return self.client.get_open_orders(
                symbol=symbol,
                productType=product_type
            )
        except BitgetAPIException as e:
            logger.error(f"Error getting open orders for {symbol}: {str(e)}")
            raise
    
    def get_order_history(
        self,
        symbol: str,
        product_type: str = USDT_FUTURES,
        limit: int = 100
    ) -> List[Dict]:
        """Get order history for a symbol."""
        try:
            return self.client.get_order_history(
                symbol=symbol,
                productType=product_type,
                limit=limit
            )
        except BitgetAPIException as e:
            logger.error(f"Error getting order history for {symbol}: {str(e)}")
            raise
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate for a symbol."""
        try:
            return self.client.get_funding_rate(symbol=symbol)
        except BitgetAPIException as e:
            logger.error(f"Error getting funding rate for {symbol}: {str(e)}")
            raise
    
    def get_mark_price(self, symbol: str) -> Decimal:
        """Get current mark price for a symbol."""
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return Decimal(ticker['markPrice'])
        except BitgetAPIException as e:
            logger.error(f"Error getting mark price for {symbol}: {str(e)}")
            raise
    
    def get_available_symbols(
        self,
        product_type: str = USDT_FUTURES
    ) -> List[Dict]:
        """Get all available trading symbols."""
        try:
            return self.client.get_contracts_info(productType=product_type)
        except BitgetAPIException as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            raise 