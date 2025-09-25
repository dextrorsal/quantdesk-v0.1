"""
Exchange handlers package.
"""

import logging
from typing import Optional

from exchanges.base import BaseExchangeHandler
from exchanges.coinbase.coinbase import CoinbaseHandler
from exchanges.bitget import BitgetHandler
from exchanges.jupiter.jup import JupiterHandler

# Initialize logger
logger = logging.getLogger(__name__)

__all__ = [
    "BaseExchangeHandler",
    "CoinbaseHandler",
    "BitgetHandler",
    "JupiterHandler",
]

EXCHANGE_HANDLERS = {
    "coinbase": CoinbaseHandler,
    "bitget": BitgetHandler,
    "jupiter": JupiterHandler,
}


def get_exchange_handler(exchange_config) -> Optional[BaseExchangeHandler]:
    """
    Get the appropriate exchange handler instance based on the exchange config.

    Args:
        exchange_config (ExchangeConfig): Configuration for the exchange

    Returns:
        Optional[BaseExchangeHandler]: Exchange handler instance if found,
        None otherwise
    """
    exchange_map = {
        "coinbase": CoinbaseHandler,
        "jupiter": JupiterHandler,
        "bitget": BitgetHandler,
        "bitget_demo": BitgetHandler,  # Demo environment support
    }

    handler_class = exchange_map.get(exchange_config.name.lower())
    if handler_class is None:
        logger.error(
            f"No handler found for exchange {exchange_config.name}"
        )
        return None

    try:
        return handler_class(exchange_config)
    except Exception as e:
        logger.error(
            f"Error initializing handler for {exchange_config.name}: {e}"
        )
        return None


"""Exchange implementations."""
