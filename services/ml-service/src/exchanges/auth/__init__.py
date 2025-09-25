"""
Authentication module for exchange API authentication.
Provides authentication handlers for various exchanges.
"""

from src.exchanges.auth.base_auth import BaseAuth
from src.exchanges.drift.auth import DriftAuth
from src.exchanges.auth.binance_auth import BinanceAuth
from src.exchanges.auth.coinbase_auth import CoinbaseAuth

__all__ = [
    'BaseAuth',
    'DriftAuth',
    'BinanceAuth',
    'CoinbaseAuth'
]