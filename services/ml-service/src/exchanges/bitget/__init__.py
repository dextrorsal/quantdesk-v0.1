"""
Bitget exchange integration package.
"""

from trading.bitget.config import (
    USDT_FUTURES,
    COIN_FUTURES,
    SPOT,
    LONG,
    SHORT,
    BUY,
    SELL,
    LIMIT,
    MARKET,
    POST_ONLY,
    FOK,
    IOC,
    ISOLATED,
    CROSS,
    HEDGE_MODE,
    ONE_WAY_MODE,
    CONTRACT_SIZE,
    MIN_LEVERAGE,
    MAX_LEVERAGE,
    MIN_TRADE_AMOUNT,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    FUNDING_INTERVAL,
)

from .bitget_handler import BitgetHandler

__all__ = [
    "USDT_FUTURES",
    "COIN_FUTURES",
    "SPOT",
    "LONG",
    "SHORT",
    "BUY",
    "SELL",
    "LIMIT",
    "MARKET",
    "POST_ONLY",
    "FOK",
    "IOC",
    "ISOLATED",
    "CROSS",
    "HEDGE_MODE",
    "ONE_WAY_MODE",
    "CONTRACT_SIZE",
    "MIN_LEVERAGE",
    "MAX_LEVERAGE",
    "MIN_TRADE_AMOUNT",
    "MAKER_FEE_RATE",
    "TAKER_FEE_RATE",
    "FUNDING_INTERVAL",
    "BitgetHandler",
]
