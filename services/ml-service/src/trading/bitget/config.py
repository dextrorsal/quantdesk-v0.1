"""
Bitget exchange configuration and constants.
"""

# API Base URLs
REST_URL = "https://api.bitget.com"
WS_URL = "wss://ws.bitget.com/spot/v1/stream"

# Product Types
USDT_FUTURES = "USDT-FUTURES"
COIN_FUTURES = "COIN-FUTURES"
SPOT = "SPOT"

# Trading Modes
ISOLATED = "isolated"
CROSS = "cross"

# Order Types
LIMIT = "limit"
MARKET = "market"
POST_ONLY = "post_only"
FOK = "fok"  # Fill or Kill
IOC = "ioc"  # Immediate or Cancel

# Time in Force
GTC = "gtc"  # Good Till Cancel
IOC = "ioc"  # Immediate or Cancel
FOK = "fok"  # Fill or Kill
POST_ONLY = "post_only"

# Position Sides
LONG = "long"
SHORT = "short"

# Order Sides
BUY = "buy"
SELL = "sell"

# Contract Size (in USD)
CONTRACT_SIZE = 1  # 1 USD per contract for USDT futures

# Leverage Limits
MIN_LEVERAGE = 1
MAX_LEVERAGE = 125  # Maximum leverage for most pairs

# Minimum Trade Amounts (in contracts)
MIN_TRADE_AMOUNT = 1

# Trading Fee Rates (in percentage)
MAKER_FEE_RATE = 0.02  # 0.02%
TAKER_FEE_RATE = 0.06  # 0.06%

# Funding Rate Intervals (in hours)
FUNDING_INTERVAL = 8

# Position Modes
HEDGE_MODE = "hedge"
ONE_WAY_MODE = "one_way" 