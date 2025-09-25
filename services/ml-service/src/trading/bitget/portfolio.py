import logging
from decimal import Decimal
from typing import Dict, List, Optional
from .client import BitgetClient

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Advanced portfolio manager for Bitget trading bots.
    Tracks positions, balances, PnL, and manages risk and order execution.
    """
    def __init__(self, client: BitgetClient, leverage: int = 20, risk_per_trade: float = 0.01, symbols: Optional[List[str]] = None):
        self.client = client
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade  # e.g., 0.01 = 1% of balance per trade
        self.symbols = symbols or ["SOLUSDT"]
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.trade_history: List[Dict] = []
        self.balance: Decimal = Decimal('0')
        self.update_balance()

    def update_balance(self, coin: str = 'USDT'):
        self.balance = self.client.get_balance(coin)
        logger.info(f"Updated balance: {self.balance} {coin}")

    def sync_positions(self):
        for symbol in self.symbols:
            pos = self.client.get_position(symbol)
            self.positions[symbol] = pos
        logger.info(f"Synced positions: {self.positions}")

    def calculate_position_size(self, symbol: str, stop_loss_pct: float = 0.01, method: str = 'fixed') -> Decimal:
        # Fixed % of balance per trade, adjusted for leverage
        if method == 'fixed':
            risk_amount = self.balance * Decimal(str(self.risk_per_trade))
            position_size = risk_amount * self.leverage / Decimal(str(stop_loss_pct))
            return position_size
        # Kelly or other methods can be added here
        else:
            raise NotImplementedError(f"Position sizing method '{method}' not implemented.")

    def open_position(self, symbol: str, side: str, size: Decimal, order_type: str = 'market', price: Optional[Decimal] = None, stop_loss: Optional[Decimal] = None, take_profit: Optional[Decimal] = None):
        # Place order via BitgetClient
        order = self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            product_type='USDT-FUTURES',
            reduce_only=False
        )
        logger.info(f"Opened position: {order}")
        # Track in local state
        self.sync_positions()
        self.log_trade(order, action='open')
        return order

    def close_position(self, symbol: str, size: Optional[Decimal] = None, order_type: str = 'market'):
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"No open position to close for {symbol}")
            return None
        side = 'sell' if pos['side'] == 'long' else 'buy'
        close_size = size or pos['size']
        order = self.client.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=close_size,
            product_type='USDT-FUTURES',
            reduce_only=True
        )
        logger.info(f"Closed position: {order}")
        self.sync_positions()
        self.log_trade(order, action='close')
        return order

    def log_trade(self, order: Dict, action: str = 'open'):
        # Log to local history
        self.trade_history.append({'action': action, 'order': order})
        # TODO: Log to Supabase (stub)
        logger.info(f"Logged trade ({action}): {order}")

    def get_pnl(self, symbol: str) -> Optional[Decimal]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        # Example: calculate PnL from position info (depends on Bitget API response structure)
        entry_price = Decimal(pos.get('entryPrice', '0'))
        current_price = self.client.get_mark_price(symbol)
        size = Decimal(pos.get('size', '0'))
        if pos['side'] == 'long':
            pnl = (current_price - entry_price) * size
        else:
            pnl = (entry_price - current_price) * size
        return pnl

    def summary(self):
        return {
            'balance': str(self.balance),
            'positions': self.positions,
            'trade_history': self.trade_history[-10:],  # last 10 trades
        } 