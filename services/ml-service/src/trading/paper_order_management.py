"""
Paper Trading Order Management System
Safe testing environment that simulates orders without real money
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import uuid
import json
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"
    OPEN_LONG = "open_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT = "open_short"
    CLOSE_SHORT = "close_short"

class TimeInForce(Enum):
    """Time in force enumeration"""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # Day order
    NORMAL = "normal"  # Bitget specific

@dataclass
class OrderRequest:
    """Order request data structure"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"paper_{uuid.uuid4().hex[:8]}"

@dataclass
class OrderResponse:
    """Order response data structure"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None
    fees: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size: Decimal
    max_order_size: Decimal
    max_daily_loss: Decimal
    max_drawdown: Decimal
    max_leverage: int = 10
    min_order_size: Decimal = Decimal('0.001')
    max_orders_per_minute: int = 60
    max_orders_per_day: int = 1000

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # 'long' or 'short'
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: int
    margin: Decimal
    timestamp: datetime = field(default_factory=datetime.now)

class PaperOrderManager:
    """
    Paper Trading Order Management System
    
    Features:
    - Simulates all order types without real money
    - Risk management and position sizing
    - Order lifecycle management
    - Performance monitoring
    - Safe testing environment
    """
    
    def __init__(self, 
                 risk_limits: RiskLimits,
                 initial_balance: Decimal = Decimal('10000'),
                 order_persistence_file: str = "data/paper_orders.json"):
        self.risk_limits = risk_limits
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.order_persistence_file = Path(order_persistence_file)
        
        # Order tracking
        self.active_orders: Dict[str, OrderResponse] = {}
        self.order_history: List[OrderResponse] = []
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.daily_stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_volume': Decimal('0'),
            'total_fees': Decimal('0'),
            'pnl': Decimal('0')
        }
        
        # Rate limiting
        self.order_timestamps: List[datetime] = []
        
        # Mock market data
        self.mock_prices = {
            'BTCUSDT': Decimal('45000'),
            'ETHUSDT': Decimal('3000'),
            'SOLUSDT': Decimal('100'),
            'ADAUSDT': Decimal('0.5'),
            'DOTUSDT': Decimal('7'),
            'AVAXUSDT': Decimal('25'),
            'MATICUSDT': Decimal('0.8'),
            'LINKUSDT': Decimal('15'),
            'UNIUSDT': Decimal('6'),
            'ATOMUSDT': Decimal('10')
        }
        
        # Load existing orders
        self._load_orders()
        
        logger.info("Paper Order Manager initialized")
    
    def _load_orders(self):
        """Load orders from persistence file"""
        if self.order_persistence_file.exists():
            try:
                with open(self.order_persistence_file, 'r') as f:
                    data = json.load(f)
                    # Restore active orders and history
                    for order_data in data.get('active_orders', []):
                        order = self._dict_to_order_response(order_data)
                        self.active_orders[order.order_id] = order
                    for order_data in data.get('order_history', []):
                        order = self._dict_to_order_response(order_data)
                        self.order_history.append(order)
                logger.info(f"Loaded {len(self.active_orders)} active orders and {len(self.order_history)} historical orders")
            except Exception as e:
                logger.error(f"Error loading orders: {e}")
    
    def _save_orders(self):
        """Save orders to persistence file"""
        try:
            data = {
                'active_orders': [self._order_response_to_dict(order) for order in self.active_orders.values()],
                'order_history': [self._order_response_to_dict(order) for order in self.order_history[-1000:]]  # Keep last 1000
            }
            self.order_persistence_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.order_persistence_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving orders: {e}")
    
    def _dict_to_order_response(self, data: Dict) -> OrderResponse:
        """Convert dictionary to OrderResponse"""
        return OrderResponse(
            order_id=data['order_id'],
            client_order_id=data['client_order_id'],
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['order_type']),
            quantity=Decimal(str(data['quantity'])),
            price=Decimal(str(data['price'])) if data['price'] else None,
            stop_price=Decimal(str(data['stop_price'])) if data['stop_price'] else None,
            status=OrderStatus(data['status']),
            filled_quantity=Decimal(str(data['filled_quantity'])),
            remaining_quantity=Decimal(str(data['remaining_quantity'])),
            average_price=Decimal(str(data['average_price'])) if data['average_price'] else None,
            fees=Decimal(str(data['fees'])),
            timestamp=datetime.fromisoformat(data['timestamp']),
            exchange_order_id=data.get('exchange_order_id'),
            error_message=data.get('error_message')
        )
    
    def _order_response_to_dict(self, order: OrderResponse) -> Dict:
        """Convert OrderResponse to dictionary"""
        return {
            'order_id': order.order_id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': str(order.quantity),
            'price': str(order.price) if order.price else None,
            'stop_price': str(order.stop_price) if order.stop_price else None,
            'status': order.status.value,
            'filled_quantity': str(order.filled_quantity),
            'remaining_quantity': str(order.remaining_quantity),
            'average_price': str(order.average_price) if order.average_price else None,
            'fees': str(order.fees),
            'timestamp': order.timestamp.isoformat(),
            'exchange_order_id': order.exchange_order_id,
            'error_message': order.error_message
        }
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Remove old timestamps
        self.order_timestamps = [ts for ts in self.order_timestamps if now - ts < timedelta(minutes=1)]
        
        # Check per-minute limit
        if len(self.order_timestamps) >= self.risk_limits.max_orders_per_minute:
            logger.warning(f"Rate limit exceeded: {len(self.order_timestamps)} orders in last minute")
            return False
        
        # Check daily limit
        today_orders = [ts for ts in self.order_timestamps if now.date() == ts.date()]
        if len(today_orders) >= self.risk_limits.max_orders_per_day:
            logger.warning(f"Daily order limit exceeded: {len(today_orders)} orders today")
            return False
        
        return True
    
    def _validate_order_request(self, request: OrderRequest) -> bool:
        """Validate order request against risk limits"""
        # Check order size
        if request.quantity < self.risk_limits.min_order_size:
            logger.error(f"Order size too small: {request.quantity} < {self.risk_limits.min_order_size}")
            return False
        
        if request.quantity > self.risk_limits.max_order_size:
            logger.error(f"Order size too large: {request.quantity} > {self.risk_limits.max_order_size}")
            return False
        
        # Check position size limits
        current_position = self.positions.get(request.symbol)
        if current_position:
            new_size = current_position.size + (request.quantity if request.side in [OrderSide.BUY, OrderSide.OPEN_LONG] else -request.quantity)
            if abs(new_size) > self.risk_limits.max_position_size:
                logger.error(f"Position size limit exceeded: {new_size} > {self.risk_limits.max_position_size}")
                return False
        
        return True
    
    def _simulate_market_price(self, symbol: str) -> Decimal:
        """Simulate market price with some randomness"""
        base_price = self.mock_prices.get(symbol, Decimal('100'))
        # Add ±2% randomness
        variation = random.uniform(-0.02, 0.02)
        return base_price * (1 + Decimal(str(variation)))
    
    def _simulate_order_fill(self, order: OrderResponse) -> bool:
        """Simulate order fill based on order type"""
        current_price = self._simulate_market_price(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            # Market orders always fill
            order.filled_quantity = order.quantity
            order.remaining_quantity = Decimal('0')
            order.average_price = current_price
            order.status = OrderStatus.FILLED
            return True
            
        elif order.order_type == OrderType.LIMIT:
            if order.price:
                # Limit orders fill if price is favorable
                if order.side in [OrderSide.BUY, OrderSide.OPEN_LONG]:
                    # Buy limit fills if current price <= limit price
                    if current_price <= order.price:
                        order.filled_quantity = order.quantity
                        order.remaining_quantity = Decimal('0')
                        order.average_price = order.price
                        order.status = OrderStatus.FILLED
                        return True
                else:
                    # Sell limit fills if current price >= limit price
                    if current_price >= order.price:
                        order.filled_quantity = order.quantity
                        order.remaining_quantity = Decimal('0')
                        order.average_price = order.price
                        order.status = OrderStatus.FILLED
                        return True
            
        elif order.order_type == OrderType.STOP_MARKET:
            if order.stop_price:
                # Stop orders fill if price crosses stop level
                if order.side in [OrderSide.BUY, OrderSide.OPEN_LONG]:
                    # Buy stop fills if current price >= stop price
                    if current_price >= order.stop_price:
                        order.filled_quantity = order.quantity
                        order.remaining_quantity = Decimal('0')
                        order.average_price = current_price
                        order.status = OrderStatus.FILLED
                        return True
                else:
                    # Sell stop fills if current price <= stop price
                    if current_price <= order.stop_price:
                        order.filled_quantity = order.quantity
                        order.remaining_quantity = Decimal('0')
                        order.average_price = current_price
                        order.status = OrderStatus.FILLED
                        return True
        
        # Order remains unfilled
        order.status = OrderStatus.SUBMITTED
        return False
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal):
        """Update position after order fill"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                side='long' if side in [OrderSide.BUY, OrderSide.OPEN_LONG] else 'short',
                size=Decimal('0'),
                entry_price=price,
                current_price=price,
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0'),
                leverage=1,
                margin=Decimal('0')
            )
        
        position = self.positions[symbol]
        
        # Update position size
        if side in [OrderSide.BUY, OrderSide.OPEN_LONG]:
            position.size += quantity
        else:
            position.size -= quantity
        
        # Update entry price (volume-weighted average)
        if position.size != 0:
            total_value = position.entry_price * (position.size - quantity) + price * quantity
            position.entry_price = total_value / position.size
        
        # Update current price
        position.current_price = price
        
        logger.info(f"Updated paper position for {symbol}: size={position.size}, entry_price={position.entry_price}")
    
    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place a paper order"""
        # Generate order ID
        order_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        # Create order response
        order_response = OrderResponse(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            status=OrderStatus.PENDING,
            remaining_quantity=request.quantity
        )
        
        try:
            # Check rate limits
            if not self._check_rate_limits():
                order_response.status = OrderStatus.REJECTED
                order_response.error_message = "Rate limit exceeded"
                return order_response
            
            # Validate order
            if not self._validate_order_request(request):
                order_response.status = OrderStatus.REJECTED
                order_response.error_message = "Order validation failed"
                return order_response
            
            # Add to active orders
            self.active_orders[order_id] = order_response
            
            # Simulate order processing
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Try to fill the order
            filled = self._simulate_order_fill(order_response)
            
            if filled:
                # Update position
                self._update_position(
                    request.symbol, 
                    request.side, 
                    request.quantity, 
                    order_response.average_price or Decimal('0')
                )
                
                # Calculate fees (0.1% for paper trading)
                order_response.fees = request.quantity * (order_response.average_price or Decimal('0')) * Decimal('0.001')
                
                # Update daily stats
                self.daily_stats['orders_filled'] += 1
                self.daily_stats['total_volume'] += request.quantity * (order_response.average_price or Decimal('0'))
                self.daily_stats['total_fees'] += order_response.fees
                
                logger.info(f"Paper order filled: {order_id}")
            else:
                logger.info(f"Paper order submitted: {order_id}")
            
            # Update rate limiting
            self.order_timestamps.append(datetime.now())
            
            # Update daily stats
            self.daily_stats['orders_placed'] += 1
            
            # Save orders
            self._save_orders()
            
        except Exception as e:
            order_response.status = OrderStatus.REJECTED
            order_response.error_message = str(e)
            logger.error(f"Error placing paper order: {e}")
        
        return order_response
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order"""
        if order_id not in self.active_orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.active_orders[order_id]
        
        try:
            order.status = OrderStatus.CANCELLED
            self.daily_stats['orders_cancelled'] += 1
            logger.info(f"Paper order cancelled: {order_id}")
            return True
                
        except Exception as e:
            logger.error(f"Error cancelling paper order: {e}")
            return False
        finally:
            self._save_orders()
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all paper orders for a symbol or all symbols"""
        cancelled_count = 0
        
        orders_to_cancel = []
        for order_id, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
        
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} paper orders")
        return cancelled_count
    
    async def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Get paper order status"""
        return self.active_orders.get(order_id)
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all active paper orders"""
        if symbol:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
        return list(self.active_orders.values())
    
    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Position]:
        """Get current paper positions"""
        if symbol:
            return {k: v for k, v in self.positions.items() if k == symbol}
        return self.positions.copy()
    
    async def get_account_balance(self) -> Dict:
        """Get paper account balance"""
        return {
            'available': str(self.current_balance),
            'total': str(self.current_balance),
            'unrealized_pnl': str(sum(pos.unrealized_pnl for pos in self.positions.values())),
            'realized_pnl': str(sum(pos.realized_pnl for pos in self.positions.values()))
        }
    
    def get_daily_stats(self) -> Dict:
        """Get daily paper trading statistics"""
        return self.daily_stats.copy()
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_pnl,
            'active_positions': len(self.positions),
            'active_orders': len(self.active_orders),
            'daily_orders': self.daily_stats['orders_placed'],
            'daily_volume': self.daily_stats['total_volume'],
            'current_balance': self.current_balance
        }

# Example usage and testing
async def main():
    """Example usage of Paper Order Manager"""
    
    # Set up risk limits
    risk_limits = RiskLimits(
        max_position_size=Decimal('10.0'),
        max_order_size=Decimal('1.0'),
        max_daily_loss=Decimal('1000.0'),
        max_drawdown=Decimal('0.1'),
        max_leverage=10,
        min_order_size=Decimal('0.001')
    )
    
    # Initialize paper order manager
    order_manager = PaperOrderManager(risk_limits)
    
    # Example: Place a market buy order
    order_request = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('0.001'),
        client_order_id="paper_test_001"
    )
    
    order_response = await order_manager.place_order(order_request)
    print(f"Paper order placed: {order_response}")
    
    # Get positions
    positions = await order_manager.get_positions()
    print(f"Paper positions: {positions}")
    
    # Get risk metrics
    risk_metrics = order_manager.get_risk_metrics()
    print(f"Paper risk metrics: {risk_metrics}")

if __name__ == "__main__":
    asyncio.run(main())
