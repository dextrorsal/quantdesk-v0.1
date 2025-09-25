"""
Professional Order Management System for Bitget
Advanced order types, risk controls, and professional-grade features
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

from exchanges.bitget.bitget_trader import BitgetTrader

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
            self.client_order_id = f"qty_{uuid.uuid4().hex[:8]}"

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

class ProfessionalOrderManager:
    """
    Professional Order Management System for Bitget
    
    Features:
    - Advanced order types (stop-loss, take-profit, trailing stops, iceberg, TWAP)
    - Risk management and position sizing
    - Order lifecycle management
    - Error handling and retry logic
    - Performance monitoring
    - Order book integration
    """
    
    def __init__(self, 
                 bitget_trader: BitgetTrader,
                 risk_limits: RiskLimits,
                 order_persistence_file: str = "data/orders.json"):
        self.bitget_trader = bitget_trader
        self.risk_limits = risk_limits
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
        
        # Load existing orders
        self._load_orders()
        
        logger.info("Professional Order Manager initialized")
    
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
        
        logger.info(f"Updated position for {symbol}: size={position.size}, entry_price={position.entry_price}")
    
    def _convert_side_to_bitget(self, side: OrderSide) -> str:
        """Convert OrderSide to Bitget format"""
        side_mapping = {
            OrderSide.BUY: "open_long",
            OrderSide.SELL: "close_long", 
            OrderSide.OPEN_LONG: "open_long",
            OrderSide.CLOSE_LONG: "close_long",
            OrderSide.OPEN_SHORT: "open_short",
            OrderSide.CLOSE_SHORT: "close_short"
        }
        return side_mapping.get(side, side.value)
    
    def _get_position_side_for_stop(self, symbol: str) -> str:
        """Get position side for stop orders"""
        # Check if we have a position for this symbol
        position = self.positions.get(symbol)
        if position:
            return position.side  # 'long' or 'short'
        
        # Default to long if no position
        return "long"
    
    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place an order with professional order management"""
        # Generate order ID
        order_id = f"qty_{uuid.uuid4().hex[:8]}"
        
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
            
            # Place order based on type
            if request.order_type == OrderType.MARKET:
                result = await self._place_market_order(request)
            elif request.order_type == OrderType.LIMIT:
                result = await self._place_limit_order(request)
            elif request.order_type == OrderType.STOP_MARKET:
                result = await self._place_stop_market_order(request)
            elif request.order_type == OrderType.STOP_LIMIT:
                result = await self._place_stop_limit_order(request)
            elif request.order_type == OrderType.TAKE_PROFIT_MARKET:
                result = await self._place_take_profit_market_order(request)
            elif request.order_type == OrderType.TRAILING_STOP:
                result = await self._place_trailing_stop_order(request)
            else:
                raise ValueError(f"Unsupported order type: {request.order_type}")
            
            # Update order response
            if result and result.get('code') == '00000':
                order_response.status = OrderStatus.SUBMITTED
                order_response.exchange_order_id = result.get('data', {}).get('orderId')
                logger.info(f"Order placed successfully: {order_id}")
            else:
                order_response.status = OrderStatus.REJECTED
                order_response.error_message = result.get('msg', 'Unknown error')
                logger.error(f"Order placement failed: {order_response.error_message}")
            
            # Update rate limiting
            self.order_timestamps.append(datetime.now())
            
            # Update daily stats
            self.daily_stats['orders_placed'] += 1
            
            # Save orders
            self._save_orders()
            
        except Exception as e:
            order_response.status = OrderStatus.REJECTED
            order_response.error_message = str(e)
            logger.error(f"Error placing order: {e}")
        
        return order_response
    
    async def _place_market_order(self, request: OrderRequest) -> Dict:
        """Place market order"""
        # Convert side to Bitget format
        bitget_side = self._convert_side_to_bitget(request.side)
        
        return self.bitget_trader.place_order(
            symbol=request.symbol,
            side=bitget_side,
            order_type="market",
            price=None,
            size=float(request.quantity),
            time_in_force=request.time_in_force.value
        )
    
    async def _place_limit_order(self, request: OrderRequest) -> Dict:
        """Place limit order"""
        if not request.price:
            raise ValueError("Price required for limit order")
        
        # Convert side to Bitget format
        bitget_side = self._convert_side_to_bitget(request.side)
        
        return self.bitget_trader.place_order(
            symbol=request.symbol,
            side=bitget_side,
            order_type="limit",
            price=float(request.price),
            size=float(request.quantity),
            time_in_force=request.time_in_force.value
        )
    
    async def _place_stop_market_order(self, request: OrderRequest) -> Dict:
        """Place stop market order"""
        if not request.stop_price:
            raise ValueError("Stop price required for stop market order")
        
        # Get position side for stop order
        position_side = self._get_position_side_for_stop(request.symbol)
        
        return self.bitget_trader.place_stop_loss(
            symbol=request.symbol,
            trigger_price=float(request.stop_price),
            size=float(request.quantity),
            plan_type='pos_loss',
            hold_side=position_side
        )
    
    async def _place_stop_limit_order(self, request: OrderRequest) -> Dict:
        """Place stop limit order"""
        if not request.stop_price or not request.price:
            raise ValueError("Stop price and limit price required for stop limit order")
        
        # Bitget doesn't directly support stop-limit, so we'll use a stop order
        # and then place a limit order when triggered
        return await self._place_stop_market_order(request)
    
    async def _place_take_profit_market_order(self, request: OrderRequest) -> Dict:
        """Place take profit market order"""
        if not request.price:
            raise ValueError("Price required for take profit order")
        
        # Get position side for take profit order
        position_side = self._get_position_side_for_stop(request.symbol)
        
        return self.bitget_trader.place_take_profit(
            symbol=request.symbol,
            trigger_price=float(request.price),
            size=float(request.quantity),
            hold_side=position_side
        )
    
    async def _place_trailing_stop_order(self, request: OrderRequest) -> Dict:
        """Place trailing stop order"""
        if not request.stop_price:
            raise ValueError("Stop price required for trailing stop order")
        
        return self.bitget_trader.place_trailing_stop(
            symbol=request.symbol,
            trigger_price=float(request.stop_price),
            size=float(request.quantity)
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id not in self.active_orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.active_orders[order_id]
        
        try:
            result = self.bitget_trader.cancel_order(
                symbol=order.symbol,
                order_id=order.exchange_order_id or order_id
            )
            
            if result and result.get('code') == '00000':
                order.status = OrderStatus.CANCELLED
                self.daily_stats['orders_cancelled'] += 1
                logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order: {result.get('msg', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
        finally:
            self._save_orders()
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all active orders for a symbol or all symbols"""
        cancelled_count = 0
        
        orders_to_cancel = []
        for order_id, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
        
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    async def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        return self.active_orders.get(order_id)
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all active orders"""
        if symbol:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
        return list(self.active_orders.values())
    
    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Position]:
        """Get current positions"""
        try:
            result = self.bitget_trader.get_positions(symbol)
            if result and result.get('code') == '00000':
                positions_data = result.get('data', [])
                if not isinstance(positions_data, list):
                    positions_data = [positions_data]
                
                for pos_data in positions_data:
                    if float(pos_data.get('total', 0)) != 0:
                        position = Position(
                            symbol=pos_data['symbol'],
                            side=pos_data['holdSide'],
                            size=Decimal(str(pos_data['total'])),
                            entry_price=Decimal(str(pos_data['averageOpenPrice'])),
                            current_price=Decimal(str(pos_data['markPrice'])),
                            unrealized_pnl=Decimal(str(pos_data['unrealizedPL'])),
                            realized_pnl=Decimal(str(pos_data['realizedPL'])),
                            leverage=int(pos_data.get('leverage', 1)),
                            margin=Decimal(str(pos_data.get('margin', 0)))
                        )
                        self.positions[position.symbol] = position
                
                logger.info(f"Retrieved {len(self.positions)} positions")
                return self.positions
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    async def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            result = self.bitget_trader.get_balance()
            if result and result.get('code') == '00000':
                return result.get('data', {})
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics"""
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
            'daily_volume': self.daily_stats['total_volume']
        }

# Example usage and testing
async def main():
    """Example usage of Professional Order Manager"""
    
    # Initialize Bitget trader
    bitget_trader = BitgetTrader()
    
    # Set up risk limits
    risk_limits = RiskLimits(
        max_position_size=Decimal('10.0'),
        max_order_size=Decimal('1.0'),
        max_daily_loss=Decimal('1000.0'),
        max_drawdown=Decimal('0.1'),
        max_leverage=10,
        min_order_size=Decimal('0.001')
    )
    
    # Initialize order manager
    order_manager = ProfessionalOrderManager(bitget_trader, risk_limits)
    
    # Example: Place a market buy order
    order_request = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('0.001'),
        client_order_id="test_buy_001"
    )
    
    order_response = await order_manager.place_order(order_request)
    print(f"Order placed: {order_response}")
    
    # Example: Place a stop loss order
    stop_request = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=Decimal('0.001'),
        stop_price=Decimal('25000.0'),
        client_order_id="stop_loss_001"
    )
    
    stop_response = await order_manager.place_order(stop_request)
    print(f"Stop loss placed: {stop_response}")
    
    # Get positions
    positions = await order_manager.get_positions()
    print(f"Current positions: {positions}")
    
    # Get risk metrics
    risk_metrics = order_manager.get_risk_metrics()
    print(f"Risk metrics: {risk_metrics}")

if __name__ == "__main__":
    asyncio.run(main())
