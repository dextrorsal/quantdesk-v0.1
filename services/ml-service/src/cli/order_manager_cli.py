#!/usr/bin/env python3
"""
Professional Order Management CLI
Command-line interface for the professional order management system
"""

import asyncio
import argparse
import json
import sys
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from trading.order_management import (
    ProfessionalOrderManager, 
    OrderRequest, 
    OrderType, 
    OrderSide, 
    TimeInForce,
    RiskLimits
)
from exchanges.bitget.bitget_trader import BitgetTrader

class OrderManagerCLI:
    """Command-line interface for order management"""
    
    def __init__(self):
        self.order_manager: Optional[ProfessionalOrderManager] = None
        self.bitget_trader: Optional[BitgetTrader] = None
        
    async def initialize(self):
        """Initialize the order manager"""
        try:
            print("🔧 Initializing Professional Order Management System...")
            
            # Initialize Bitget trader
            self.bitget_trader = BitgetTrader()
            
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
            self.order_manager = ProfessionalOrderManager(
                self.bitget_trader, 
                risk_limits,
                order_persistence_file="data/orders.json"
            )
            
            print("✅ Order Management System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    async def place_order(self, args):
        """Place an order"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            # Parse order parameters
            side = OrderSide(args.side.lower())
            order_type = OrderType(args.type.lower())
            quantity = Decimal(str(args.quantity))
            price = Decimal(str(args.price)) if args.price else None
            stop_price = Decimal(str(args.stop_price)) if args.stop_price else None
            
            # Create order request
            order_request = OrderRequest(
                symbol=args.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
                client_order_id=args.client_id
            )
            
            print(f"📤 Placing {order_type.value} {side.value} order for {quantity} {args.symbol}...")
            
            # Place order
            order_response = await self.order_manager.place_order(order_request)
            
            # Display result
            if order_response.status.value == "submitted":
                print(f"✅ Order placed successfully!")
                print(f"   Order ID: {order_response.order_id}")
                print(f"   Exchange ID: {order_response.exchange_order_id}")
                print(f"   Status: {order_response.status.value}")
            else:
                print(f"❌ Order failed: {order_response.error_message}")
                
        except Exception as e:
            print(f"❌ Error placing order: {e}")
    
    async def cancel_order(self, args):
        """Cancel an order"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            print(f"🚫 Cancelling order {args.order_id}...")
            
            success = await self.order_manager.cancel_order(args.order_id)
            
            if success:
                print("✅ Order cancelled successfully!")
            else:
                print("❌ Failed to cancel order")
                
        except Exception as e:
            print(f"❌ Error cancelling order: {e}")
    
    async def cancel_all_orders(self, args):
        """Cancel all orders"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            symbol = args.symbol if hasattr(args, 'symbol') else None
            print(f"🚫 Cancelling all orders{' for ' + symbol if symbol else ''}...")
            
            cancelled_count = await self.order_manager.cancel_all_orders(symbol)
            print(f"✅ Cancelled {cancelled_count} orders")
            
        except Exception as e:
            print(f"❌ Error cancelling orders: {e}")
    
    async def list_orders(self, args):
        """List active orders"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            symbol = args.symbol if hasattr(args, 'symbol') else None
            orders = await self.order_manager.get_active_orders(symbol)
            
            if not orders:
                print("📋 No active orders found")
                return
            
            print(f"📋 Active Orders ({len(orders)}):")
            print("-" * 80)
            print(f"{'Order ID':<12} {'Symbol':<10} {'Side':<8} {'Type':<12} {'Quantity':<10} {'Price':<10} {'Status':<12}")
            print("-" * 80)
            
            for order in orders:
                price_str = f"{order.price:.4f}" if order.price else "Market"
                print(f"{order.order_id:<12} {order.symbol:<10} {order.side.value:<8} {order.order_type.value:<12} {order.quantity:<10} {price_str:<10} {order.status.value:<12}")
                
        except Exception as e:
            print(f"❌ Error listing orders: {e}")
    
    async def get_positions(self, args):
        """Get current positions"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            symbol = args.symbol if hasattr(args, 'symbol') else None
            positions = await self.order_manager.get_positions(symbol)
            
            if not positions:
                print("📊 No open positions found")
                return
            
            print(f"📊 Open Positions ({len(positions)}):")
            print("-" * 100)
            print(f"{'Symbol':<10} {'Side':<6} {'Size':<12} {'Entry Price':<12} {'Current Price':<14} {'PnL':<12} {'Leverage':<8}")
            print("-" * 100)
            
            total_pnl = Decimal('0')
            for position in positions.values():
                pnl_str = f"{position.unrealized_pnl:+.4f}"
                total_pnl += position.unrealized_pnl
                print(f"{position.symbol:<10} {position.side:<6} {position.size:<12} {position.entry_price:<12} {position.current_price:<14} {pnl_str:<12} {position.leverage:<8}")
            
            print("-" * 100)
            print(f"Total Unrealized PnL: {total_pnl:+.4f}")
            
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
    
    async def get_balance(self, args):
        """Get account balance"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            balance = await self.order_manager.get_account_balance()
            
            if not balance:
                print("💰 No balance information available")
                return
            
            print("💰 Account Balance:")
            print("-" * 50)
            
            if isinstance(balance, list):
                for account in balance:
                    print(f"Account: {account.get('marginCoin', 'Unknown')}")
                    print(f"  Available: {account.get('available', '0')}")
                    print(f"  Frozen: {account.get('frozen', '0')}")
                    print(f"  Total: {account.get('total', '0')}")
                    print()
            else:
                print(f"Available: {balance.get('available', '0')}")
                print(f"Frozen: {balance.get('frozen', '0')}")
                print(f"Total: {balance.get('total', '0')}")
                
        except Exception as e:
            print(f"❌ Error getting balance: {e}")
    
    async def get_stats(self, args):
        """Get trading statistics"""
        if not self.order_manager:
            print("❌ Order manager not initialized")
            return
        
        try:
            daily_stats = self.order_manager.get_daily_stats()
            risk_metrics = self.order_manager.get_risk_metrics()
            
            print("📈 Trading Statistics:")
            print("-" * 50)
            print(f"Orders Placed Today: {daily_stats['orders_placed']}")
            print(f"Orders Filled Today: {daily_stats['orders_filled']}")
            print(f"Orders Cancelled Today: {daily_stats['orders_cancelled']}")
            print(f"Total Volume Today: {daily_stats['total_volume']}")
            print(f"Total Fees Today: {daily_stats['total_fees']}")
            print(f"PnL Today: {daily_stats['pnl']}")
            print()
            print("⚠️  Risk Metrics:")
            print("-" * 50)
            print(f"Total Exposure: {risk_metrics['total_exposure']}")
            print(f"Total Unrealized PnL: {risk_metrics['total_unrealized_pnl']}")
            print(f"Active Positions: {risk_metrics['active_positions']}")
            print(f"Active Orders: {risk_metrics['active_orders']}")
            print(f"Daily Orders: {risk_metrics['daily_orders']}")
            print(f"Daily Volume: {risk_metrics['daily_volume']}")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
    async def set_leverage(self, args):
        """Set leverage for a symbol"""
        if not self.bitget_trader:
            print("❌ Bitget trader not initialized")
            return
        
        try:
            print(f"⚙️  Setting leverage for {args.symbol} to {args.leverage}x...")
            
            result = self.bitget_trader.set_leverage(
                symbol=args.symbol,
                leverage=args.leverage,
                hold_side=args.side.lower()
            )
            
            if result and result.get('code') == '00000':
                print("✅ Leverage set successfully!")
            else:
                print(f"❌ Failed to set leverage: {result.get('msg', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error setting leverage: {e}")

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Professional Order Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Place a market buy order
  python order_manager_cli.py place --symbol BTCUSDT --side buy --type market --quantity 0.001
  
  # Place a limit sell order
  python order_manager_cli.py place --symbol BTCUSDT --side sell --type limit --quantity 0.001 --price 50000
  
  # Place a stop loss order
  python order_manager_cli.py place --symbol BTCUSDT --side sell --type stop_market --quantity 0.001 --stop-price 45000
  
  # Cancel an order
  python order_manager_cli.py cancel --order-id qty_12345678
  
  # List all active orders
  python order_manager_cli.py list
  
  # Get positions
  python order_manager_cli.py positions
  
  # Get account balance
  python order_manager_cli.py balance
  
  # Get trading statistics
  python order_manager_cli.py stats
  
  # Set leverage
  python order_manager_cli.py leverage --symbol BTCUSDT --leverage 5 --side long
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Place order command
    place_parser = subparsers.add_parser('place', help='Place an order')
    place_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    place_parser.add_argument('--side', required=True, choices=['buy', 'sell', 'open_long', 'close_long', 'open_short', 'close_short'], help='Order side')
    place_parser.add_argument('--type', required=True, choices=['market', 'limit', 'stop_market', 'stop_limit', 'take_profit_market', 'trailing_stop'], help='Order type')
    place_parser.add_argument('--quantity', required=True, type=float, help='Order quantity')
    place_parser.add_argument('--price', type=float, help='Order price (required for limit orders)')
    place_parser.add_argument('--stop-price', type=float, help='Stop price (required for stop orders)')
    place_parser.add_argument('--client-id', help='Client order ID')
    
    # Cancel order command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel an order')
    cancel_parser.add_argument('--order-id', required=True, help='Order ID to cancel')
    
    # Cancel all orders command
    cancel_all_parser = subparsers.add_parser('cancel-all', help='Cancel all orders')
    cancel_all_parser.add_argument('--symbol', help='Symbol to cancel orders for (optional)')
    
    # List orders command
    list_parser = subparsers.add_parser('list', help='List active orders')
    list_parser.add_argument('--symbol', help='Symbol to filter by (optional)')
    
    # Positions command
    positions_parser = subparsers.add_parser('positions', help='Get current positions')
    positions_parser.add_argument('--symbol', help='Symbol to filter by (optional)')
    
    # Balance command
    balance_parser = subparsers.add_parser('balance', help='Get account balance')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get trading statistics')
    
    # Leverage command
    leverage_parser = subparsers.add_parser('leverage', help='Set leverage for a symbol')
    leverage_parser.add_argument('--symbol', required=True, help='Trading symbol')
    leverage_parser.add_argument('--leverage', required=True, type=int, help='Leverage amount')
    leverage_parser.add_argument('--side', required=True, choices=['long', 'short'], help='Position side')
    
    return parser

async def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = OrderManagerCLI()
    
    if not await cli.initialize():
        return
    
    # Execute command
    if args.command == 'place':
        await cli.place_order(args)
    elif args.command == 'cancel':
        await cli.cancel_order(args)
    elif args.command == 'cancel-all':
        await cli.cancel_all_orders(args)
    elif args.command == 'list':
        await cli.list_orders(args)
    elif args.command == 'positions':
        await cli.get_positions(args)
    elif args.command == 'balance':
        await cli.get_balance(args)
    elif args.command == 'stats':
        await cli.get_stats(args)
    elif args.command == 'leverage':
        await cli.set_leverage(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
