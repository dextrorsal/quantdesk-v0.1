"""
High-Leverage Trading Manager

Manages 25x leverage positions with cross margin and hedging capabilities.
Designed for aggressive meme coin trading with $1,000 starting balance.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

@dataclass
class LeverageConfig:
    """Configuration for leverage management"""
    starting_balance: float = 1000.0      # Starting account balance
    base_position_size: float = 0.10      # 10% of account per trade
    max_leverage: float = 25.0            # Maximum 25x leverage
    max_total_exposure: float = 0.60      # Max 60% of account leveraged
    max_correlation_exposure: float = 0.30  # Max 30% in correlated pairs
    per_trade_risk: float = 0.02          # 2% risk per trade
    daily_loss_limit: float = 0.05        # 5% daily loss limit
    max_drawdown: float = 0.10            # 10% max drawdown
    leverage_reduction_factor: float = 0.8  # Reduce leverage on losses

class LeverageManager:
    """Manages high-leverage trading positions with risk controls."""
    
    def __init__(self, config: Optional[LeverageConfig] = None):
        self.config = config or LeverageConfig()
        self.logger = logging.getLogger(__name__)
        
        # Account state
        self.current_balance = self.config.starting_balance
        self.max_equity = self.config.starting_balance
        self.daily_pnl = 0.0
        self.daily_start_balance = self.config.starting_balance
        
        # Position tracking
        self.positions = {}  # symbol -> position details
        self.position_history = []
        
        # Risk metrics
        self.current_drawdown = 0.0
        self.total_exposure = 0.0
        self.correlation_matrix = {}
        
        # Performance tracking
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Leverage Manager using device: {self.device}")
    
    def calculate_position_size(self, symbol: str, price: float, 
                              signal_strength: float) -> Tuple[float, float]:
        """
        Calculate optimal position size and leverage.
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal_strength: Signal confidence (0-1)
            
        Returns:
            Tuple of (position_size, leverage_used)
        """
        # Base position value (10% of account)
        base_position_value = self.current_balance * self.config.base_position_size
        
        # Adjust for signal strength
        adjusted_value = base_position_value * signal_strength
        
        # Calculate leverage needed
        leverage_needed = adjusted_value / (self.current_balance * self.config.base_position_size)
        leverage_needed = min(leverage_needed, self.config.max_leverage)
        
        # Apply leverage reduction if in drawdown
        if self.current_drawdown > 0.05:  # 5% drawdown
            leverage_needed *= self.config.leverage_reduction_factor
        
        # Calculate final position size
        position_size = (adjusted_value * leverage_needed) / price
        
        return position_size, leverage_needed
    
    def check_risk_limits(self, new_position_value: float, 
                         symbol: str) -> Tuple[bool, str]:
        """
        Check if new position violates risk limits.
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check daily loss limit
        if self.daily_pnl < -(self.current_balance * self.config.daily_loss_limit):
            return False, "Daily loss limit exceeded"
        
        # Check drawdown limit
        if self.current_drawdown > self.config.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        # Check total exposure limit
        total_exposure_with_new = self.total_exposure + new_position_value
        if total_exposure_with_new > (self.current_balance * self.config.max_total_exposure):
            return False, "Total exposure limit exceeded"
        
        # Check correlation limits
        correlated_exposure = self.calculate_correlated_exposure(symbol)
        if correlated_exposure > (self.current_balance * self.config.max_correlation_exposure):
            return False, "Correlation exposure limit exceeded"
        
        return True, "Position allowed"
    
    def calculate_correlated_exposure(self, symbol: str) -> float:
        """Calculate exposure to correlated pairs."""
        # Simple correlation groups for meme coins
        meme_coins = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        if symbol in meme_coins:
            # Sum exposure to all meme coins
            correlated_exposure = 0.0
            for pos_symbol, position in self.positions.items():
                if pos_symbol in meme_coins:
                    correlated_exposure += position['value']
            return correlated_exposure
        
        return 0.0
    
    def open_position(self, symbol: str, side: str, price: float, 
                     signal_strength: float) -> Dict:
        """
        Open a new leveraged position.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            price: Entry price
            signal_strength: Signal confidence
            
        Returns:
            Position details
        """
        # Calculate position size
        position_size, leverage = self.calculate_position_size(
            symbol, price, signal_strength
        )
        
        # Calculate position value
        position_value = position_size * price
        
        # Check risk limits
        allowed, reason = self.check_risk_limits(position_value, symbol)
        
        if not allowed:
            self.logger.warning(f"Position rejected: {reason}")
            return {
                'success': False,
                'reason': reason,
                'position_size': 0.0,
                'leverage': 0.0
            }
        
        # Create position
        position = {
            'symbol': symbol,
            'side': side,
            'size': position_size,
            'entry_price': price,
            'leverage': leverage,
            'value': position_value,
            'entry_time': datetime.now(),
            'signal_strength': signal_strength
        }
        
        # Update tracking
        self.positions[symbol] = position
        self.total_exposure += position_value
        
        self.logger.info(
            f"Opened {side} position: {symbol} - "
            f"Size: {position_size:.4f}, Leverage: {leverage:.1f}x, "
            f"Value: ${position_value:.2f}"
        )
        
        return {
            'success': True,
            'position': position
        }
    
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """
        Close an existing position.
        
        Returns:
            Trade result details
        """
        if symbol not in self.positions:
            return {
                'success': False,
                'reason': 'Position not found'
            }
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:  # short
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        # Calculate return percentage
        return_pct = pnl / position['value']
        
        # Update account
        self.current_balance += pnl
        self.daily_pnl += pnl
        self.total_exposure -= position['value']
        
        # Update max equity and drawdown
        self.max_equity = max(self.max_equity, self.current_balance)
        self.current_drawdown = (self.max_equity - self.current_balance) / self.max_equity
        
        # Update trade statistics
        self.trades_today += 1
        if pnl > 0:
            self.wins_today += 1
        else:
            self.losses_today += 1
        
        # Store trade history
        trade_record = {
            'symbol': symbol,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'leverage': position['leverage'],
            'pnl': pnl,
            'return_pct': return_pct,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'signal_strength': position['signal_strength']
        }
        self.position_history.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.info(
            f"Closed {position['side']} position: {symbol} - "
            f"PnL: ${pnl:.2f} ({return_pct:.2%}), "
            f"Balance: ${self.current_balance:.2f}"
        )
        
        return {
            'success': True,
            'pnl': pnl,
            'return_pct': return_pct,
            'trade_record': trade_record
        }
    
    def update_positions(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Update unrealized PnL for all positions.
        
        Returns:
            Dict of unrealized PnL by symbol
        """
        unrealized_pnl = {}
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                if position['side'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['size']
                else:  # short
                    pnl = (position['entry_price'] - current_price) * position['size']
                
                unrealized_pnl[symbol] = pnl
                
                # Update position
                position['current_price'] = current_price
                position['unrealized_pnl'] = pnl
        
        return unrealized_pnl
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        total_unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0) for pos in self.positions.values()
        )
        
        total_equity = self.current_balance + total_unrealized_pnl
        
        return {
            'current_balance': self.current_balance,
            'total_equity': total_equity,
            'total_exposure': self.total_exposure,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'win_rate_today': self.wins_today / max(self.trades_today, 1),
            'active_positions': len(self.positions),
            'unrealized_pnl': total_unrealized_pnl,
            'positions': self.positions
        }
    
    def reset_daily(self):
        """Reset daily tracking metrics."""
        self.daily_pnl = 0.0
        self.daily_start_balance = self.current_balance
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.logger.info("Daily metrics reset")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from trade history."""
        if not self.position_history:
            return {}
        
        df = pd.DataFrame(self.position_history)
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.current_drawdown
        }

def main():
    """Test leverage management system."""
    # Create manager
    manager = LeverageManager()
    
    # Test position opening
    print("🧪 Testing Leverage Management System...")
    
    # Open a long position
    result = manager.open_position(
        symbol='FARTCOIN',
        side='long',
        price=0.001,
        signal_strength=0.8
    )
    
    if result['success']:
        print(f"✅ Opened position: {result['position']}")
        
        # Update with new price
        current_prices = {'FARTCOIN': 0.0012}
        unrealized_pnl = manager.update_positions(current_prices)
        print(f"📊 Unrealized PnL: {unrealized_pnl}")
        
        # Close position
        close_result = manager.close_position('FARTCOIN', 0.0012)
        if close_result['success']:
            print(f"✅ Closed position: PnL ${close_result['pnl']:.2f}")
    
    # Get summary
    summary = manager.get_portfolio_summary()
    print(f"📈 Portfolio Summary: {summary}")
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    print(f"📊 Performance Metrics: {metrics}")

if __name__ == "__main__":
    main() 