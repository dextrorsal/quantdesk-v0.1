#!/usr/bin/env python3
"""
🛡️ HFT Risk Management System

Risk management layer for high-frequency trading that can be applied on top of any strategy.
This provides position sizing, stop-loss, and risk controls without interfering with the core strategy.

Features:
- Dynamic position sizing based on volatility
- ATR-based stop-loss management
- Maximum drawdown protection
- Daily loss limits
- Position limits per symbol
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class RiskConfig:
    """Configuration for HFT risk management"""
    max_risk_per_trade: float = 0.01  # 1% risk per trade
    max_daily_loss: float = 0.02      # 2% max daily loss
    max_drawdown: float = 0.03        # 3% max drawdown
    max_position_size: float = 0.05   # 5% max position size
    atr_multiplier: float = 2.0       # ATR multiplier for stop-loss
    atr_period: int = 14              # ATR calculation period

class HFTRiskManager:
    """Risk management system for high-frequency trading."""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.current_equity = 0.0
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range for stop-loss sizing."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.config.atr_period).mean()
        
        return atr
    
    def calculate_position_size(self, signal: int, price: float, atr: float, 
                              account_balance: float) -> Tuple[float, float]:
        """
        Calculate optimal position size and stop-loss level.
        
        Args:
            signal: 1 for long, -1 for short, 0 for no position
            price: Current price
            atr: Current ATR value
            account_balance: Current account balance
            
        Returns:
            Tuple of (position_size, stop_loss_price)
        """
        if signal == 0:
            return 0.0, 0.0
        
        # Calculate stop-loss distance
        stop_distance = atr * self.config.atr_multiplier
        
        # Calculate stop-loss price
        if signal == 1:  # Long position
            stop_loss = price - stop_distance
        else:  # Short position
            stop_loss = price + stop_distance
        
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss)
        
        # Calculate maximum risk amount
        max_risk_amount = account_balance * self.config.max_risk_per_trade
        
        # Calculate position size based on risk
        position_size = max_risk_amount / risk_per_share
        
        # Apply position size limits
        max_position_value = account_balance * self.config.max_position_size
        max_shares = max_position_value / price
        
        position_size = min(position_size, max_shares)
        
        return position_size, stop_loss
    
    def check_risk_limits(self, new_position_value: float, 
                         account_balance: float) -> bool:
        """
        Check if new position violates risk limits.
        
        Returns:
            True if position is allowed, False if it violates limits
        """
        # Check daily loss limit
        if self.daily_pnl < -(account_balance * self.config.max_daily_loss):
            self.logger.warning("Daily loss limit exceeded")
            return False
        
        # Check drawdown limit
        current_drawdown = (self.max_equity - self.current_equity) / self.max_equity
        if current_drawdown > self.config.max_drawdown:
            self.logger.warning("Maximum drawdown exceeded")
            return False
        
        # Check position size limit
        if new_position_value > account_balance * self.config.max_position_size:
            self.logger.warning("Position size limit exceeded")
            return False
        
        return True
    
    def update_position(self, symbol: str, position_size: float, 
                       entry_price: float, current_price: float):
        """Update position tracking."""
        self.current_positions[symbol] = {
            'size': position_size,
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pnl': (current_price - entry_price) * position_size
        }
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized PnL."""
        if symbol not in self.current_positions:
            return 0.0
        
        position = self.current_positions[symbol]
        realized_pnl = (exit_price - position['entry_price']) * position['size']
        
        # Update daily PnL
        self.daily_pnl += realized_pnl
        
        # Update equity tracking
        self.current_equity += realized_pnl
        self.max_equity = max(self.max_equity, self.current_equity)
        
        # Remove position
        del self.current_positions[symbol]
        
        return realized_pnl
    
    def process_signal(self, signal: int, price: float, atr: float, 
                      account_balance: float, symbol: str) -> Dict:
        """
        Process a trading signal with risk management.
        
        Args:
            signal: Strategy signal (1=long, -1=short, 0=no position)
            price: Current price
            atr: Current ATR value
            account_balance: Current account balance
            symbol: Trading symbol
            
        Returns:
            Dictionary with position details and risk metrics
        """
        result = {
            'action': 'hold',
            'position_size': 0.0,
            'stop_loss': 0.0,
            'risk_amount': 0.0,
            'allowed': False,
            'reason': 'no_signal'
        }
        
        if signal == 0:
            return result
        
        # Calculate position size and stop-loss
        position_size, stop_loss = self.calculate_position_size(
            signal, price, atr, account_balance
        )
        
        # Calculate position value
        position_value = position_size * price
        
        # Check risk limits
        if not self.check_risk_limits(position_value, account_balance):
            result['reason'] = 'risk_limit_exceeded'
            return result
        
        # Determine action
        current_position = self.current_positions.get(symbol, None)
        
        if current_position is None:
            # No current position - can enter new position
            if position_size > 0:
                result['action'] = 'enter_long' if signal == 1 else 'enter_short'
                result['position_size'] = position_size
                result['stop_loss'] = stop_loss
                result['risk_amount'] = abs(price - stop_loss) * position_size
                result['allowed'] = True
                result['reason'] = 'new_position'
                
                # Update position tracking
                self.update_position(symbol, position_size, price, price)
        
        else:
            # Check if we need to exit current position
            current_pnl = (price - current_position['entry_price']) * current_position['size']
            
            # Exit if stop-loss hit or signal reversed
            if (signal == 1 and current_position['size'] < 0) or \
               (signal == -1 and current_position['size'] > 0) or \
               (signal == 1 and price <= stop_loss) or \
               (signal == -1 and price >= stop_loss):
                
                result['action'] = 'exit'
                result['position_size'] = current_position['size']
                result['stop_loss'] = stop_loss
                result['allowed'] = True
                result['reason'] = 'exit_signal'
                
                # Close position
                self.close_position(symbol, price)
        
        return result
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        total_unrealized_pnl = sum(
            pos['unrealized_pnl'] for pos in self.current_positions.values()
        )
        
        current_drawdown = 0.0
        if self.max_equity > 0:
            current_drawdown = (self.max_equity - self.current_equity) / self.max_equity
        
        return {
            'daily_pnl': self.daily_pnl,
            'current_drawdown': current_drawdown,
            'max_equity': self.max_equity,
            'current_equity': self.current_equity,
            'open_positions': len(self.current_positions),
            'total_unrealized_pnl': total_unrealized_pnl
        }
    
    def reset_daily(self):
        """Reset daily tracking (call at start of new day)."""
        self.daily_pnl = 0.0
        self.current_positions = {}

class HFTStrategyWithRisk:
    """Combines HFT strategy with risk management."""
    
    def __init__(self, strategy_model, risk_manager: HFTRiskManager):
        self.strategy = strategy_model
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, df: pd.DataFrame, account_balance: float) -> pd.DataFrame:
        """Generate trading signals with risk management."""
        # Get strategy signals
        strategy_signals = self.strategy.calculate_signals(df)
        
        # Calculate ATR for risk management
        atr = self.risk_manager.calculate_atr(df)
        
        # Process each signal with risk management
        results = []
        
        for i in range(len(df)):
            signal = strategy_signals['buy_signals'][i] - strategy_signals['sell_signals'][i]
            price = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            
            # Process with risk management
            risk_result = self.risk_manager.process_signal(
                signal, price, current_atr, account_balance, 'BTC'
            )
            
            results.append({
                'timestamp': df['timestamp'].iloc[i],
                'price': price,
                'strategy_signal': signal,
                'action': risk_result['action'],
                'position_size': risk_result['position_size'],
                'stop_loss': risk_result['stop_loss'],
                'risk_amount': risk_result['risk_amount'],
                'allowed': risk_result['allowed'],
                'reason': risk_result['reason']
            })
        
        return pd.DataFrame(results)

def main():
    """Example usage of HFT risk management."""
    # Create risk manager
    risk_config = RiskConfig(
        max_risk_per_trade=0.01,  # 1% risk per trade
        max_daily_loss=0.02,      # 2% max daily loss
        max_drawdown=0.03,        # 3% max drawdown
        max_position_size=0.05    # 5% max position size
    )
    
    risk_manager = HFTRiskManager(risk_config)
    
    print("🛡️ HFT Risk Management System Ready!")
    print(f"Max Risk per Trade: {risk_config.max_risk_per_trade*100}%")
    print(f"Max Daily Loss: {risk_config.max_daily_loss*100}%")
    print(f"Max Drawdown: {risk_config.max_drawdown*100}%")
    print(f"Max Position Size: {risk_config.max_position_size*100}%")

if __name__ == "__main__":
    main() 