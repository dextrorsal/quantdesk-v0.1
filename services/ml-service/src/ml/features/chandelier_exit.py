"""
Enhanced Chandelier Exit Implementation using PyTorch

Features:
- GPU acceleration for faster calculations
- Real-time signal generation
- PineScript-accurate calculations
- Advanced ML features
- Backtesting metrics
- Debug capabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple
from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig

@dataclass
class ChandelierConfig:
    """Configuration for Chandelier Exit"""
    atr_period: int = 22
    atr_multiplier: float = 3.0
    use_close: bool = True
    await_confirmation: bool = True
    show_labels: bool = True
    highlight_state: bool = True

@dataclass
class ChandelierMetrics:
    """Container for Chandelier Exit trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_stop_distance: float = 0.0
    # ML-specific metrics
    avg_trend_duration: float = 0.0
    avg_stop_volatility: float = 0.0
    trend_strength: float = 0.0
    stop_efficiency: float = 0.0

class ChandelierExitIndicator(BaseTorchIndicator):
    """
    PyTorch-based Chandelier Exit implementation matching PineScript with ML enhancements
    
    The Chandelier Exit sets trailing stops based on ATR, helping to:
    - Protect profits in trending markets
    - Limit losses with volatility-based stops
    - Generate clear entry/exit signals
    - Provide ML-friendly features for model training
    """
    
    def __init__(
        self,
        config: Optional[ChandelierConfig] = None,
        torch_config: Optional[TorchIndicatorConfig] = None
    ):
        """
        Initialize Chandelier Exit indicator with PyTorch backend
        
        Args:
            config: Chandelier-specific configuration
            torch_config: PyTorch configuration for GPU/CPU
        """
        super().__init__(torch_config)
        
        self.config = config or ChandelierConfig()
        self.metrics = ChandelierMetrics()
        self.last_price = None
        self.trend_start_price = None
        self.trend_start_idx = 0
        
    def calculate_true_range(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> torch.Tensor:
        """Calculate True Range for ATR computation using PyTorch"""
        high_low = high - low
        high_close_prev = torch.abs(high - F.pad(close, (1, 0))[:close.shape[0]])
        low_close_prev = torch.abs(low - F.pad(close, (1, 0))[:close.shape[0]])
        
        ranges = torch.stack([high_low, high_close_prev, low_close_prev], dim=1)
        true_range = torch.max(ranges, dim=1)[0]
        
        return true_range
    
    def calculate_ohlc4(self, open: torch.Tensor, high: torch.Tensor, 
                       low: torch.Tensor, close: torch.Tensor) -> torch.Tensor:
        """Calculate OHLC4 price using PyTorch"""
        return (open + high + low + close) / 4.0
    
    def forward(
        self, 
        open: torch.Tensor, 
        high: torch.Tensor, 
        low: torch.Tensor, 
        close: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate Chandelier Exit values using PyTorch operations
        
        Args:
            open: Open prices tensor
            high: High prices tensor
            low: Low prices tensor
            close: Close prices tensor
            
        Returns:
            Dictionary with Chandelier Exit values and signals
        """
        # Calculate True Range and ATR (matching PineScript ta.atr)
        tr = self.calculate_true_range(high, low, close)
        atr = self.torch_sma(tr, self.config.atr_period)
        atr = self.config.atr_multiplier * atr
        
        # Calculate highest/lowest (matching PineScript ta.highest/ta.lowest)
        if self.config.use_close:
            highest = self.rolling_max(close, self.config.atr_period)
            lowest = self.rolling_min(close, self.config.atr_period)
        else:
            highest = self.rolling_max(high, self.config.atr_period)
            lowest = self.rolling_min(low, self.config.atr_period)
        
        # Calculate initial stops
        long_stop = highest - atr
        short_stop = lowest + atr
        
        # Apply trailing stop logic (matching PineScript logic)
        long_stop_final = torch.zeros_like(long_stop)
        short_stop_final = torch.zeros_like(short_stop)
        direction = torch.ones_like(close)  # 1 for long, -1 for short
        
        # Initialize first values
        long_stop_final[0] = long_stop[0]
        short_stop_final[0] = short_stop[0]
        
        # Calculate OHLC4 for ML features
        ohlc4 = self.calculate_ohlc4(open, high, low, close)
        
        # Vectorized trailing stop calculation
        for i in range(1, len(close)):
            # Long stop logic (matching PineScript)
            if close[i-1] > long_stop_final[i-1]:
                long_stop_final[i] = torch.max(long_stop[i], long_stop_final[i-1])
            else:
                long_stop_final[i] = long_stop[i]
            
            # Short stop logic (matching PineScript)
            if close[i-1] < short_stop_final[i-1]:
                short_stop_final[i] = torch.min(short_stop[i], short_stop_final[i-1])
            else:
                short_stop_final[i] = short_stop[i]
            
            # Direction logic (matching PineScript)
            if close[i-1] > short_stop_final[i-1]:
                direction[i] = 1
            elif close[i-1] < long_stop_final[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        
        # Generate signals (matching PineScript)
        prev_direction = F.pad(direction[:-1], (1, 0), value=1)
        buy_signals = (direction == 1) & (prev_direction == -1)
        sell_signals = (direction == -1) & (prev_direction == 1)
        
        # Calculate ML-friendly features
        stop_distances = torch.where(
            direction == 1,
            close - long_stop_final,
            short_stop_final - close
        )
        
        trend_duration = torch.zeros_like(direction)
        current_duration = 0
        current_dir = direction[0]
        
        for i in range(len(direction)):
            if direction[i] == current_dir:
                current_duration += 1
            else:
                current_dir = direction[i]
                current_duration = 1
            trend_duration[i] = current_duration
        
        # Calculate trend strength using price momentum and stop distance
        momentum = (close - F.pad(close, (10, 0))[:close.shape[0]]) / close
        trend_strength = torch.abs(momentum) * (stop_distances / atr)
        
        return {
            'atr': atr,
            'long_stop': long_stop_final,
            'short_stop': short_stop_final,
            'direction': direction,
            'buy_signals': buy_signals.float(),
            'sell_signals': sell_signals.float(),
            'ohlc4': ohlc4,
            'stop_distances': stop_distances,
            'trend_duration': trend_duration,
            'trend_strength': trend_strength,
            'is_confirmed': torch.ones_like(close) if not self.config.await_confirmation else F.pad(torch.ones_like(close[:-1]), (1, 0), value=0)
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate Chandelier Exit and generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Chandelier Exit values and signals
        """
        # Convert price data to tensors
        open = self.to_tensor(data['open'])
        high = self.to_tensor(data['high'])
        low = self.to_tensor(data['low'])
        close = self.to_tensor(data['close'])
        
        # Calculate signals with automatic mixed precision
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            results = self.forward(open, high, low, close)
        
        # Update ML metrics
        if len(close) > 1:
            self.update_ml_metrics(results)
        
        return results
    
    def update_ml_metrics(self, results: Dict[str, torch.Tensor]) -> None:
        """Update ML-specific metrics"""
        with torch.no_grad():
            # Update trend metrics
            trend_lengths = results['trend_duration']
            self.metrics.avg_trend_duration = float(torch.mean(trend_lengths))
            
            # Update stop volatility
            stop_distances = results['stop_distances']
            self.metrics.avg_stop_volatility = float(torch.std(stop_distances))
            
            # Update trend strength
            self.metrics.trend_strength = float(torch.mean(results['trend_strength']))
            
            # Calculate stop efficiency (how well stops trail the price)
            direction = results['direction']
            close = results['ohlc4']
            stops = torch.where(
                direction == 1,
                results['long_stop'],
                results['short_stop']
            )
            ideal_stops = torch.where(
                direction == 1,
                self.rolling_min(close, 5),  # 5-bar trailing min for longs
                self.rolling_max(close, 5)   # 5-bar trailing max for shorts
            )
            stop_efficiency = torch.mean(torch.abs(stops - ideal_stops) / results['atr'])
            self.metrics.stop_efficiency = float(stop_efficiency)
    
    def get_metrics(self) -> Dict:
        """Get current trading and ML metrics"""
        return {
            # Trading metrics
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.win_rate,
            'profit_factor': self.metrics.profit_factor,
            'avg_win': self.metrics.avg_win,
            'avg_loss': self.metrics.avg_loss,
            'avg_stop_distance': self.metrics.avg_stop_distance,
            # ML metrics
            'avg_trend_duration': self.metrics.avg_trend_duration,
            'avg_stop_volatility': self.metrics.avg_stop_volatility,
            'trend_strength': self.metrics.trend_strength,
            'stop_efficiency': self.metrics.stop_efficiency
        }
    
    def plot_signals(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> None:
        """Plot Chandelier Exit with signals using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
            
            # Plot price and signals
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            ax1.plot(df.index, signals['long_stop'], '--', label='Long Stop', color='green', alpha=0.5)
            ax1.plot(df.index, signals['short_stop'], '--', label='Short Stop', color='red', alpha=0.5)
            
            # Highlight long/short states if enabled
            if self.config.highlight_state:
                long_mask = signals['direction'] == 1
                short_mask = signals['direction'] == -1
                
                ax1.fill_between(df.index, df['close'], signals['long_stop'],
                               where=long_mask, color='green', alpha=0.1)
                ax1.fill_between(df.index, df['close'], signals['short_stop'],
                               where=short_mask, color='red', alpha=0.1)
            
            # Plot signals
            if self.config.show_labels:
                buy_points = df.index[signals['buy_signals'] == 1]
                sell_points = df.index[signals['sell_signals'] == 1]
                
                if len(buy_points) > 0:
                    ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
                              color='green', marker='^', label='Buy')
                if len(sell_points) > 0:
                    ax1.scatter(sell_points, df.loc[sell_points, 'close'], 
                              color='red', marker='v', label='Sell')
            
            ax1.set_title('Price with Chandelier Exit Signals')
            ax1.legend()
            
            # Plot ML metrics
            ax2.plot(df.index, signals['trend_strength'], label='Trend Strength', color='blue')
            ax2.plot(df.index, signals['stop_distances'] / signals['atr'], 
                    label='Normalized Stop Distance', color='purple', alpha=0.5)
            ax2.set_title('ML Metrics')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting signals: {str(e)}")
    
    def rolling_max(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate rolling maximum using PyTorch"""
        if window == 1:
            return x
            
        max_pool = nn.MaxPool1d(window, stride=1, padding=window-1)
        padded = F.pad(x.unsqueeze(0).unsqueeze(0), (window-1, 0), mode='replicate')
        return max_pool(padded).squeeze()
    
    def rolling_min(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate rolling minimum using PyTorch"""
        if window == 1:
            return x
            
        return -self.rolling_max(-x, window) 