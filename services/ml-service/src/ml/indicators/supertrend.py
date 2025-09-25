"""
Supertrend Indicator Implementation

This module implements the Supertrend indicator with PyTorch acceleration,
matching TradingView's Supertrend parameters exactly.

Supertrend is a trend-following indicator that combines ATR (Average True Range)
with price action to generate buy/sell signals.

TradingView Parameters:
- Period: 10 (ATR period)
- Multiplier: 3.0 (ATR multiplier)
- Source: Close price
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig


@dataclass
class SupertrendConfig(TorchIndicatorConfig):
    """Configuration for Supertrend indicator"""
    
    period: int = 10
    multiplier: float = 3.0
    source: str = "close"  # "close", "hl2", "hlc3", "ohlc4"


class SupertrendIndicator(BaseTorchIndicator):
    """
    Supertrend indicator implementation with PyTorch acceleration.
    
    The Supertrend indicator is a trend-following indicator that uses ATR
    to determine trend direction and generate buy/sell signals.
    
    Formula:
    1. Calculate ATR(period)
    2. Calculate basic upper band = (high + low) / 2 + (multiplier * ATR)
    3. Calculate basic lower band = (high + low) / 2 - (multiplier * ATR)
    4. Calculate final upper band = min(basic_upper_band, previous_final_upper_band) if close > previous_final_upper_band else basic_upper_band
    5. Calculate final lower band = max(basic_lower_band, previous_final_lower_band) if close < previous_final_lower_band else basic_lower_band
    6. Supertrend = final_upper_band if close <= final_upper_band else final_lower_band
    7. Direction = 1 if close > supertrend else -1
    """
    
    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        source: str = "close",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[SupertrendConfig] = None,
    ):
        """
        Initialize Supertrend indicator
        
        Args:
            period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 3.0)
            source: Price source ("close", "hl2", "hlc3", "ohlc4")
            device: Device to use for computations
            dtype: Data type to use for computations
            config: Configuration object
        """
        if config is None:
            config = SupertrendConfig(
                period=period,
                multiplier=multiplier,
                source=source,
                device=device,
                dtype=dtype,
            )
        super().__init__(config)
        
        self._period = config.period
        self._multiplier = config.multiplier
        self._source = config.source
    
    @property
    def period(self) -> int:
        """Get ATR period"""
        return self._period
    
    @property
    def multiplier(self) -> float:
        """Get ATR multiplier"""
        return self._multiplier
    
    @property
    def source(self) -> str:
        """Get price source"""
        return self._source
    
    def calculate_atr(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> torch.Tensor:
        """Calculate Average True Range"""
        # Calculate True Range
        high_low = high - low
        high_close_prev = torch.abs(high - torch.roll(close, 1))
        low_close_prev = torch.abs(low - torch.roll(close, 1))
        
        # Handle first element
        high_close_prev[0] = high_low[0]
        low_close_prev[0] = high_low[0]
        
        # True Range is the maximum of the three
        tr = torch.maximum(high_low, torch.maximum(high_close_prev, low_close_prev))
        
        # Calculate ATR using EMA
        atr = self.torch_ema(tr, alpha=2.0 / (self._period + 1))
        
        return atr
    
    def get_price_source(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, open: torch.Tensor = None) -> torch.Tensor:
        """Get price source based on configuration"""
        if self._source == "close":
            return close
        elif self._source == "hl2":
            return (high + low) / 2.0
        elif self._source == "hlc3":
            return (high + low + close) / 3.0
        elif self._source == "ohlc4":
            if open is None:
                raise ValueError("Open prices required for ohlc4 source")
            return (open + high + low + close) / 4.0
        else:
            raise ValueError(f"Unknown source: {self._source}")
    
    def forward(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, open: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Calculate Supertrend indicator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open: Open prices (optional, required for ohlc4 source)
            
        Returns:
            Dictionary with supertrend values and signals
        """
        # Get price source
        price = self.get_price_source(high, low, close, open)
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)
        
        # Calculate basic bands
        hl2 = (high + low) / 2.0
        basic_upper_band = hl2 + (self._multiplier * atr)
        basic_lower_band = hl2 - (self._multiplier * atr)
        
        # Initialize final bands
        final_upper_band = torch.zeros_like(close)
        final_lower_band = torch.zeros_like(close)
        
        # Initialize with basic bands
        final_upper_band[0] = basic_upper_band[0]
        final_lower_band[0] = basic_lower_band[0]
        
        # Calculate final bands iteratively
        for i in range(1, len(close)):
            # Final upper band logic
            if basic_upper_band[i] < final_upper_band[i-1] or close[i-1] > final_upper_band[i-1]:
                final_upper_band[i] = basic_upper_band[i]
            else:
                final_upper_band[i] = final_upper_band[i-1]
            
            # Final lower band logic
            if basic_lower_band[i] > final_lower_band[i-1] or close[i-1] < final_lower_band[i-1]:
                final_lower_band[i] = basic_lower_band[i]
            else:
                final_lower_band[i] = final_lower_band[i-1]
        
        # Calculate Supertrend
        supertrend = torch.zeros_like(close)
        direction = torch.zeros_like(close)
        
        for i in range(len(close)):
            if i == 0:
                supertrend[i] = final_lower_band[i]
                direction[i] = -1
            else:
                if close[i] <= final_upper_band[i]:
                    supertrend[i] = final_upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower_band[i]
                    direction[i] = 1
        
        # Generate buy/sell signals
        buy_signals = torch.zeros_like(close, dtype=torch.bool)
        sell_signals = torch.zeros_like(close, dtype=torch.bool)
        
        # Detect direction changes
        for i in range(1, len(direction)):
            if direction[i] != direction[i-1]:
                if direction[i] == 1:  # Bullish
                    buy_signals[i] = True
                else:  # Bearish
                    sell_signals[i] = True
        
        return {
            "supertrend": supertrend,
            "direction": direction,
            "final_upper_band": final_upper_band,
            "final_lower_band": final_lower_band,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "atr": atr
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate Supertrend signals from OHLCV data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with supertrend values and signals
        """
        high = self.to_tensor(data["high"])
        low = self.to_tensor(data["low"])
        close = self.to_tensor(data["close"])
        open_price = self.to_tensor(data["open"]) if "open" in data.columns else None
        
        result = self.forward(high, low, close, open_price)
        
        # Add compatibility keys
        result["predictions"] = result["supertrend"]
        result["long_signals"] = result["buy_signals"]
        result["short_signals"] = result["sell_signals"]
        
        return result


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0, source: str = "close") -> pd.DataFrame:
    """
    Calculate Supertrend indicator for a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        multiplier: ATR multiplier
        source: Price source
        
    Returns:
        DataFrame with Supertrend columns added
    """
    indicator = SupertrendIndicator(period=period, multiplier=multiplier, source=source)
    signals = indicator.calculate_signals(df)
    
    # Convert tensors to numpy arrays
    result_df = df.copy()
    result_df["supertrend"] = signals["supertrend"].cpu().numpy()
    result_df["supertrend_direction"] = signals["direction"].cpu().numpy()
    result_df["supertrend_upper"] = signals["final_upper_band"].cpu().numpy()
    result_df["supertrend_lower"] = signals["final_lower_band"].cpu().numpy()
    result_df["supertrend_buy"] = signals["buy_signals"].cpu().numpy()
    result_df["supertrend_sell"] = signals["sell_signals"].cpu().numpy()
    result_df["atr"] = signals["atr"].cpu().numpy()
    
    return result_df


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        "open": [100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104, 103, 102, 101, 100],
        "high": [102, 103, 104, 102, 101, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
        "low": [99, 100, 101, 100, 99, 98, 99, 100, 101, 102, 103, 102, 101, 100, 99],
        "close": [101, 102, 103, 100, 99, 99, 100, 101, 102, 103, 104, 102, 101, 100, 99],
        "volume": [1000, 1100, 1200, 1050, 950, 900, 1000, 1100, 1200, 1300, 1400, 1200, 1100, 1000, 900]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Supertrend
    result = calculate_supertrend(df, period=10, multiplier=3.0)
    
    print("Supertrend Results:")
    print(result[["close", "supertrend", "supertrend_direction", "supertrend_buy", "supertrend_sell"]])
