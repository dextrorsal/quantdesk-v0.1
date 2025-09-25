import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from src.ml.indicators.base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig
from contextlib import nullcontext


@dataclass
class ChandelierConfig:
    """Configuration for Chandelier Exit"""

    atr_period: int = 22
    atr_multiplier: float = 3.0
    use_close: bool = True
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    use_amp: bool = False


class ChandelierExit(BaseTorchIndicator):
    """
    Chandelier Exit indicator for risk management.
    Provides trailing stop levels based on ATR.
    """

    def __init__(
        self,
        atr_period: int = 22,
        atr_multiplier: float = 3.0,
        use_close: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ChandelierConfig] = None,
    ):
        """Initialize with configuration"""
        if config is None:
            config = ChandelierConfig(
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
                use_close=use_close,
                device=device,
                dtype=dtype,
            )
        # Convert ChandelierConfig to TorchIndicatorConfig for base class
        torch_config = TorchIndicatorConfig(
            device=config.device,
            dtype=config.dtype,
            use_amp=config.use_amp
        )
        super().__init__(torch_config)
        self._chandelier_config = config
        self.device = self._chandelier_config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = self._chandelier_config.dtype or torch.float32

    @property
    def atr_period(self) -> int:
        return self._chandelier_config.atr_period

    @property
    def atr_multiplier(self) -> float:
        return self._chandelier_config.atr_multiplier

    @property
    def use_close(self) -> bool:
        return self._chandelier_config.use_close

    def calculate_atr(
        self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Average True Range using PyTorch operations"""
        high_low = high - low
        high_close_prev = torch.abs(high - torch.roll(close, 1))
        low_close_prev = torch.abs(low - torch.roll(close, 1))

        # Handle first element where prev close doesn't exist
        high_close_prev[0] = high_low[0]
        low_close_prev[0] = high_low[0]

        tr = torch.maximum(high_low, torch.maximum(high_close_prev, low_close_prev))
        atr = self.torch_ema(tr, alpha=2.0 / (self.atr_period + 1))

        return atr

    def forward(
        self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate Chandelier Exit levels"""
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)

        # Calculate highest high and lowest low over lookback period
        highest_high = self.calculate_rolling_max(
            high if not self.use_close else close, self.atr_period
        )
        lowest_low = self.calculate_rolling_min(
            low if not self.use_close else close, self.atr_period
        )

        # Calculate trailing stop levels
        long_stop = torch.zeros_like(close)
        short_stop = torch.zeros_like(close)
        
        # Initialize first values
        long_stop[0] = close[0] - self.atr_multiplier * atr[0]
        short_stop[0] = close[0] + self.atr_multiplier * atr[0]
        
        # Update trailing stops
        for i in range(1, len(close)):
            # Long stop: trail upward when price rises, reset when price breaks below
            new_long_stop = highest_high[i] - self.atr_multiplier * atr[i]
            long_stop[i] = torch.maximum(new_long_stop, long_stop[i-1]) if close[i-1] > long_stop[i-1] else new_long_stop
            
            # Short stop: trail downward when price falls, reset when price breaks above  
            new_short_stop = lowest_low[i] + self.atr_multiplier * atr[i]
            short_stop[i] = torch.minimum(new_short_stop, short_stop[i-1]) if close[i-1] < short_stop[i-1] else new_short_stop

        # Replace NaN values with reasonable defaults
        long_stop = torch.nan_to_num(long_stop, nan=close[0].item() * 0.98)
        short_stop = torch.nan_to_num(short_stop, nan=close[0].item() * 1.02)

        # Generate signals with improved logic
        buy_signals = torch.zeros_like(close, dtype=torch.bool, device=self.device)
        sell_signals = torch.zeros_like(close, dtype=torch.bool, device=self.device)
        
        # Track position state
        position = 0  # 0=neutral, 1=long, -1=short
        
        for i in range(1, len(close)):
            current_price = close[i]
            prev_price = close[i-1]
            
            # Entry signals
            if position <= 0:  # Not long, can go long
                # Buy when price moves above long stop with momentum
                if current_price > long_stop[i] and prev_price <= long_stop[i-1]:
                    buy_signals[i] = True
                    position = 1
                    
            if position >= 0:  # Not short, can go short  
                # Sell when price moves below short stop with momentum
                if current_price < short_stop[i] and prev_price >= short_stop[i-1]:
                    sell_signals[i] = True
                    position = -1
                    
            # Exit signals based on stop violations
            if position == 1:  # Long position
                if current_price < long_stop[i]:  # Stop loss hit
                    sell_signals[i] = True
                    position = 0
                    
            if position == -1:  # Short position  
                if current_price > short_stop[i]:  # Stop loss hit
                    buy_signals[i] = True
                    position = 0

        return {
            "long_stop": long_stop,
            "short_stop": short_stop,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
        }

    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Calculate Chandelier Exit signals from OHLCV data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with buy_signals, sell_signals, predictions, long_signals, short_signals
        """
        high = self.to_tensor(data["high"])
        low = self.to_tensor(data["low"])
        close = self.to_tensor(data["close"])

        with torch.amp.autocast("cuda") if self._chandelier_config.use_amp else nullcontext():
            result = self.forward(high, low, close)
            
            # Add required keys for paper trading framework compatibility
            result["predictions"] = (result["long_stop"] + result["short_stop"]) / 2  # Average of stops as prediction
            result["long_signals"] = result["buy_signals"]  # Alias for consistency
            result["short_signals"] = result["sell_signals"]  # Alias for consistency
            
            return result

    def update_stops(
        self, current_price: float, position_type: str, current_stop: float
    ) -> float:
        """
        Update stop levels for an open position.

        Args:
            current_price: Current market price
            position_type: 'long' or 'short'
            current_stop: Current stop level

        Returns:
            Updated stop level
        """
        price_tensor = torch.tensor(
            [[current_price]], dtype=self.dtype, device=self.device
        )
        signals = self.forward(price_tensor, price_tensor, price_tensor)

        if position_type.lower() == "long":
            return float(signals["long_stop"].item())
        else:
            return float(signals["short_stop"].item())

    def calculate_rolling_max(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate rolling maximum over a window

        Args:
            x: Input tensor
            window: Window size

        Returns:
            Rolling maximum tensor
        """
        # Create rolling windows
        if len(x) < window:
            # If data length is less than window, return the max of available data
            max_val = torch.max(x)
            return torch.full_like(x, max_val)

        x_unfold = x.unfold(0, window, 1)

        # Calculate max for each window
        rolling_max = torch.max(x_unfold, dim=1)[0]

        # Pad initial values
        padding = torch.full(
            (window - 1,), float("nan"), device=self.device, dtype=self.dtype
        )
        rolling_max = torch.cat([padding, rolling_max])

        return rolling_max

    def calculate_rolling_min(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate rolling minimum over a window

        Args:
            x: Input tensor
            window: Window size

        Returns:
            Rolling minimum tensor
        """
        # Create rolling windows
        if len(x) < window:
            # If data length is less than window, return the min of available data
            min_val = torch.min(x)
            return torch.full_like(x, min_val)

        x_unfold = x.unfold(0, window, 1)

        # Calculate min for each window
        rolling_min = torch.min(x_unfold, dim=1)[0]

        # Pad initial values
        padding = torch.full(
            (window - 1,), float("nan"), device=self.device, dtype=self.dtype
        )
        rolling_min = torch.cat([padding, rolling_min])

        return rolling_min


def calculate_ohlc4(df):
    """Calculate OHLC4 (average of open, high, low, close)"""
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4


# Example usage:
if __name__ == "__main__":
    # Sample data
    data = {
        "open": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
        "high": [12, 13, 14, 12, 11, 10, 11, 12, 13, 14],
        "low": [9, 10, 11, 10, 9, 8, 9, 10, 11, 12],
        "close": [11, 12, 13, 10, 9, 9, 10, 11, 12, 13],
    }

    df = pd.DataFrame(data)

    # Calculate Chandelier Exit
    config = ChandelierConfig(atr_period=22, atr_multiplier=3.0, use_close=True)
    chandelier = ChandelierExit(config)
    result = chandelier.calculate_signals(df)

    # Print results
    print("Chandelier Exit Results:")
    print(result)
