"""
Enhanced WaveTrend Implementation using PyTorch

Features:
- GPU acceleration for faster calculations
- Real-time signal generation
- Configurable parameters
- Advanced signal filtering
- Backtesting metrics
- Debug capabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Union
from src.ml.indicators.base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig
from contextlib import nullcontext


@dataclass
class WaveTrendMetrics:
    """Container for WaveTrend trading metrics"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0


@dataclass
class WaveTrendConfig(TorchIndicatorConfig):
    """Configuration for WaveTrend indicator"""

    channel_length: int = 10
    average_length: int = 21
    smoothing_length: int = 4
    overbought: float = 60.0
    oversold: float = -60.0
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    use_amp: bool = False


class WaveTrendIndicator(BaseTorchIndicator):
    """
    PyTorch-based WaveTrend implementation with advanced features
    """

    def __init__(
        self,
        channel_length: int = 10,
        average_length: int = 11,
        smoothing_length: int = 4,
        overbought: float = 60.0,
        oversold: float = -60.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[WaveTrendConfig] = None,
    ):
        """
        Initialize WaveTrend indicator with PyTorch backend

        Args:
            channel_length: Length of the channel period
            average_length: Length of the average period
            smoothing_length: Length of the smoothing SMA
            overbought: Overbought threshold
            oversold: Oversold threshold
            device: Device to use for computations
            dtype: Data type to use for computations
            config: Configuration object
        """
        if config is None:
            config = WaveTrendConfig(
                channel_length=channel_length,
                average_length=average_length,
                smoothing_length=smoothing_length,
                overbought=overbought,
                oversold=oversold,
                device=device,
                dtype=dtype,
            )
        super().__init__(config)

        self._channel_length = config.channel_length
        self._average_length = config.average_length
        self._smoothing_length = config.smoothing_length
        self._overbought = config.overbought
        self._oversold = config.oversold

        # Trading metrics
        self.metrics = WaveTrendMetrics()
        self.last_price = None

        # Initialize internal configuration values
        self.channel_multiplier = 0.015  # Standard multiplier for WaveTrend

    @property
    def channel_length(self) -> int:
        """Get channel length"""
        return self._channel_length

    @channel_length.setter
    def channel_length(self, value: int):
        """Set channel length"""
        self._channel_length = value

    @property
    def average_length(self) -> int:
        """Get average length"""
        return self._average_length

    @average_length.setter
    def average_length(self, value: int):
        """Set average length"""
        self._average_length = value

    @property
    def smoothing_length(self) -> int:
        """Get smoothing length"""
        return self._smoothing_length

    @smoothing_length.setter
    def smoothing_length(self, value: int):
        """Set smoothing length"""
        self._smoothing_length = value

    @property
    def overbought(self) -> float:
        """Get overbought threshold"""
        return self._overbought

    @overbought.setter
    def overbought(self, value: float):
        """Set overbought threshold"""
        self._overbought = value

    @property
    def oversold(self) -> float:
        """Get oversold threshold"""
        return self._oversold

    @oversold.setter
    def oversold(self, value: float):
        """Set oversold threshold"""
        self._oversold = value

    def forward(
        self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate WaveTrend indicator values

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Dictionary with WT1, WT2 and trading signals
        """
        # Calculate HLC3 (typical price)
        hlc3 = (high + low + close) / 3.0

        # Calculate EMA and standard deviation
        ema1 = self.ema(hlc3, self._channel_length)

        # Calculate absolute price distance
        abs_diff = torch.abs(hlc3 - ema1)

        # Calculate average distance (ATR-like calculation)
        avg_diff = self.ema(abs_diff, self._channel_length)

        # Normalize using channel multiplier
        normalized = self.channel_multiplier * avg_diff

        # Apply limits to avoid division by zero and large values
        epsilon = 1e-10
        normalized = torch.clamp(normalized, min=epsilon)

        # Wave 1 calculation - normalized oscillator
        wave1 = (hlc3 - ema1) / normalized

        # Wave 2 calculation - signal line
        wave2 = self.ema(wave1, self._smoothing_length)

        # Apply clamps to keep values within reasonable ranges
        wt1 = torch.clamp(wave1, min=-100.0, max=100.0)
        wt2 = torch.clamp(wave2, min=-100.0, max=100.0)

        # Generate signals
        buy_signals = torch.zeros_like(wt1, dtype=self.dtype)
        sell_signals = torch.zeros_like(wt1, dtype=self.dtype)

        # Calculate crossovers
        valid_mask = ~torch.isnan(wt1) & ~torch.isnan(wt2)
        buy_signals[valid_mask] = (wt1[valid_mask] > wt2[valid_mask]).to(self.dtype)
        sell_signals[valid_mask] = (wt1[valid_mask] < wt2[valid_mask]).to(self.dtype)

        return {
            "wt1": wt1,
            "wt2": wt2,
            "wave_trend": wt1,  # Add wave_trend key for compatibility with Lorentzian Classifier
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
        }

    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate WaveTrend and generate trading signals

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with WaveTrend values and signals
        """
        # Convert price data to tensors
        high = self.to_tensor(data["high"])
        low = self.to_tensor(data["low"])
        close = self.to_tensor(data["close"])

        # Calculate WaveTrend and signals
        with torch.amp.autocast("cuda") if self.config.use_amp else nullcontext():
            results = self.forward(high, low, close)

        return results

    def update_metrics(
        self, current_price: float, signal: int, last_signal: int
    ) -> None:
        """Update trading metrics"""
        if last_signal != 0 and signal != last_signal:
            self.metrics.total_trades += 1
            pnl = (current_price - self.last_price) * last_signal

            if pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.avg_win = (
                    self.metrics.avg_win * (self.metrics.winning_trades - 1) + pnl
                ) / self.metrics.winning_trades
            else:
                self.metrics.losing_trades += 1
                self.metrics.avg_loss = (
                    self.metrics.avg_loss * (self.metrics.losing_trades - 1) + abs(pnl)
                ) / self.metrics.losing_trades

            if self.metrics.total_trades > 0:
                self.metrics.win_rate = (
                    self.metrics.winning_trades / self.metrics.total_trades
                )

            if self.metrics.avg_loss > 0:
                self.metrics.profit_factor = (
                    self.metrics.avg_win * self.metrics.winning_trades
                ) / (self.metrics.avg_loss * self.metrics.losing_trades)

        self.last_price = current_price

    def plot_signals(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> None:
        """Plot WaveTrend with signals using matplotlib"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

            # Plot price and signals
            ax1.plot(df.index, df["close"], label="Price", alpha=0.7)
            buy_points = df.index[signals["buy_signals"] == 1]
            sell_points = df.index[signals["sell_signals"] == 1]

            if len(buy_points) > 0:
                ax1.scatter(
                    buy_points,
                    df.loc[buy_points, "close"],
                    color="green",
                    marker="^",
                    label="Buy",
                )
            if len(sell_points) > 0:
                ax1.scatter(
                    sell_points,
                    df.loc[sell_points, "close"],
                    color="red",
                    marker="v",
                    label="Sell",
                )

            ax1.set_title("Price with WaveTrend Signals")
            ax1.legend()

            # Plot WaveTrend
            ax2.plot(df.index, signals["wt1"], label="WaveTrend 1", color="blue")
            ax2.plot(df.index, signals["wt2"], label="WaveTrend 2", color="orange")
            ax2.axhline(y=self.overbought, color="r", linestyle="--", alpha=0.5)
            ax2.axhline(y=self.oversold, color="g", linestyle="--", alpha=0.5)
            ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            ax2.set_title("WaveTrend")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting signals: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current trading metrics"""
        return {
            "total_trades": self.metrics.total_trades,
            "win_rate": self.metrics.win_rate,
            "profit_factor": self.metrics.profit_factor,
            "avg_win": self.metrics.avg_win,
            "avg_loss": self.metrics.avg_loss,
        }
