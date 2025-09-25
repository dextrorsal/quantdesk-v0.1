"""
Base PyTorch Indicator Class

This module provides a base class for all PyTorch-based technical indicators.
Features:
- GPU acceleration
- Batch processing
- Automatic differentiation
- Memory efficient operations
- Real-time signal generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
from dataclasses import dataclass
from contextlib import nullcontext


@dataclass
class TorchIndicatorConfig:
    """Configuration for PyTorch indicators"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 128
    use_amp: bool = True  # Automatic Mixed Precision for faster computation


class BaseTorchIndicator(nn.Module):
    """Base class for all PyTorch-based indicators"""

    def __init__(self, config: Optional[TorchIndicatorConfig] = None):
        super().__init__()
        self.config = config or TorchIndicatorConfig()
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.config.device is None
            else torch.device(self.config.device)
        )
        self.dtype = torch.float32 if self.config.dtype is None else self.config.dtype
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

    def to_tensor(
        self, data: Union[pd.Series, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Convert input data to tensor"""
        if isinstance(data, pd.Series):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device).to(self.dtype)

    def ema(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate Exponential Moving Average (EMA)

        Args:
            data: Input tensor
            period: EMA period

        Returns:
            EMA values as tensor
        """
        alpha = 2.0 / (period + 1)
        kernel = torch.tensor(
            [(1 - alpha) ** i for i in range(period)],
            device=self.device,
            dtype=self.dtype,
        )
        kernel = kernel / kernel.sum()

        # Pad the input data
        padding = torch.full(
            (period - 1,), data[0].item(), device=self.device, dtype=self.dtype
        )
        padded_data = torch.cat([padding, data])

        # Calculate EMA using convolution
        ema_values = torch.nn.functional.conv1d(
            padded_data.view(1, 1, -1), kernel.view(1, 1, -1), padding=period - 1
        )

        return ema_values.view(-1)[: len(data)]

    def sma(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate Simple Moving Average (SMA)

        Args:
            data: Input tensor
            period: SMA period

        Returns:
            SMA values as tensor
        """
        kernel = torch.ones(period, device=self.device, dtype=self.dtype) / period

        # Pad the input data
        padding = torch.full(
            (period - 1,), data[0].item(), device=self.device, dtype=self.dtype
        )
        padded_data = torch.cat([padding, data])

        # Calculate SMA using convolution
        sma_values = torch.nn.functional.conv1d(
            padded_data.view(1, 1, -1), kernel.view(1, 1, -1), padding=period - 1
        )

        return sma_values.view(-1)[: len(data)]

    @staticmethod
    def torch_sma(x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-accelerated Simple Moving Average"""
        if len(x) < window:
            return torch.full_like(x, torch.nan)

        return F.avg_pool1d(
            x.view(1, 1, -1), kernel_size=window, stride=1, padding=window // 2
        ).view(-1)

    @staticmethod
    def torch_ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
        """GPU-accelerated Exponential Moving Average"""
        if len(x) == 0:
            return x

        # Initialize output tensor
        result = torch.zeros_like(x)
        result[0] = x[0]  # First value is same as input

        # Calculate EMA
        for i in range(1, len(x)):
            result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    def torch_stddev(x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-accelerated Rolling Standard Deviation"""
        if len(x) < window:
            return torch.full_like(x, torch.nan)

        # Use unfold for rolling window
        rolling = x.unfold(0, window, 1)
        return torch.std(rolling, dim=1)

    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate indicator signals. Must be implemented by child classes.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary of calculated signals
        """
        raise NotImplementedError("Subclasses must implement calculate_signals")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the indicator. Must be implemented by child classes.

        Args:
            x: Input tensor

        Returns:
            Dictionary of calculated values
        """
        raise NotImplementedError("Subclasses must implement forward")

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Main calculation method that handles data conversion and processing

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary of calculated values as pandas Series
        """
        try:
            # Convert to tensors and calculate
            with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
                signals = self.calculate_signals(data)

            # Convert back to pandas
            return {
                k: pd.Series(v.cpu().numpy(), index=data.index)
                for k, v in signals.items()
            }

        except Exception as e:
            print(f"Error in indicator calculation: {str(e)}")
            return {}
