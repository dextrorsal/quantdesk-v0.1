"""
TradingView-Style Logistic Regression with PyTorch Acceleration

This module implements a classification-based Logistic Regression algorithm for trading,
similar to TradingView's popular Logistic Regression indicator but with GPU acceleration.

Key concepts:
- Classification algorithm that separates price action into BUY/SELL signals
- S-shaped sigmoid curve to better separate data points
- Gradient descent for finding optimal parameters (weights)
- Normalization for better prediction stability
- Optional filters for volatility and volume

GPU acceleration enables much faster training iterations and real-time signal generation
compared to the original TradingView implementation.

Example:
    >>> model = LogisticRegression(lookback=3, learning_rate=0.0009)
    >>> signals = model.calculate_signals(df)
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional
from dataclasses import dataclass

from contextlib import nullcontext


@dataclass
class LogisticConfig:
    """Configuration for TradingView-style logistic regression model"""

    lookback: int = 3  # Lookback window size (TradingView default)
    learning_rate: float = 0.0009  # Learning rate for gradient descent
    iterations: int = 1000  # Training iterations
    norm_lookback: int = 20  # Lookback for normalization
    use_amp: bool = False  # Use automatic mixed precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    volatility_filter: bool = True  # Filter signals with volatility
    volume_filter: bool = True  # Filter signals with volume
    threshold: float = 0.5  # Signal threshold
    use_price_data: bool = True  # Use price directly for signal generation
    holding_period: int = 5  # Number of bars to hold a position


@dataclass
class TradingMetrics:
    """Container for trading metrics"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    cumulative_return: float = 0.0
    win_loss_ratio: float = 0.0
    profit_factor: float = 0.0


def minimax_scale(
    value: float,
    data_window: np.ndarray,
    target_min: float = None,
    target_max: float = None,
) -> float:
    """
    Scale value to window's min-max range with TradingView-like normalization

    Args:
        value: Value to scale
        data_window: Window of data to use for scaling
        target_min: Optional target minimum value
        target_max: Optional target maximum value

    Returns:
        Scaled value between target_min and target_max
    """
    try:
        window_min = np.min(data_window)
        window_max = np.max(data_window)

        if np.isclose(window_max, window_min):
            return value

        # If target range not specified, keep original range
        if target_min is None:
            target_min = window_min
        if target_max is None:
            target_max = window_max

        return (target_max - target_min) * (value - window_min) / (
            window_max - window_min
        ) + target_min
    except Exception as e:
        print(f"Error in minimax scaling: {str(e)}")
        return value


class LogisticRegression:
    """
    TradingView-style Logistic Regression implementation with PyTorch acceleration.
    This implementation follows the approach from TradingView's popular indicator.
    """

    def __init__(self, config: Optional[LogisticConfig] = None):
        """
        Initialize the TradingView-style logistic regression model

        Args:
            config: Model configuration parameters
        """
        self.config = config or LogisticConfig()
        self.metrics = TradingMetrics()
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype

        # Signal constants
        self.BUY = 1
        self.SELL = -1
        self.HOLD = 0

        # State variables
        self.last_signal = self.HOLD
        self.holding_counter = 0
        self.last_price = 0.0

    def to_tensor(self, data):
        """Convert numpy array to PyTorch tensor"""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.dtype)
        return torch.tensor(data, device=self.device, dtype=self.dtype)

    def sigmoid(self, z):
        """Compute sigmoid function with PyTorch"""
        return torch.sigmoid(z)

    def dot_product(self, v, w):
        """Compute dot product with proper broadcasting"""
        return torch.sum(v * w)

    def logistic_regression(self, X, Y, lookback, lr, iterations):
        """
        PyTorch implementation of TradingView's logistic regression algorithm

        Args:
            X: Features tensor
            Y: Target tensor
            lookback: Lookback window size
            lr: Learning rate
            iterations: Number of training iterations

        Returns:
            Tuple of (loss, prediction)
        """
        # Convert to tensors if needed
        X = self.to_tensor(X)
        Y = self.to_tensor(Y)

        # Initialize weight
        w = torch.zeros(1, device=self.device, dtype=self.dtype)

        # Gradient descent loop
        for i in range(iterations):
            with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
                # Forward pass
                z = X * w
                hypothesis = self.sigmoid(z)

                # Compute loss (binary cross-entropy)
                eps = 1e-7  # small value to avoid log(0)
                loss = -torch.mean(
                    Y * torch.log(hypothesis + eps)
                    + (1.0 - Y) * torch.log(1.0 - hypothesis + eps)
                )

                # Compute gradient
                gradient = torch.mean(X * (hypothesis - Y))

                # Update weights
                w = w - lr * gradient

        # Final prediction
        final_pred = self.sigmoid(X[-1] * w)

        return loss.item(), final_pred.item()

    def volatility_break(self, df, i):
        """Check if volatility is increasing (ATR1 > ATR10)"""
        if "atr1" in df.columns and "atr10" in df.columns:
            return df["atr1"].iloc[i] > df["atr10"].iloc[i]
        elif "atr" in df.columns:
            # If only one ATR column, compare with shifted version
            return df["atr"].iloc[i] > df["atr"].iloc[max(0, i - 9)]
        else:
            # Calculate ATR on the fly if not available
            tr = max(
                df["high"].iloc[i] - df["low"].iloc[i],
                abs(df["high"].iloc[i] - df["close"].iloc[max(0, i - 1)]),
                abs(df["low"].iloc[i] - df["close"].iloc[max(0, i - 1)]),
            )
            tr_10 = (
                sum(
                    [
                        max(
                            df["high"].iloc[max(0, i - j)]
                            - df["low"].iloc[max(0, i - j)],
                            abs(
                                df["high"].iloc[max(0, i - j)]
                                - df["close"].iloc[max(0, i - j - 1)]
                            ),
                            abs(
                                df["low"].iloc[max(0, i - j)]
                                - df["close"].iloc[max(0, i - j - 1)]
                            ),
                        )
                        for j in range(10)
                    ]
                )
                / 10
            )
            return tr > tr_10

    def volume_break(self, df, i):
        """Check if volume RSI is above threshold (49)"""
        if "volume_rsi" in df.columns:
            return df["volume_rsi"].iloc[i] > 49
        elif "volume" in df.columns:
            # Calculate a simple volume ratio if RSI not available
            recent_vol = df["volume"].iloc[max(0, i - 13) : i + 1].mean()
            older_vol = df["volume"].iloc[max(0, i - 27) : max(0, i - 13)].mean()
            return recent_vol > older_vol
        return True  # Default to True if no volume data

    def prepare_data(self, df):
        """
        Prepare data for logistic regression processing

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of price, base and synthetic datasets
        """
        # Use close price by default, could be extended to support other price types
        price = df["close"].values

        # Generate synthetic dataset similar to TradingView implementation
        synth_data = np.log(np.abs(np.power(price, 2) - 1) + 0.5)

        return price, price, synth_data

    def calculate_signals(self, df):
        """
        Calculate trading signals using TradingView-style logistic regression

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with signals and predictions
        """
        lookback = self.config.lookback
        lr = self.config.learning_rate
        iterations = self.config.iterations
        norm_lookback = self.config.norm_lookback

        # Initialize result arrays
        signals = np.zeros(len(df))
        losses = np.zeros(len(df))
        predictions = np.zeros(len(df))

        # Prepare price and synthetic data
        price, base_data, synth_data = self.prepare_data(df)

        # Trading state variables
        current_signal = self.HOLD
        holding_counter = 0

        # Process each bar (starting after lookback period)
        for i in range(lookback, len(df)):
            try:
                # Prepare data windows
                X = base_data[i - lookback : i]
                Y = synth_data[i - lookback : i]

                # Run logistic regression
                loss, pred = self.logistic_regression(X, Y, lookback, lr, iterations)

                # Scale values to price range for better visualization
                if i >= lookback + 2:
                    min_price = np.min(price[max(0, i - norm_lookback) : i])
                    max_price = np.max(price[max(0, i - norm_lookback) : i])

                    scaled_loss = minimax_scale(
                        loss, price[max(0, i - norm_lookback) : i], min_price, max_price
                    )
                    scaled_pred = minimax_scale(
                        pred, price[max(0, i - norm_lookback) : i], min_price, max_price
                    )

                    # Store for visualization
                    losses[i] = scaled_loss
                    predictions[i] = scaled_pred

                    # Calculate signal based on TradingView's approach
                    # Check filters
                    passes_filter = True
                    if self.config.volatility_filter:
                        passes_filter = passes_filter and self.volatility_break(df, i)
                    if self.config.volume_filter:
                        passes_filter = passes_filter and self.volume_break(df, i)

                    # Generate signal using improved logic
                    # Calculate relative position of price vs loss/prediction
                    price_ratio = (price[i] - scaled_loss) / (scaled_loss + 1e-7)
                    pred_ratio = (scaled_pred - scaled_loss) / (scaled_loss + 1e-7)
                    
                    # More balanced signal generation
                    if self.config.use_price_data:
                        # Use price momentum and position relative to loss
                        price_change = (price[i] - price[i-1]) / price[i-1] if i > 0 else 0
                        
                        if passes_filter:
                            # Strong upward momentum or price well above loss line
                            if (price_ratio > 0.01 and price_change > 0.001) or price_ratio > 0.02:
                                new_signal = self.BUY
                            # Strong downward momentum or price well below loss line  
                            elif (price_ratio < -0.01 and price_change < -0.001) or price_ratio < -0.02:
                                new_signal = self.SELL
                            # Hold current signal if not strong enough signal
                            else:
                                new_signal = current_signal
                        else:
                            new_signal = current_signal
                    else:
                        # Crossover method with improved logic
                        if passes_filter:
                            if pred_ratio > 0.01 and scaled_pred > scaled_loss:
                                new_signal = self.BUY
                            elif pred_ratio < -0.01 and scaled_pred < scaled_loss:
                                new_signal = self.SELL
                            else:
                                new_signal = current_signal
                        else:
                            new_signal = current_signal

                    # Simplified holding period logic - force signal changes
                    if new_signal != current_signal:
                        holding_counter = 0
                        current_signal = new_signal
                    else:
                        holding_counter += 1
                        
                        # Force exit after holding period to allow new signals
                        if holding_counter >= self.config.holding_period:
                            if current_signal == self.BUY:
                                new_signal = self.SELL
                            elif current_signal == self.SELL:  
                                new_signal = self.BUY
                            else:
                                new_signal = self.HOLD
                            holding_counter = 0
                            current_signal = new_signal

                    signals[i] = new_signal
                    current_signal = new_signal

                    # Update metrics when signal changes
                    if i > lookback + 2 and signals[i] != signals[i - 1]:
                        self.update_metrics(price[i], signals[i], signals[i - 1])

            except Exception as e:
                print(f"Error in signal generation at bar {i}: {str(e)}")
                signals[i] = current_signal

        # Create result dataframe
        result = pd.DataFrame(
            {
                "close": df["close"],
                "signal": signals,
                "loss": losses,
                "prediction": predictions,
            }
        )

        # Convert to dictionary format with the expected keys
        buy_signals = np.zeros(len(df), dtype=bool)
        sell_signals = np.zeros(len(df), dtype=bool)

        # Convert signal values to boolean arrays
        buy_signals[signals == self.BUY] = True
        sell_signals[signals == self.SELL] = True

        # Convert numpy arrays to torch tensors to match expected format
        return {
            "buy_signals": torch.tensor(
                buy_signals, device=self.device, dtype=torch.bool
            ),
            "sell_signals": torch.tensor(
                sell_signals, device=self.device, dtype=torch.bool
            ),
            "predictions": torch.tensor(
                predictions, device=self.device, dtype=self.dtype
            ),
            "long_signals": torch.tensor(
                buy_signals, device=self.device, dtype=torch.bool
            ),
            "short_signals": torch.tensor(
                sell_signals, device=self.device, dtype=torch.bool
            ),
            "dataframe": result,
        }

    def update_metrics(self, current_price, signal, last_signal):
        """
        Update trading metrics based on signals

        Args:
            current_price: Current price
            signal: Current signal
            last_signal: Previous signal

        Returns:
            Updated metrics
        """
        # Skip if we don't have a previous price
        if self.last_price == 0:
            self.last_price = current_price
            return

        # Calculate P&L based on the last signal
        pnl = 0
        if last_signal == self.BUY and signal != self.BUY:
            # Close a long position
            pnl = current_price - self.last_price
            self.metrics.total_trades += 1
            if pnl > 0:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1

        elif last_signal == self.SELL and signal != self.SELL:
            # Close a short position
            pnl = self.last_price - current_price
            self.metrics.total_trades += 1
            if pnl > 0:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1

        # Update cumulative return
        if pnl != 0 and self.last_price != 0:
            self.metrics.cumulative_return += pnl / self.last_price * 100

        # Update win rate and win/loss ratio
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = (
                self.metrics.winning_trades / self.metrics.total_trades
            )
            if self.metrics.losing_trades > 0:
                self.metrics.win_loss_ratio = (
                    self.metrics.winning_trades / self.metrics.losing_trades
                )
            else:
                self.metrics.win_loss_ratio = float("inf")  # All wins, no losses

        # Update last price
        self.last_price = current_price

    def get_metrics(self):
        """Get current trading metrics"""
        return {
            "total_trades": self.metrics.total_trades,
            "winning_trades": self.metrics.winning_trades,
            "losing_trades": self.metrics.losing_trades,
            "win_rate": self.metrics.win_rate,
            "win_loss_ratio": self.metrics.win_loss_ratio,
            "cumulative_return": self.metrics.cumulative_return,
        }

    def plot_signals(self, df, result):
        """
        Plot trading signals and prediction curves

        Args:
            df: Original price dataframe
            result: DataFrame with signals and predictions
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

            # Plot price and signals
            ax1.plot(df.index, df["close"], label="Price", alpha=0.7)

            # Plot buy and sell signals
            buy_points = df.index[result["signal"] == self.BUY]
            sell_points = df.index[result["signal"] == self.SELL]

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

            ax1.set_title("Price with Logistic Regression Signals")
            ax1.legend()

            # Plot prediction curves
            ax2.plot(df.index, result["loss"], label="Loss", color="blue")
            ax2.plot(df.index, result["prediction"], label="Prediction", color="lime")
            ax2.set_title("TradingView-Style Logistic Regression Curves")
            ax2.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting signals: {str(e)}")


# Example usage:
# model = LogisticRegression(lookback=3, learning_rate=0.0009)
# result = model.calculate_signals(df)
# model.plot_signals(df, result)
