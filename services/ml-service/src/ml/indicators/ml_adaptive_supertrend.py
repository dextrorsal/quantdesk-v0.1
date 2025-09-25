"""
Machine Learning Adaptive SuperTrend Indicator

This module implements the Machine Learning Adaptive SuperTrend indicator
based on the TradingView script by AlgoAlpha. It uses K-means clustering
to dynamically adapt SuperTrend parameters based on market volatility.

Original TradingView Script: https://www.tradingview.com/v/CLk71Qgy/
Author: AlgoAlpha
License: Mozilla Public License 2.0

Key Features:
- K-means clustering for volatility classification (High/Medium/Low)
- Dynamic SuperTrend factor adaptation
- Real-time volatility assessment
- TradingView-compatible parameters
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans

from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig


@dataclass
class MLAdaptiveSupertrendConfig(TorchIndicatorConfig):
    """Configuration for ML Adaptive SuperTrend indicator"""
    
    atr_length: int = 10
    supertrend_factor: float = 3.0
    training_data_period: int = 100
    high_volatility_percentile: float = 0.75
    medium_volatility_percentile: float = 0.5
    low_volatility_percentile: float = 0.25
    max_iterations: int = 100
    convergence_threshold: float = 1e-6


class MLAdaptiveSupertrendIndicator(BaseTorchIndicator):
    """
    Machine Learning Adaptive SuperTrend indicator implementation.
    
    This indicator uses K-means clustering to classify market volatility
    into three levels (High, Medium, Low) and dynamically adjusts the
    SuperTrend factor accordingly.
    
    The algorithm:
    1. Calculates ATR over a training period
    2. Uses K-means clustering to classify volatility levels
    3. Assigns current volatility to a cluster
    4. Applies cluster-specific SuperTrend factor
    5. Generates buy/sell signals based on trend direction
    """
    
    def __init__(
        self,
        atr_length: int = 10,
        supertrend_factor: float = 3.0,
        training_data_period: int = 100,
        high_volatility_percentile: float = 0.75,
        medium_volatility_percentile: float = 0.5,
        low_volatility_percentile: float = 0.25,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[MLAdaptiveSupertrendConfig] = None,
    ):
        """
        Initialize ML Adaptive SuperTrend indicator
        
        Args:
            atr_length: ATR calculation period
            supertrend_factor: Base SuperTrend factor
            training_data_period: Period for volatility training
            high_volatility_percentile: Initial high volatility guess
            medium_volatility_percentile: Initial medium volatility guess
            low_volatility_percentile: Initial low volatility guess
            max_iterations: Maximum K-means iterations
            convergence_threshold: Convergence threshold for K-means
            device: Device to use for computations
            dtype: Data type to use for computations
            config: Configuration object
        """
        if config is None:
            config = MLAdaptiveSupertrendConfig(
                atr_length=atr_length,
                supertrend_factor=supertrend_factor,
                training_data_period=training_data_period,
                high_volatility_percentile=high_volatility_percentile,
                medium_volatility_percentile=medium_volatility_percentile,
                low_volatility_percentile=low_volatility_percentile,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                device=device,
                dtype=dtype,
            )
        super().__init__(config)
        
        self._atr_length = config.atr_length
        self._supertrend_factor = config.supertrend_factor
        self._training_data_period = config.training_data_period
        self._high_vol_percentile = config.high_volatility_percentile
        self._medium_vol_percentile = config.medium_volatility_percentile
        self._low_vol_percentile = config.low_volatility_percentile
        self._max_iterations = config.max_iterations
        self._convergence_threshold = config.convergence_threshold
        
        # Volatility clusters
        self._high_volatility_centroid = None
        self._medium_volatility_centroid = None
        self._low_volatility_centroid = None
        self._cluster_sizes = [0, 0, 0]
        self._current_cluster = None
        self._is_trained = False
    
    @property
    def atr_length(self) -> int:
        """Get ATR length"""
        return self._atr_length
    
    @property
    def supertrend_factor(self) -> float:
        """Get SuperTrend factor"""
        return self._supertrend_factor
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
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
        atr = self.torch_ema(tr, alpha=2.0 / (self._atr_length + 1))
        
        return atr
    
    def train_volatility_clusters(self, atr_values: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Train K-means clustering on volatility data
        
        Args:
            atr_values: ATR values for training
            
        Returns:
            Tuple of (high_centroid, medium_centroid, low_centroid, iterations)
        """
        if len(atr_values) < self._training_data_period:
            # Not enough data for training
            return None, None, None, 0
        
        # Use the last training_data_period values
        training_data = atr_values[-self._training_data_period:].reshape(-1, 1)
        
        # Initialize centroids based on percentiles
        upper = np.max(training_data)
        lower = np.min(training_data)
        
        high_vol = lower + (upper - lower) * self._high_vol_percentile
        medium_vol = lower + (upper - lower) * self._medium_vol_percentile
        low_vol = lower + (upper - lower) * self._low_vol_percentile
        
        # Initialize centroids
        centroids = np.array([[high_vol], [medium_vol], [low_vol]])
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=3,
            init=centroids,
            n_init=1,
            max_iter=self._max_iterations,
            tol=self._convergence_threshold,
            random_state=42
        )
        
        kmeans.fit(training_data)
        
        # Get final centroids and cluster sizes
        final_centroids = kmeans.cluster_centers_.flatten()
        cluster_labels = kmeans.labels_
        
        # Sort centroids by value (low, medium, high)
        sorted_indices = np.argsort(final_centroids)
        sorted_centroids = final_centroids[sorted_indices]
        
        # Calculate cluster sizes
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(3)]
        sorted_sizes = [cluster_sizes[i] for i in sorted_indices]
        
        self._low_volatility_centroid = sorted_centroids[0]
        self._medium_volatility_centroid = sorted_centroids[1]
        self._high_volatility_centroid = sorted_centroids[2]
        self._cluster_sizes = sorted_sizes
        self._is_trained = True
        
        return self._high_volatility_centroid, self._medium_volatility_centroid, self._low_volatility_centroid, kmeans.n_iter_
    
    def classify_current_volatility(self, current_atr: float) -> int:
        """
        Classify current volatility into a cluster
        
        Args:
            current_atr: Current ATR value
            
        Returns:
            Cluster index (0=high, 1=medium, 2=low)
        """
        if not self._is_trained:
            return 1  # Default to medium
        
        # Calculate distances to each centroid
        distances = [
            abs(current_atr - self._high_volatility_centroid),
            abs(current_atr - self._medium_volatility_centroid),
            abs(current_atr - self._low_volatility_centroid)
        ]
        
        # Return cluster with minimum distance
        return np.argmin(distances)
    
    def get_adaptive_factor(self, cluster: int) -> float:
        """
        Get adaptive SuperTrend factor based on volatility cluster
        
        Args:
            cluster: Volatility cluster (0=high, 1=medium, 2=low)
            
        Returns:
            Adaptive SuperTrend factor
        """
        if cluster == 0:  # High volatility
            return self._supertrend_factor * 1.5  # More sensitive
        elif cluster == 1:  # Medium volatility
            return self._supertrend_factor  # Standard
        else:  # Low volatility
            return self._supertrend_factor * 0.6  # Less sensitive
    
    def calculate_supertrend(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, adaptive_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate SuperTrend with adaptive factor
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            adaptive_factor: Adaptive SuperTrend factor
            
        Returns:
            Tuple of (supertrend, direction)
        """
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)
        
        # Calculate basic bands
        hl2 = (high + low) / 2.0
        basic_upper_band = hl2 + (adaptive_factor * atr)
        basic_lower_band = hl2 - (adaptive_factor * atr)
        
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
        
        # Calculate SuperTrend and direction
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
        
        return supertrend, direction
    
    def forward(self, high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate ML Adaptive SuperTrend indicator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Dictionary with supertrend values and signals
        """
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)
        
        # Convert to numpy for training
        atr_numpy = atr.cpu().numpy()
        
        # Train volatility clusters if not trained yet
        if not self._is_trained and len(atr_numpy) >= self._training_data_period:
            self.train_volatility_clusters(atr_numpy)
        
        # Calculate SuperTrend with dynamic adaptive factors
        supertrend = torch.zeros_like(close)
        direction = torch.zeros_like(close)
        adaptive_factors = torch.zeros_like(close)
        volatility_clusters = torch.zeros_like(close, dtype=torch.long)
        
        # Calculate SuperTrend for each bar with its own adaptive factor
        for i in range(len(close)):
            if i < self._training_data_period:
                # Use base factor before training is complete
                current_factor = self._supertrend_factor
                current_cluster = 1  # Default to medium
            else:
                # Classify current volatility
                current_atr = atr_numpy[i] if i < len(atr_numpy) else atr_numpy[-1]
                current_cluster = self.classify_current_volatility(current_atr)
                current_factor = self.get_adaptive_factor(current_cluster)
            
            adaptive_factors[i] = current_factor
            volatility_clusters[i] = current_cluster
            
            # Calculate SuperTrend for this bar
            if i == 0:
                supertrend[i] = close[i]
                direction[i] = -1
            else:
                # Calculate ATR for this bar
                atr_val = atr[i].item()
                
                # Calculate basic bands
                hl2 = (high[i] + low[i]) / 2.0
                basic_upper_band = hl2 + (current_factor * atr_val)
                basic_lower_band = hl2 - (current_factor * atr_val)
                
                # Calculate final bands
                if i == 1:
                    final_upper_band = basic_upper_band
                    final_lower_band = basic_lower_band
                else:
                    prev_upper = supertrend[i-1] if direction[i-1] == -1 else supertrend[i-1]
                    prev_lower = supertrend[i-1] if direction[i-1] == 1 else supertrend[i-1]
                    
                    if basic_upper_band < prev_upper or close[i-1] > prev_upper:
                        final_upper_band = basic_upper_band
                    else:
                        final_upper_band = prev_upper
                    
                    if basic_lower_band > prev_lower or close[i-1] < prev_lower:
                        final_lower_band = basic_lower_band
                    else:
                        final_lower_band = prev_lower
                
                # Calculate SuperTrend and direction
                if close[i] <= final_upper_band:
                    supertrend[i] = final_upper_band
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower_band
                    direction[i] = 1
        
        # Set current cluster
        self._current_cluster = volatility_clusters[-1].item()
        
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
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "atr": atr,
            "adaptive_factor": adaptive_factors,
            "volatility_cluster": volatility_clusters,
            "high_volatility_centroid": torch.full_like(close, self._high_volatility_centroid or 0.0),
            "medium_volatility_centroid": torch.full_like(close, self._medium_volatility_centroid or 0.0),
            "low_volatility_centroid": torch.full_like(close, self._low_volatility_centroid or 0.0),
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate ML Adaptive SuperTrend signals from OHLCV data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with supertrend values and signals
        """
        high = self.to_tensor(data["high"])
        low = self.to_tensor(data["low"])
        close = self.to_tensor(data["close"])
        
        result = self.forward(high, low, close)
        
        # Add compatibility keys
        result["predictions"] = result["supertrend"]
        result["long_signals"] = result["buy_signals"]
        result["short_signals"] = result["sell_signals"]
        
        return result
    
    def get_cluster_info(self) -> Dict[str, any]:
        """Get current cluster information"""
        return {
            "is_trained": self._is_trained,
            "current_cluster": self._current_cluster,
            "high_volatility_centroid": self._high_volatility_centroid,
            "medium_volatility_centroid": self._medium_volatility_centroid,
            "low_volatility_centroid": self._low_volatility_centroid,
            "cluster_sizes": self._cluster_sizes,
            "cluster_names": ["High", "Medium", "Low"]
        }


def calculate_ml_adaptive_supertrend(
    df: pd.DataFrame, 
    atr_length: int = 10, 
    supertrend_factor: float = 3.0,
    training_data_period: int = 100
) -> pd.DataFrame:
    """
    Calculate ML Adaptive SuperTrend indicator for a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        atr_length: ATR calculation period
        supertrend_factor: Base SuperTrend factor
        training_data_period: Period for volatility training
        
    Returns:
        DataFrame with ML Adaptive SuperTrend columns added
    """
    indicator = MLAdaptiveSupertrendIndicator(
        atr_length=atr_length,
        supertrend_factor=supertrend_factor,
        training_data_period=training_data_period
    )
    signals = indicator.calculate_signals(df)
    
    # Convert tensors to numpy arrays
    result_df = df.copy()
    result_df["ml_supertrend"] = signals["supertrend"].cpu().numpy()
    result_df["ml_supertrend_direction"] = signals["direction"].cpu().numpy()
    result_df["ml_supertrend_buy"] = signals["buy_signals"].cpu().numpy()
    result_df["ml_supertrend_sell"] = signals["sell_signals"].cpu().numpy()
    result_df["ml_supertrend_atr"] = signals["atr"].cpu().numpy()
    result_df["ml_supertrend_factor"] = signals["adaptive_factor"].cpu().numpy()
    result_df["ml_supertrend_cluster"] = signals["volatility_cluster"].cpu().numpy()
    
    # Add cluster information
    cluster_info = indicator.get_cluster_info()
    result_df["volatility_cluster_name"] = [cluster_info["cluster_names"][c] for c in result_df["ml_supertrend_cluster"]]
    
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
    
    # Calculate ML Adaptive SuperTrend
    result = calculate_ml_adaptive_supertrend(df, atr_length=10, supertrend_factor=3.0)
    
    print("ML Adaptive SuperTrend Results:")
    print(result[["close", "ml_supertrend", "ml_supertrend_direction", "ml_supertrend_buy", "ml_supertrend_sell", "volatility_cluster_name"]])
