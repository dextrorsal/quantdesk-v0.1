"""
Custom Signal Filters for the Paper Trading Framework

This module provides custom filters that can be applied to trading signals
to improve performance. These filters can be used to remove noise, identify
stronger trends, and apply risk management rules to signals.

Usage:
    from src.ml.custom_filters import sol_conservative_filter
    
    # In your backtesting code:
    filtered_signals = sol_conservative_filter(
        signals=raw_signals, 
        data=market_data,
        min_trend_strength=0.1,
        signal_threshold=0.7
    )
"""

import numpy as np
import pandas as pd


def sol_conservative_filter(signals: np.ndarray, data: pd.DataFrame, 
                          min_trend_strength: float = 0.1, 
                          signal_threshold: float = 0.7) -> np.ndarray:
    """
    Apply conservative filtering to SOL trading signals
    
    This filter implements the following rules:
    1. Only trade when trend strength exceeds threshold
    2. Apply higher confidence threshold for signals
    3. Filter signals based on key moving averages
    4. Use ATR for volatility-based filtering
    
    Args:
        signals: Raw trading signals array
        data: Market data DataFrame with technical indicators
        min_trend_strength: Minimum trend strength threshold
        signal_threshold: Minimum signal strength threshold
        
    Returns:
        Filtered signals array
    """
    filtered_signals = signals.copy()
    
    # Add technical indicators if they don't exist
    if 'sma20' not in data.columns:
        data['sma20'] = data['close'].rolling(window=20).mean()
    
    if 'atr14' not in data.columns:
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift(1)).abs()
        low_close_prev = (data['low'] - data['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr14'] = true_range.rolling(window=14).mean()
    
    if 'trend_strength' not in data.columns:
        data['trend_strength'] = ((data['close'] - data['open']) / data['close']).abs()
    
    # Fill missing values
    data.fillna(method='bfill', inplace=True)
    
    # Apply minimum trend strength filter
    trend_strength = data['trend_strength'].values
    filtered_signals[trend_strength < min_trend_strength] = 0
    
    # Apply higher threshold to signals (more conservative)
    signal_strength = np.abs(signals)
    filtered_signals[signal_strength < signal_threshold] = 0
    
    # Only take long signals when price is above 20 SMA
    close = data['close'].values
    sma20 = data['sma20'].values
    long_mask = (filtered_signals > 0) & (close < sma20)
    filtered_signals[long_mask] = 0
    
    # Only take short signals when price is below 20 SMA
    short_mask = (filtered_signals < 0) & (close > sma20)
    filtered_signals[short_mask] = 0
    
    # Filter out signals during excessive volatility
    atr14 = data['atr14'].values
    atr14_mean = pd.Series(atr14).rolling(window=100).mean().fillna(method='bfill').values
    high_volatility = atr14 > (atr14_mean * 2)
    filtered_signals[high_volatility] = 0
    
    return filtered_signals


def apply_signal_filter(signals: np.ndarray, data: pd.DataFrame, 
                      filter_name: str = None, **filter_params) -> np.ndarray:
    """
    Apply a named signal filter to trading signals
    
    Args:
        signals: Raw trading signals array
        data: Market data DataFrame with technical indicators
        filter_name: Name of the filter to apply
        filter_params: Parameters to pass to the filter
        
    Returns:
        Filtered signals array
    """
    # Select the appropriate filter based on name
    if filter_name == 'sol_conservative':
        return sol_conservative_filter(signals, data, **filter_params)
    else:
        # Default: return original signals
        return signals
