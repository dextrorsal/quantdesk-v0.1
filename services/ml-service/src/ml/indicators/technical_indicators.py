"""
Technical Indicators Implementation

This module implements common technical indicators using the PyTorch backend
for GPU acceleration and efficient calculation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig

class RSIIndicator(BaseTorchIndicator):
    """Relative Strength Index (RSI) indicator"""
    
    def __init__(self, period: int = 14, config: Optional[TorchIndicatorConfig] = None):
        super().__init__(config)
        self.period = period
        
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Calculate RSI signal"""
        close = self.to_tensor(data['close'])
        
        # Calculate price changes
        delta = torch.zeros_like(close)
        delta[1:] = close[1:] - close[:-1]
        
        # Separate gains and losses
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # Calculate average gains and losses
        avg_gain = self.torch_ema(gains, 1/self.period)
        avg_loss = self.torch_ema(losses, 1/self.period)
        
        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-10)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return {'rsi': rsi}
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for RSI calculation"""
        # Calculate price changes
        delta = torch.zeros_like(x)
        delta[1:] = x[1:] - x[:-1]
        
        # Separate gains and losses
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # Calculate average gains and losses
        avg_gain = self.torch_ema(gains, 1/self.period)
        avg_loss = self.torch_ema(losses, 1/self.period)
        
        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return {'rsi': rsi}

class MACDIndicator(BaseTorchIndicator):
    """Moving Average Convergence Divergence (MACD) indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, config: Optional[TorchIndicatorConfig] = None):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Calculate MACD signals"""
        close = self.to_tensor(data['close'])
        
        # Calculate EMAs
        fast_ema = self.torch_ema(close, 2 / (self.fast_period + 1))
        slow_ema = self.torch_ema(close, 2 / (self.slow_period + 1))
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self.torch_ema(macd_line, 2 / (self.signal_period + 1))
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for MACD calculation"""
        # Calculate EMAs
        fast_ema = self.torch_ema(x, 2 / (self.fast_period + 1))
        slow_ema = self.torch_ema(x, 2 / (self.slow_period + 1))
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self.torch_ema(macd_line, 2 / (self.signal_period + 1))
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }

class BollingerBandsIndicator(BaseTorchIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, num_std: float = 2.0, 
                 config: Optional[TorchIndicatorConfig] = None):
        super().__init__(config)
        self.period = period
        self.num_std = num_std
        
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Calculate Bollinger Bands signals"""
        close = self.to_tensor(data['close'])
        
        # Calculate middle band (SMA)
        middle_band = self.torch_sma(close, self.period)
        
        # Calculate standard deviation
        std = self.torch_stddev(close, self.period)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        return {
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for Bollinger Bands calculation"""
        # Calculate middle band (SMA)
        middle_band = self.torch_sma(x, self.period)
        
        # Calculate standard deviation
        std = self.torch_stddev(x, self.period)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        return {
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band
        }

class WaveTrendIndicator(BaseTorchIndicator):
    """WaveTrend Oscillator indicator"""
    
    def __init__(self, channel_len: int = 10, avg_len: int = 21, 
                 config: Optional[TorchIndicatorConfig] = None):
        super().__init__(config)
        self.channel_len = channel_len
        self.avg_len = avg_len
        
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Calculate WaveTrend signals"""
        # Get data
        high = self.to_tensor(data['high'])
        low = self.to_tensor(data['low'])
        close = self.to_tensor(data['close'])
        
        # Calculate TP (typical price)
        tp = (high + low + close) / 3.0
        
        # Calculate EMA of TP
        esa = self.torch_ema(tp, 2 / (self.channel_len + 1))
        
        # Calculate absolute difference between TP and EMA
        d = torch.abs(tp - esa)
        
        # Calculate EMA of d
        asd = self.torch_ema(d, 2 / (self.channel_len + 1))
        
        # Calculate CI (Channel Index)
        ci = (tp - esa) / (0.015 * asd + 1e-10)
        
        # Calculate WaveTrend values
        wt1 = self.torch_ema(ci, 2 / (self.avg_len + 1))
        wt2 = self.torch_sma(wt1, 4)
        
        return {
            'wt1': wt1,
            'wt2': wt2
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for WaveTrend calculation"""
        # Use input as typical price
        tp = x
        
        # Calculate EMA of TP
        esa = self.torch_ema(tp, 2 / (self.channel_len + 1))
        
        # Calculate absolute difference between TP and EMA
        d = torch.abs(tp - esa)
        
        # Calculate EMA of d
        asd = self.torch_ema(d, 2 / (self.channel_len + 1))
        
        # Calculate CI (Channel Index)
        ci = (tp - esa) / (0.015 * asd + 1e-10)
        
        # Calculate WaveTrend values
        wt1 = self.torch_ema(ci, 2 / (self.avg_len + 1))
        wt2 = self.torch_sma(wt1, 4)
        
        return {
            'wt1': wt1,
            'wt2': wt2
        }

class ADXIndicator(BaseTorchIndicator):
    """Average Directional Index (ADX) indicator"""
    
    def __init__(self, period: int = 14, config: Optional[TorchIndicatorConfig] = None):
        super().__init__(config)
        self.period = period
        
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Calculate ADX signals"""
        # Get data
        high = self.to_tensor(data['high'])
        low = self.to_tensor(data['low'])
        close = self.to_tensor(data['close'])
        
        # Prepare previous close
        prev_close = torch.zeros_like(close)
        prev_close[1:] = close[:-1]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = torch.abs(high - prev_close)
        tr3 = torch.abs(low - prev_close)
        tr = torch.max(torch.max(tr1, tr2), tr3)
        
        # Calculate directional movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        # Pad to match original length
        up_move = torch.cat([torch.zeros(1, device=self.device), up_move])
        down_move = torch.cat([torch.zeros(1, device=self.device), down_move])
        
        # Get positive and negative directional movements
        pos_dm = torch.where((up_move > down_move) & (up_move > 0), up_move, torch.zeros_like(up_move))
        neg_dm = torch.where((down_move > up_move) & (down_move > 0), down_move, torch.zeros_like(down_move))
        
        # Smooth the values
        smoothed_tr = self.torch_ema(tr, 2 / (self.period + 1))
        smoothed_pos_dm = self.torch_ema(pos_dm, 2 / (self.period + 1))
        smoothed_neg_dm = self.torch_ema(neg_dm, 2 / (self.period + 1))
        
        # Calculate directional indicators
        pdi = 100 * smoothed_pos_dm / (smoothed_tr + 1e-10)
        ndi = 100 * smoothed_neg_dm / (smoothed_tr + 1e-10)
        
        # Calculate directional index
        dx = 100 * torch.abs(pdi - ndi) / (pdi + ndi + 1e-10)
        
        # Calculate ADX
        adx = self.torch_ema(dx, 2 / (self.period + 1))
        
        return {
            'adx': adx,
            'pdi': pdi,
            'ndi': ndi
        }
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for ADX calculation - x should be dict with high, low, close"""
        high = x['high']
        low = x['low']
        close = x['close']
        
        # Prepare previous close
        prev_close = torch.zeros_like(close)
        prev_close[1:] = close[:-1]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = torch.abs(high - prev_close)
        tr3 = torch.abs(low - prev_close)
        tr = torch.max(torch.max(tr1, tr2), tr3)
        
        # Calculate directional movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        # Pad to match original length
        up_move = torch.cat([torch.zeros(1, device=self.device), up_move])
        down_move = torch.cat([torch.zeros(1, device=self.device), down_move])
        
        # Get positive and negative directional movements
        pos_dm = torch.where((up_move > down_move) & (up_move > 0), up_move, torch.zeros_like(up_move))
        neg_dm = torch.where((down_move > up_move) & (down_move > 0), down_move, torch.zeros_like(down_move))
        
        # Smooth the values
        smoothed_tr = self.torch_ema(tr, 2 / (self.period + 1))
        smoothed_pos_dm = self.torch_ema(pos_dm, 2 / (self.period + 1))
        smoothed_neg_dm = self.torch_ema(neg_dm, 2 / (self.period + 1))
        
        # Calculate directional indicators
        pdi = 100 * smoothed_pos_dm / (smoothed_tr + 1e-10)
        ndi = 100 * smoothed_neg_dm / (smoothed_tr + 1e-10)
        
        # Calculate directional index
        dx = 100 * torch.abs(pdi - ndi) / (pdi + ndi + 1e-10)
        
        # Calculate ADX
        adx = self.torch_ema(dx, 2 / (self.period + 1))
        
        return {
            'adx': adx,
            'pdi': pdi,
            'ndi': ndi
        }

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all indicators and add them to the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added indicator columns
    """
    # Initialize technical indicators
    config = TorchIndicatorConfig()
    rsi = RSIIndicator(period=14, config=config)
    macd = MACDIndicator(config=config)
    bb = BollingerBandsIndicator(config=config)
    wt = WaveTrendIndicator(config=config)
    adx = ADXIndicator(config=config)
    
    # Calculate indicators
    rsi_values = rsi.calculate(df)
    macd_values = macd.calculate(df)
    bb_values = bb.calculate(df)
    wt_values = wt.calculate(df)
    adx_values = adx.calculate(df)
    
    # Add to dataframe
    df_with_indicators = df.copy()
    
    # Add RSI
    df_with_indicators['rsi'] = rsi_values.get('rsi', pd.Series(np.nan, index=df.index))
    
    # Add MACD
    df_with_indicators['macd'] = macd_values.get('macd', pd.Series(np.nan, index=df.index))
    df_with_indicators['macd_signal'] = macd_values.get('macd_signal', pd.Series(np.nan, index=df.index))
    df_with_indicators['macd_hist'] = macd_values.get('macd_hist', pd.Series(np.nan, index=df.index))
    
    # Add Bollinger Bands
    df_with_indicators['bb_middle'] = bb_values.get('bb_middle', pd.Series(np.nan, index=df.index))
    df_with_indicators['bb_upper'] = bb_values.get('bb_upper', pd.Series(np.nan, index=df.index))
    df_with_indicators['bb_lower'] = bb_values.get('bb_lower', pd.Series(np.nan, index=df.index))
    
    # Add WaveTrend
    df_with_indicators['wt1'] = wt_values.get('wt1', pd.Series(np.nan, index=df.index))
    df_with_indicators['wt2'] = wt_values.get('wt2', pd.Series(np.nan, index=df.index))
    
    # Add ADX
    df_with_indicators['adx'] = adx_values.get('adx', pd.Series(np.nan, index=df.index))
    df_with_indicators['pdi'] = adx_values.get('pdi', pd.Series(np.nan, index=df.index))
    df_with_indicators['ndi'] = adx_values.get('ndi', pd.Series(np.nan, index=df.index))
    
    # Add ATR (placeholder for now, using TR from ADX)
    # In a real system, you'd implement a proper ATR calculation
    df_with_indicators['atr'] = pd.Series(np.nan, index=df.index)
    
    # Calculate Stochastic (simple implementation)
    # This would normally use the BaseTorchIndicator as well
    high_max = df['high'].rolling(window=14).max()
    low_min = df['low'].rolling(window=14).min()
    
    df_with_indicators['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
    df_with_indicators['stoch_d'] = df_with_indicators['stoch_k'].rolling(window=3).mean()
    
    # Calculate CCI (placeholder)
    df_with_indicators['cci'] = pd.Series(np.nan, index=df.index)
    
    return df_with_indicators 