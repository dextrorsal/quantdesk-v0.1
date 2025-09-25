import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class DataProcessor:
    def __init__(self, timeframe: str):
        """
        Initialize DataProcessor with timeframe settings
        
        Args:
            timeframe (str): Chart timeframe (e.g., '15m', '1h', '4h')
        """
        self.timeframe = timeframe
        self.lookback_days = self._set_lookback_period()
    
    def _set_lookback_period(self) -> int:
        """
        Set lookback period based on timeframe
        - Sub-30min charts: 10-14 days
        - Above 30min charts: 30-90 days
        """
        timeframe_minutes = self._convert_timeframe_to_minutes()
        return 14 if timeframe_minutes < 30 else 90
    
    def _convert_timeframe_to_minutes(self) -> int:
        """Convert timeframe string to minutes"""
        unit = self.timeframe[-1].lower()
        value = int(self.timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for model training
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Processed data with features
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        df = df.copy()
        
        # Calculate range metrics
        df['daily_range'] = df['high'] - df['low']
        df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action features
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['low']
        
        return df.dropna()

    def detect_double_bottom(self, df: pd.DataFrame, tolerance: float = 0.02) -> pd.Series:
        """
        Detect potential double bottom patterns
        
        Args:
            df (pd.DataFrame): Price data
            tolerance (float): Price tolerance for pattern detection
            
        Returns:
            pd.Series: Boolean series indicating double bottom patterns
        """
        lows = df['low'].rolling(window=5, center=True).min()
        potential_bottoms = (df['low'] == lows)
        
        double_bottoms = pd.Series(False, index=df.index)
        
        for i in range(len(df)-20):
            if potential_bottoms.iloc[i]:
                # Look for second bottom within tolerance
                for j in range(i+5, min(i+20, len(df))):
                    if (potential_bottoms.iloc[j] and 
                        abs(df['low'].iloc[i] - df['low'].iloc[j]) <= 
                        df['low'].iloc[i] * tolerance):
                        double_bottoms.iloc[j] = True
                        
        return double_bottoms 