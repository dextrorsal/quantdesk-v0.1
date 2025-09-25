"""
Neon Data Processor - Prepares market data from Neon for ML training
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import List, Tuple, Optional
import logging
from datetime import datetime

class NeonDataProcessor:
    def __init__(self, connection_string: str):
        """
        Initialize the data processor
        
        Args:
            connection_string: Neon database connection string
        """
        self.engine = create_engine(connection_string)
        self.logger = logging.getLogger(__name__)
        
    def get_price_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get price data from Neon database
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with price data
        """
        try:
            # Build query
            query = """
                SELECT *
                FROM trading_bot.price_data
                WHERE symbol = :symbol
            """
            
            if start_date:
                query += " AND timestamp >= :start_date"
            if end_date:
                query += " AND timestamp <= :end_date"
                
            query += " ORDER BY timestamp ASC"
            
            # Execute query
            params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }
            
            df = pd.read_sql(query, self.engine, params=params)
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting price data: {str(e)}")
            raise
            
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for ML model
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with features
        """
        try:
            # Copy dataframe
            df = df.copy()
            
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            
            # Moving averages
            for window in [7, 14, 21]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                
            # Volatility
            df['volatility'] = df['returns'].rolling(14).std()
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(14).mean()
            df['volume_std'] = df['volume'].rolling(14).std()
            
            # Price ranges
            df['high_low_range'] = df['high'] - df['low']
            df['close_open_range'] = df['close'] - df['open']
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            raise
            
    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'returns',
        lookback: int = 10,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ML model training
        
        Args:
            df: DataFrame with features
            target_column: Column to predict
            lookback: Number of past time steps to use
            prediction_horizon: Number of future time steps to predict
            
        Returns:
            Tuple of (X, y) arrays for ML model
        """
        try:
            # Get feature columns (exclude timestamp and symbol)
            feature_cols = df.columns.difference(['timestamp', 'symbol'])
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(df) - lookback - prediction_horizon + 1):
                # Get sequence of past data
                sequence = df[feature_cols].iloc[i:i+lookback].values
                
                # Get future target
                target = df[target_column].iloc[i+lookback+prediction_horizon-1]
                
                sequences.append(sequence)
                targets.append(target)
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error preparing ML data: {str(e)}")
            raise
            
    def create_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature array
            y: Target array
            train_size: Proportion of data for training
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Calculate split index
            split_idx = int(len(X) * train_size)
            
            if shuffle:
                # Generate random indices
                indices = np.random.permutation(len(X))
                X = X[indices]
                y = y[indices]
            
            # Split data
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise 