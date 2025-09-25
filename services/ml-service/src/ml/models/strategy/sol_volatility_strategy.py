"""
Solana Volatility-Aware Trading Strategy

This module implements a specialized trading strategy for Solana (SOL) that is designed
to work well with its high volatility characteristics. The strategy combines:

1. Lorentzian classifier for base signals
2. Volatility-based position sizing
3. ATR-based stop loss and take profit levels
4. Momentum confirmation filters
5. Volatility breakout detection

Key features:
- Adapts to SOL's large price swings ($30-$100)
- Reduces false signals during choppy periods
- Captures momentum during trending markets
- Dynamic risk management based on current volatility
- Optimized for 5-minute timeframe

Usage:
```python
# Initialize strategy
strategy = SOLVolatilityStrategy(
    lookback_bars=20,
    prediction_bars=4,
    k_neighbors=15,
    volatility_factor=1.5
)

# Calculate signals
signals = strategy.calculate_signals(data)

# Access components
buy_signals = signals['buy_signals']
sell_signals = signals['sell_signals']
predictions = signals['predictions']
```
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple, Union

# Import base Lorentzian classifier
from .lorentzian_classifier import LorentzianANN

# Set up GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SOLVolatilityStrategy:
    """
    Solana Volatility-Aware Trading Strategy
    
    This strategy is specifically designed for SOL's high volatility characteristics,
    with large daily price movements and occasional extreme swings.
    
    Parameters
    ----------
    lookback_bars : int, optional
        Number of historical bars to consider (default: 20).
    prediction_bars : int, optional
        Number of bars into the future to predict (default: 4).
    k_neighbors : int, optional
        Number of nearest neighbors for Lorentzian classifier (default: 15).
    volatility_factor : float, optional
        Multiplier for ATR to determine stop loss/take profit (default: 1.5).
    momentum_period : int, optional
        Period for momentum calculation (default: 10).
    min_volatility : float, optional
        Minimum volatility threshold for trading (default: 0.005).
    max_volatility : float, optional
        Maximum volatility threshold for trading (default: 0.05).
    """
    
    def __init__(
        self,
        lookback_bars=20,
        prediction_bars=4,
        k_neighbors=15,
        volatility_factor=1.5,
        momentum_period=10,
        min_volatility=0.005,
        max_volatility=0.05,
    ):
        """Initialize the SOL Volatility Strategy"""
        # Store parameters
        self.lookback_bars = lookback_bars
        self.prediction_bars = prediction_bars
        self.k_neighbors = k_neighbors
        self.volatility_factor = volatility_factor
        self.momentum_period = momentum_period
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        
        # Initialize base Lorentzian classifier
        self.lorentzian = LorentzianANN(
            lookback_bars=lookback_bars,
            prediction_bars=prediction_bars,
            k_neighbors=k_neighbors,
            use_regime_filter=True,
            use_volatility_filter=True
        )
        
        # Initialize state
        self.is_fitted = False
        
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate trading signals for Solana based on volatility-aware strategy
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with buy_signals, sell_signals, and predictions
        """
        # Ensure we have a copy to avoid modifying the original
        df = data.copy()
        
        # Add required technical indicators
        df = self._add_indicators(df)
        
        # Prepare features for Lorentzian classifier
        features = self._prepare_features(df)
        
        # Get base signals from Lorentzian classifier
        if not self.is_fitted:
            # First time - fit the model
            prices = torch.tensor(df['close'].values, dtype=torch.float32)
            self.lorentzian.fit(features, prices)
            self.is_fitted = True
            
        # Get base predictions from Lorentzian
        lorentzian_signals = self.lorentzian.calculate_signals(df)
        base_predictions = lorentzian_signals['predictions']
        
        # Apply volatility filters
        filtered_signals = self._apply_volatility_filters(
            base_predictions, df
        )
        
        # Generate buy/sell signals
        buy_signals = torch.zeros_like(filtered_signals)
        sell_signals = torch.zeros_like(filtered_signals)
        
        # Buy when signal is positive and passes momentum filter
        buy_mask = (filtered_signals > 0) & (df['momentum'].values > 0)
        buy_signals[buy_mask] = 1.0
        
        # Sell when signal is negative and passes momentum filter
        sell_mask = (filtered_signals < 0) & (df['momentum'].values < 0)
        sell_signals[sell_mask] = 1.0
        
        # Return signals dictionary
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'predictions': filtered_signals
        }
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators needed for the strategy
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added indicators
        """
        # Calculate ATR (Average True Range) for volatility measurement
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate normalized ATR (as percentage of price)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Calculate momentum
        df['momentum'] = df['close'].pct_change(periods=self.momentum_period)
        
        # Calculate volatility (standard deviation of returns)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Calculate moving averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # Calculate trend strength
        df['trend_strength'] = ((df['close'] - df['open']) / df['close']).abs()
        
        # Volatility breakout indicator
        df['vol_ratio'] = df['atr_pct'] / df['atr_pct'].rolling(window=100).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare feature tensor for the Lorentzian classifier
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicators
            
        Returns
        -------
        torch.Tensor
            Feature tensor for model input
        """
        # Select features for the model
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'momentum', 'volatility', 
            'trend_strength', 'vol_ratio'
        ]
        
        # Extract features
        features = df[feature_columns].values
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32)
    
    def _apply_volatility_filters(self, predictions: torch.Tensor, df: pd.DataFrame) -> torch.Tensor:
        """
        Apply volatility-based filters to raw predictions
        
        Parameters
        ----------
        predictions : torch.Tensor
            Raw predictions from Lorentzian classifier
        df : pd.DataFrame
            DataFrame with indicators
            
        Returns
        -------
        torch.Tensor
            Filtered predictions
        """
        # Convert tensors to numpy for easier filtering
        preds = predictions.cpu().numpy()
        filtered = preds.copy()
        
        # Get indicator values
        volatility = df['volatility'].values
        atr_pct = df['atr_pct'].values
        vol_ratio = df['vol_ratio'].values
        trend_strength = df['trend_strength'].values
        
        # Filter 1: Ignore signals during extreme volatility
        extreme_volatility = volatility > self.max_volatility
        filtered[extreme_volatility] = 0
        
        # Filter 2: Ignore signals during very low volatility
        low_volatility = volatility < self.min_volatility
        filtered[low_volatility] = 0
        
        # Filter 3: Strengthen signals during volatility breakouts
        volatility_breakout = vol_ratio > 1.5
        filtered[volatility_breakout] = filtered[volatility_breakout] * 1.5
        
        # Filter 4: Scale signal strength by trend strength
        filtered = filtered * (1 + trend_strength)
        
        # Convert back to tensor
        return torch.tensor(filtered, dtype=torch.float32).to(device)
    
    def get_stop_loss_take_profit(self, entry_price: float, direction: str, 
                                  current_atr: float) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels based on ATR
        
        Parameters
        ----------
        entry_price : float
            Entry price for the trade
        direction : str
            Trade direction ('long' or 'short')
        current_atr : float
            Current ATR value
            
        Returns
        -------
        Tuple[float, float]
            (stop_loss_price, take_profit_price)
        """
        # Calculate ATR-based stop and target
        atr_multiple = current_atr * self.volatility_factor
        
        if direction.lower() == 'long':
            stop_loss = entry_price - atr_multiple
            take_profit = entry_price + (atr_multiple * 2)  # 2:1 reward-risk ratio
        else:  # short
            stop_loss = entry_price + atr_multiple
            take_profit = entry_price - (atr_multiple * 2)  # 2:1 reward-risk ratio
            
        return stop_loss, take_profit
    
    def get_position_size(self, capital: float, risk_per_trade: float, 
                         entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Parameters
        ----------
        capital : float
            Available capital
        risk_per_trade : float
            Percentage of capital to risk per trade (e.g., 0.01 for 1%)
        entry_price : float
            Entry price for the trade
        stop_loss : float
            Stop loss price
            
        Returns
        -------
        float
            Position size in units
        """
        # Calculate dollar risk
        dollar_risk = capital * risk_per_trade
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Calculate position size
        position_size = dollar_risk / risk_per_unit
        
        return position_size
    
    def save_model(self, file_path: str):
        """
        Save the model to disk
        
        Parameters
        ----------
        file_path : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Lorentzian model
        self.lorentzian.save_model(str(path) + "_lorentzian.pt")
        
        # Save strategy parameters
        params = {
            "lookback_bars": self.lookback_bars,
            "prediction_bars": self.prediction_bars,
            "k_neighbors": self.k_neighbors,
            "volatility_factor": self.volatility_factor,
            "momentum_period": self.momentum_period,
            "min_volatility": self.min_volatility,
            "max_volatility": self.max_volatility,
        }
        
        with open(str(path) + "_params.json", 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load_model(cls, file_path: str) -> 'SOLVolatilityStrategy':
        """
        Load the model from disk
        
        Parameters
        ----------
        file_path : str
            Path to load the model from
            
        Returns
        -------
        SOLVolatilityStrategy
            Loaded model
        """
        # Load parameters
        with open(str(file_path) + "_params.json", 'r') as f:
            params = json.load(f)
        
        # Create instance with loaded parameters
        instance = cls(**params)
        
        # Load Lorentzian model
        instance.lorentzian = LorentzianANN.load_model(str(file_path) + "_lorentzian.pt")
        instance.is_fitted = True
        
        return instance
