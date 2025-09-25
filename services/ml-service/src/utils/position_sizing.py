"""
Position Sizing Utilities

This module provides functions for calculating appropriate position sizes
based on account balance, risk parameters, and market conditions.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

@dataclass
class MarketConditions:
    volume_ratio: float  # Current volume compared to average
    sentiment_score: float  # -1 to 1 scale
    is_bullish: bool

class PositionSizer:
    def __init__(self, portfolio_value: float):
        """
        Initialize PositionSizer
        
        Args:
            portfolio_value (float): Total portfolio value
        """
        self.portfolio_value = portfolio_value
        self.leverage_allocations = {
            range(50, 101): (0.01, 0.10),  # 1-10% for 50-100x
            range(20, 50): (0.10, 0.20),   # 10-20% for 20-50x
            range(1, 20): (0.20, 1.0)      # 20%+ for under 20x
        }
    
    def calculate_position_size(
        self, 
        leverage: int,
        market_conditions: MarketConditions
    ) -> Tuple[float, float]:
        """
        Calculate position size based on leverage and market conditions
        
        Args:
            leverage (int): Intended leverage
            market_conditions (MarketConditions): Current market conditions
            
        Returns:
            Tuple[float, float]: (min_size, max_size) in terms of portfolio percentage
        """
        # Get base allocation range for leverage
        base_range = self._get_leverage_allocation(leverage)
        if not base_range:
            raise ValueError(f"Unsupported leverage: {leverage}")
        
        min_size, max_size = base_range
        
        # Adjust based on market conditions
        adjustment = self._calculate_condition_adjustment(market_conditions)
        
        # Apply adjustment while keeping within reasonable bounds
        adjusted_min = max(0.01, min(0.5, min_size * adjustment))
        adjusted_max = max(0.01, min(1.0, max_size * adjustment))
        
        return adjusted_min, adjusted_max
    
    def _get_leverage_allocation(self, leverage: int) -> Optional[Tuple[float, float]]:
        """Get base allocation range for given leverage"""
        for leverage_range, allocation in self.leverage_allocations.items():
            if leverage in leverage_range:
                return allocation
        return None
    
    def _calculate_condition_adjustment(self, conditions: MarketConditions) -> float:
        """
        Calculate position size adjustment based on market conditions
        Returns a multiplier (0.5 to 1.5)
        """
        # Start with base multiplier
        multiplier = 1.0
        
        # Volume impact (-0.25 to +0.25)
        volume_impact = (conditions.volume_ratio - 1) * 0.25
        multiplier += volume_impact
        
        # Sentiment impact (-0.15 to +0.15)
        sentiment_impact = conditions.sentiment_score * 0.15
        multiplier += sentiment_impact
        
        # Trend impact (+0.1 if bullish)
        if conditions.is_bullish:
            multiplier += 0.1
            
        # Ensure multiplier stays within reasonable bounds
        return max(0.5, min(1.5, multiplier))
    
    def calculate_scale_in_levels(
        self,
        entry_price: float,
        position_size: float,
        max_loss_percentage: float = 0.5  # 50% loss
    ) -> Dict[float, float]:
        """
        Calculate scale-in levels based on position loss
        
        Args:
            entry_price (float): Initial entry price
            position_size (float): Initial position size in portfolio percentage
            max_loss_percentage (float): Maximum loss percentage to scale in
            
        Returns:
            Dict[float, float]: Price levels mapped to additional position sizes
        """
        scale_levels = {}
        
        # Calculate 3 scale-in levels
        for i in range(3):
            loss_pct = (i + 1) * (max_loss_percentage / 3)
            price_level = entry_price * (1 - loss_pct)
            
            # Increase position size at each level
            additional_size = position_size * (1 + (i * 0.5))
            
            scale_levels[price_level] = additional_size
            
        return scale_levels 

def calculate_position_size(
    capital: float,
    risk_per_trade: float = 0.02,  # 2% risk per trade
    stop_distance: Optional[float] = None,
    volatility_factor: Optional[float] = None,
    min_position: float = 0.01,  # Minimum position size
    max_position: float = 1.0    # Maximum position size as fraction of capital
) -> float:
    """
    Calculate position size based on capital and risk parameters.
    
    Args:
        capital: Current account capital
        risk_per_trade: Maximum risk per trade as decimal (default 0.02 = 2%)
        stop_distance: Distance to stop loss in price units
        volatility_factor: Optional volatility scaling factor
        min_position: Minimum position size allowed
        max_position: Maximum position size as fraction of capital
        
    Returns:
        Position size in base currency units
    """
    # Calculate risk amount in currency
    risk_amount = capital * risk_per_trade
    
    # Adjust for volatility if provided
    if volatility_factor is not None:
        risk_amount = risk_amount * (1 / volatility_factor)
    
    # Calculate position size based on stop distance
    if stop_distance is not None and stop_distance > 0:
        position_size = risk_amount / stop_distance
    else:
        # Default to simple percentage of capital if no stop distance
        position_size = capital * risk_per_trade
    
    # Apply min/max constraints
    position_size = np.clip(
        position_size,
        capital * min_position,
        capital * max_position
    )
    
    return float(position_size)

def calculate_kelly_position(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    max_kelly: float = 0.2  # Cap Kelly fraction at 20%
) -> float:
    """
    Calculate position size using the Kelly Criterion.
    
    Args:
        win_rate: Historical win rate as decimal
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
        capital: Current account capital
        max_kelly: Maximum Kelly fraction to use
        
    Returns:
        Position size based on Kelly Criterion
    """
    try:
        # Kelly formula: f = (bp - q) / b
        # where: b = win/loss ratio, p = win probability, q = loss probability
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Cap the Kelly fraction
        kelly = min(kelly, max_kelly)
        
        # Ensure non-negative
        kelly = max(0, kelly)
        
        return capital * kelly
        
    except ZeroDivisionError:
        return 0.0
    except Exception as e:
        print(f"Error calculating Kelly position size: {str(e)}")
        return 0.0

def adjust_for_correlation(
    base_size: float,
    correlation: float,
    max_reduction: float = 0.5  # Maximum position reduction
) -> float:
    """
    Adjust position size based on correlation with existing positions.
    
    Args:
        base_size: Original calculated position size
        correlation: Correlation coefficient (-1 to 1)
        max_reduction: Maximum position size reduction
        
    Returns:
        Adjusted position size
    """
    try:
        # Convert correlation to positive scale
        pos_corr = abs(correlation)
        
        # Calculate reduction factor (higher correlation = more reduction)
        reduction = pos_corr * max_reduction
        
        # Apply reduction
        adjusted_size = base_size * (1 - reduction)
        
        return max(0, adjusted_size)
        
    except Exception as e:
        print(f"Error adjusting position for correlation: {str(e)}")
        return base_size 