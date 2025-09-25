"""
Market Structure Analysis for High-Leverage HFT

Implements BOS (Break of Structure), CHoCH (Change of Character),
Order Block detection, and Fair Value Gap identification.

Designed for high-frequency trading with AMD ROCm GPU acceleration.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging


@dataclass
class MarketStructureConfig:
    """Configuration for market structure analysis"""
    bos_lookback: int = 20          # Bars to look back for structure
    choch_lookback: int = 10        # Bars to look back for character change
    order_block_lookback: int = 50  # Bars to look back for order blocks
    fvg_lookback: int = 10          # Bars to look back for fair value gaps
    min_structure_strength: float = 0.6  # Minimum confidence for structure
    volume_threshold: float = 1.5   # Volume multiplier for confirmation

class MarketStructureDetector:
    """Detects market structure patterns for HFT trading."""
    
    def __init__(self, config: Optional[MarketStructureConfig] = None):
        self.config = config or MarketStructureConfig()
        self.logger = logging.getLogger(__name__)
        
        # Check GPU availability
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.logger.info(
            f"Market Structure Detector using device: {self.device}"
        )
    
    def detect_bos(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Detect Break of Structure (BOS) points.
        
        BOS occurs when price breaks above previous highs or 
        below previous lows.
        """
        self.logger.info("Detecting Break of Structure (BOS)...")
        
        # Convert to tensors
        high = torch.tensor(df['high'].values, dtype=torch.float32).to(
            self.device
        )
        low = torch.tensor(df['low'].values, dtype=torch.float32).to(
            self.device
        )
        close = torch.tensor(df['close'].values, dtype=torch.float32).to(
            self.device
        )
        volume = torch.tensor(df['volume'].values, dtype=torch.float32).to(
            self.device
        )
        
        # Initialize signals
        bos_bullish = torch.zeros_like(close)
        bos_bearish = torch.zeros_like(close)
        bos_strength = torch.zeros_like(close)
        
        # Detect BOS for each bar
        for i in range(self.config.bos_lookback, len(close)):
            # Look back for previous structure
            lookback_high = high[i-self.config.bos_lookback:i]
            lookback_low = low[i-self.config.bos_lookback:i]
            
            # Find key levels
            resistance_level = torch.max(lookback_high)
            support_level = torch.min(lookback_low)
            
            # Check for bullish BOS (break above resistance)
            if close[i] > resistance_level:
                # Calculate strength based on volume and distance
                volume_ratio = volume[i] / torch.mean(volume[i-self.config.bos_lookback:i])
                distance_ratio = (close[i] - resistance_level) / resistance_level
                
                strength = min(1.0, (volume_ratio + distance_ratio * 10) / 2)
                
                if strength > self.config.min_structure_strength:
                    bos_bullish[i] = 1.0
                    bos_strength[i] = strength
            
            # Check for bearish BOS (break below support)
            elif close[i] < support_level:
                # Calculate strength based on volume and distance
                volume_ratio = volume[i] / torch.mean(volume[i-self.config.bos_lookback:i])
                distance_ratio = (support_level - close[i]) / support_level
                
                strength = min(1.0, (volume_ratio + distance_ratio * 10) / 2)
                
                if strength > self.config.min_structure_strength:
                    bos_bearish[i] = 1.0
                    bos_strength[i] = strength
        
        return {
            'bos_bullish': bos_bullish,
            'bos_bearish': bos_bearish,
            'bos_strength': bos_strength
        }
    
    def detect_choch(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Detect Change of Character (CHoCH) points.
        
        CHoCH occurs when a trend reverses and creates a new structure.
        """
        self.logger.info("Detecting Change of Character (CHoCH)...")
        
        # Convert to tensors
        high = torch.tensor(df['high'].values, dtype=torch.float32).to(self.device)
        low = torch.tensor(df['low'].values, dtype=torch.float32).to(self.device)
        close = torch.tensor(df['close'].values, dtype=torch.float32).to(self.device)
        volume = torch.tensor(df['volume'].values, dtype=torch.float32).to(self.device)
        
        # Initialize signals
        choch_bullish = torch.zeros_like(close)
        choch_bearish = torch.zeros_like(close)
        choch_strength = torch.zeros_like(close)
        
        # Detect CHoCH for each bar
        for i in range(self.config.choch_lookback, len(close)):
            # Look back for trend analysis
            lookback_close = close[i-self.config.choch_lookback:i]
            lookback_volume = volume[i-self.config.choch_lookback:i]
            
            # Calculate trend direction
            trend_start = lookback_close[0]
            trend_end = lookback_close[-1]
            trend_direction = 1 if trend_end > trend_start else -1
            
            # Check for bullish CHoCH (uptrend reversal to downtrend)
            if trend_direction == 1:  # Was uptrend
                # Look for reversal pattern
                recent_highs = high[i-5:i]
                recent_lows = low[i-5:i]
                
                # Check if we're making lower highs and lower lows
                if (torch.max(recent_highs[:-1]) > torch.max(recent_highs[1:]) and
                    torch.min(recent_lows[:-1]) > torch.min(recent_lows[1:])):
                    
                    # Calculate strength
                    volume_ratio = volume[i] / torch.mean(lookback_volume)
                    price_change = abs(close[i] - trend_end) / trend_end
                    
                    strength = min(1.0, (volume_ratio + price_change * 5) / 2)
                    
                    if strength > self.config.min_structure_strength:
                        choch_bearish[i] = 1.0
                        choch_strength[i] = strength
            
            # Check for bearish CHoCH (downtrend reversal to uptrend)
            elif trend_direction == -1:  # Was downtrend
                # Look for reversal pattern
                recent_highs = high[i-5:i]
                recent_lows = low[i-5:i]
                
                # Check if we're making higher highs and higher lows
                if (torch.max(recent_highs[1:]) > torch.max(recent_highs[:-1]) and
                    torch.min(recent_lows[1:]) > torch.min(recent_lows[:-1])):
                    
                    # Calculate strength
                    volume_ratio = volume[i] / torch.mean(lookback_volume)
                    price_change = abs(close[i] - trend_end) / trend_end
                    
                    strength = min(1.0, (volume_ratio + price_change * 5) / 2)
                    
                    if strength > self.config.min_structure_strength:
                        choch_bullish[i] = 1.0
                        choch_strength[i] = strength
        
        return {
            'choch_bullish': choch_bullish,
            'choch_bearish': choch_bearish,
            'choch_strength': choch_strength
        }
    
    def detect_order_blocks(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Detect Order Blocks (institutional order zones).
        
        Order blocks are areas where significant buying/selling occurred.
        """
        self.logger.info("Detecting Order Blocks...")
        
        # Convert to tensors
        high = torch.tensor(df['high'].values, dtype=torch.float32).to(self.device)
        low = torch.tensor(df['low'].values, dtype=torch.float32).to(self.device)
        close = torch.tensor(df['close'].values, dtype=torch.float32).to(self.device)
        volume = torch.tensor(df['volume'].values, dtype=torch.float32).to(self.device)
        
        # Initialize signals
        order_block_bullish = torch.zeros_like(close)
        order_block_bearish = torch.zeros_like(close)
        order_block_strength = torch.zeros_like(close)
        
        # Detect order blocks
        for i in range(self.config.order_block_lookback, len(close)):
            # Look back for order block patterns
            lookback_high = high[i-self.config.order_block_lookback:i]
            lookback_low = low[i-self.config.order_block_lookback:i]
            lookback_volume = volume[i-self.config.order_block_lookback:i]
            lookback_close = close[i-self.config.order_block_lookback:i]
            
            # Find high volume bars
            avg_volume = torch.mean(lookback_volume)
            high_volume_bars = lookback_volume > (avg_volume * self.config.volume_threshold)
            
            # Check for bullish order block (high volume down bar followed by reversal)
            for j in range(len(high_volume_bars) - 1):
                if (high_volume_bars[j] and 
                    lookback_close[j] < lookback_close[j-1] and  # Down bar
                    lookback_close[j+1] > lookback_close[j]):    # Reversal
                    
                    # Check if current price is near this order block
                    ob_high = lookback_high[j]
                    ob_low = lookback_low[j]
                    
                    if ob_low <= close[i] <= ob_high:
                        strength = min(1.0, lookback_volume[j] / avg_volume)
                        order_block_bullish[i] = 1.0
                        order_block_strength[i] = max(order_block_strength[i], strength)
            
            # Check for bearish order block (high volume up bar followed by reversal)
            for j in range(len(high_volume_bars) - 1):
                if (high_volume_bars[j] and 
                    lookback_close[j] > lookback_close[j-1] and  # Up bar
                    lookback_close[j+1] < lookback_close[j]):    # Reversal
                    
                    # Check if current price is near this order block
                    ob_high = lookback_high[j]
                    ob_low = lookback_low[j]
                    
                    if ob_low <= close[i] <= ob_high:
                        strength = min(1.0, lookback_volume[j] / avg_volume)
                        order_block_bearish[i] = 1.0
                        order_block_strength[i] = max(order_block_strength[i], strength)
        
        return {
            'order_block_bullish': order_block_bullish,
            'order_block_bearish': order_block_bearish,
            'order_block_strength': order_block_strength
        }
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Detect Fair Value Gaps (FVGs).
        
        FVGs are areas where price gaps and creates inefficiencies.
        """
        self.logger.info("Detecting Fair Value Gaps (FVGs)...")
        
        # Convert to tensors
        high = torch.tensor(df['high'].values, dtype=torch.float32).to(self.device)
        low = torch.tensor(df['low'].values, dtype=torch.float32).to(self.device)
        close = torch.tensor(df['close'].values, dtype=torch.float32).to(self.device)
        
        # Initialize signals
        fvg_bullish = torch.zeros_like(close)
        fvg_bearish = torch.zeros_like(close)
        fvg_strength = torch.zeros_like(close)
        
        # Detect FVGs
        for i in range(2, len(close)):
            # Bullish FVG: gap between previous low and current high
            if low[i] > high[i-2]:
                gap_size = low[i] - high[i-2]
                gap_ratio = gap_size / high[i-2]
                
                if gap_ratio > 0.001:  # Minimum gap size
                    strength = min(1.0, gap_ratio * 100)
                    fvg_bullish[i] = 1.0
                    fvg_strength[i] = strength
            
            # Bearish FVG: gap between previous high and current low
            elif high[i] < low[i-2]:
                gap_size = low[i-2] - high[i]
                gap_ratio = gap_size / low[i-2]
                
                if gap_ratio > 0.001:  # Minimum gap size
                    strength = min(1.0, gap_ratio * 100)
                    fvg_bearish[i] = 1.0
                    fvg_strength[i] = strength
        
        return {
            'fvg_bullish': fvg_bullish,
            'fvg_bearish': fvg_bearish,
            'fvg_strength': fvg_strength
        }
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate all market structure signals.
        
        Returns comprehensive market structure analysis.
        """
        self.logger.info("Calculating market structure signals...")
        
        # Detect all patterns
        bos_signals = self.detect_bos(df)
        choch_signals = self.detect_choch(df)
        order_block_signals = self.detect_order_blocks(df)
        fvg_signals = self.detect_fair_value_gaps(df)
        
        # Combine signals
        signals = {}
        signals.update(bos_signals)
        signals.update(choch_signals)
        signals.update(order_block_signals)
        signals.update(fvg_signals)
        
        # Create composite signals
        close = torch.tensor(df['close'].values, dtype=torch.float32).to(self.device)
        
        # Long signal: bullish BOS + bullish CHoCH + bullish order block
        long_signal = (
            bos_signals['bos_bullish'] * 0.4 +
            choch_signals['choch_bullish'] * 0.3 +
            order_block_signals['order_block_bullish'] * 0.2 +
            fvg_signals['fvg_bullish'] * 0.1
        )
        
        # Short signal: bearish BOS + bearish CHoCH + bearish order block
        short_signal = (
            bos_signals['bos_bearish'] * 0.4 +
            choch_signals['choch_bearish'] * 0.3 +
            order_block_signals['order_block_bearish'] * 0.2 +
            fvg_signals['fvg_bearish'] * 0.1
        )
        
        # Buy/sell signals (binary)
        buy_signals = (long_signal > 0.5).float()
        sell_signals = (short_signal > 0.5).float()
        
        signals.update({
            'long_signal': long_signal,
            'short_signal': short_signal,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        })
        
        self.logger.info(f"Market structure analysis complete - {len(df)} bars processed")
        
        return signals

def main():
    """Test market structure detection."""
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    # Generate sample price data
    base_price = 100.0
    prices = []
    for i in range(100):
        if i == 0:
            prices.append(base_price)
        else:
            change = np.random.normal(0, 0.5)
            prices.append(prices[-1] * (1 + change/100))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.2))/100) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.2))/100) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(1000, 10000) for _ in prices]
    })
    
    # Test detector
    detector = MarketStructureDetector()
    signals = detector.calculate_signals(df)
    
    print("✅ Market Structure Detection Test Complete!")
    print(f"📊 Processed {len(df)} bars")
    print(f"🎯 Long signals: {signals['buy_signals'].sum().item()}")
    print(f"🎯 Short signals: {signals['sell_signals'].sum().item()}")

if __name__ == "__main__":
    main() 