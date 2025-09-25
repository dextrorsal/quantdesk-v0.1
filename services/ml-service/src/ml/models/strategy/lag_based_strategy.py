"""
Lag-Based Trading Strategy Implementation

This module implements a systematic approach to trading meme coins and DeFi tokens 
by identifying and exploiting lagging price movements after significant moves in 
major assets (SOL, BTC, ETH).

Core Concept: When SOL or BTC makes a significant move (2-5%), correlated smaller 
assets often follow with a delay, creating trading opportunities.

Features:
- Leader-follower correlation analysis
- Dynamic threshold optimization
- Lag time measurement and analysis
- Volume confirmation
- Risk management with position sizing
- Real-time signal generation
- Backtesting capabilities

Usage:
    from src.ml.models.strategy.lag_based_strategy import LagBasedStrategy
    
    # Initialize strategy
    strategy = LagBasedStrategy(
        leader_assets=['BTC', 'ETH', 'SOL'],
        follower_assets=['WIF', 'FARTCOIN', 'POPCAT'],
        threshold=1.5,
        max_lag_minutes=60
    )
    
    # Generate signals
    signals = strategy.generate_signals(leader_data, follower_data)
    
    # Run backtest
    results = strategy.backtest(historical_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Trading direction enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class LagSignal:
    """Represents a lag-based trading signal."""
    timestamp: pd.Timestamp
    leader_asset: str
    follower_asset: str
    leader_move_pct: float
    follower_move_pct: float
    lag_time_minutes: int
    correlation: float
    volume_ratio: float
    direction: Direction
    confidence: float
    threshold_used: float


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: Direction
    pnl: float
    pnl_pct: float
    stop_loss: float
    take_profit: float
    confidence: float
    leader_asset: str
    follower_asset: str
    lag_time_minutes: int


@dataclass
class StrategyConfig:
    """Configuration for lag-based trading strategy."""
    
    # Asset configuration
    leader_assets: List[str] = None  # ['BTC', 'ETH', 'SOL']
    follower_assets: List[str] = None  # ['WIF', 'FARTCOIN', 'POPCAT']
    
    # Signal parameters
    threshold: float = 1.5  # Minimum leader move percentage
    max_lag_minutes: int = 60  # Maximum lag time to consider
    min_correlation: float = 0.3  # Minimum correlation coefficient
    volume_confirmation: bool = True
    min_volume_ratio: float = 1.2  # Minimum volume increase for confirmation
    
    # Risk management
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 3  # Maximum concurrent positions
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    # Analysis parameters
    correlation_window: str = '1H'  # Rolling correlation window
    move_calculation_window: str = '5min'  # Window for move calculation
    volume_window: str = '15min'  # Window for volume analysis
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.leader_assets is None:
            self.leader_assets = ['BTC', 'ETH', 'SOL']
        if self.follower_assets is None:
            self.follower_assets = ['WIF', 'FARTCOIN', 'POPCAT']


class LagBasedStrategy:
    """
    Implements lag-based trading strategy for meme coins and DeFi tokens.
    Supports both all-followers and grouped-followers modes.
    If follower_data is a dict of dicts (grouped), only check specified followers for each leader.
    If follower_data is a flat dict, check all followers for all leaders.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the lag-based trading strategy.
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.signals: List[LagSignal] = []
        self.trades: List[Trade] = []
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info(f"Initialized LagBasedStrategy with {len(config.leader_assets)} "
                   f"leaders and {len(config.follower_assets)} followers")
    
    def analyze_move_distribution(self, data: pd.DataFrame, asset: str, 
                                timeframes: List[int] = [5, 15, 30, 60]) -> Dict[str, Any]:
        """
        Analyze price movement distributions across different timeframes.
        
        Args:
            data: DataFrame with OHLCV data
            asset: Asset name for analysis
            timeframes: List of timeframes in minutes to analyze
            
        Returns:
            Dictionary with analysis results for each timeframe
        """
        results = {}
        
        for tf in timeframes:
            # Resample to desired timeframe
            ohlc = data.resample(f'{tf}min').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate percentage moves
            moves = (ohlc['close'] - ohlc['open']) / ohlc['open'] * 100
            abs_moves = abs(moves)
            
            # Calculate percentiles
            percentiles = [50, 75, 90, 95, 99]
            result = {}
            
            for p in percentiles:
                pct_value = np.percentile(abs_moves, p)
                result[f'{p}th_percentile'] = pct_value
            
            results[f'{tf}min'] = result
            
            logger.info(f"{asset} {tf}min - 90th percentile: {result['90th_percentile']:.2f}%, "
                       f"95th percentile: {result['95th_percentile']:.2f}%")
        
        return results
    
    def measure_lag_times(self, leader_data: pd.DataFrame, follower_data: pd.DataFrame,
                         threshold: float = None) -> Tuple[List[Optional[int]], List[int]]:
        """
        Measure lag times between leader moves and follower responses.
        
        Args:
            leader_data: Leader asset OHLCV data
            follower_data: Follower asset OHLCV data
            threshold: Move threshold (uses config threshold if None)
            
        Returns:
            Tuple of (all_lag_times, valid_lag_times)
        """
        if threshold is None:
            threshold = self.config.threshold
        
        # Resample to consistent timeframe
        timeframe = self.config.move_calculation_window
        leader_ohlc = leader_data.resample(timeframe).agg({
            'open': 'first', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        follower_ohlc = follower_data.resample(timeframe).agg({
            'open': 'first', 'close': 'last', 'volume': 'sum'  
        }).dropna()
        
        # Calculate moves
        leader_moves = (leader_ohlc['close'] - leader_ohlc['open']) / leader_ohlc['open'] * 100
        follower_moves = (follower_ohlc['close'] - follower_ohlc['open']) / follower_ohlc['open'] * 100
        
        lag_times = []
        
        for timestamp, move_size in leader_moves.items():
            if abs(move_size) > threshold:
                # Look for follower response in next N periods
                response_found = False
                max_look_ahead = self.config.max_lag_minutes // 5  # Assuming 5min bars
                
                for i in range(1, max_look_ahead + 1):
                    future_time = timestamp + pd.Timedelta(minutes=i*5)
                    if future_time in follower_moves.index:
                        follower_move = follower_moves[future_time]
                        # Check if follower moved in same direction with >50% of leader move
                        if (np.sign(move_size) == np.sign(follower_move) and 
                            abs(follower_move) > abs(move_size) * 0.5):
                            lag_times.append(i * 5)  # Convert to minutes
                            response_found = True
                            break
                
                if not response_found:
                    lag_times.append(None)  # No response found
        
        # Analyze lag distribution
        valid_lags = [x for x in lag_times if x is not None]
        
        if valid_lags:
            logger.info(f"Lag Analysis - Total moves: {len(lag_times)}, "
                       f"Responses: {len(valid_lags)}, "
                       f"Response rate: {len(valid_lags)/len(lag_times)*100:.1f}%, "
                       f"Median lag: {np.median(valid_lags):.0f} minutes")
        
        return lag_times, valid_lags
    
    def calculate_correlation(self, leader_data: pd.DataFrame, 
                            follower_data: pd.DataFrame) -> float:
        """
        Calculate rolling correlation between leader and follower assets.
        
        Args:
            leader_data: Leader asset OHLCV data
            follower_data: Follower asset OHLCV data
            
        Returns:
            Correlation coefficient
        """
        # Align data by timestamp
        leader_returns = leader_data['close'].pct_change().dropna()
        follower_returns = follower_data['close'].pct_change().dropna()
        
        # Align timestamps
        aligned_data = pd.concat([leader_returns, follower_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        # Calculate rolling correlation
        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def check_volume_confirmation(self, follower_data: pd.DataFrame, 
                                signal_time: pd.Timestamp) -> Tuple[bool, float]:
        """
        Check if volume confirms the signal.
        
        Args:
            follower_data: Follower asset OHLCV data
            signal_time: Time of the signal
            
        Returns:
            Tuple of (volume_confirmed, volume_ratio)
        """
        if not self.config.volume_confirmation:
            return True, 1.0
        
        # Calculate average volume in the window before signal
        window = self.config.volume_window
        before_signal = follower_data.loc[:signal_time].tail(20)  # Last 20 periods
        avg_volume = before_signal['volume'].mean()
        
        # Get current volume
        current_volume = follower_data.loc[signal_time, 'volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        volume_confirmed = volume_ratio >= self.config.min_volume_ratio
        
        return volume_confirmed, volume_ratio
    
    def generate_signals(self, leader_data: Dict[str, pd.DataFrame], 
                        follower_data: Dict[str, pd.DataFrame]) -> List[LagSignal]:
        """
        Generate trading signals based on lag analysis.
        Supports both grouped and all-followers modes.
        Args:
            leader_data: Dict of leader asset dataframes
            follower_data: Dict of follower asset dataframes (all-followers) or
                          Dict[leader, Dict[follower, df]] (grouped-followers)
        Returns:
            List of generated signals
        """
        signals = []
        # Detect grouped mode by type
        if isinstance(follower_data, dict) and all(isinstance(v, dict) for v in follower_data.values()):
            # Grouped mode: follower_data = {leader: {follower: df, ...}, ...}
            for leader_asset, leader_df in leader_data.items():
                if leader_asset not in follower_data:
                    continue
                for follower_asset, follower_df in follower_data[leader_asset].items():
                    correlation = self.calculate_correlation(leader_df, follower_df)
                    if abs(correlation) < self.config.min_correlation:
                        continue
                    # Resample to consistent timeframe
                    timeframe = self.config.move_calculation_window
                    leader_ohlc = leader_df.resample(timeframe).agg({
                        'open': 'first', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                    follower_ohlc = follower_df.resample(timeframe).agg({
                        'open': 'first', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                    leader_moves = (leader_ohlc['close'] - leader_ohlc['open']) / leader_ohlc['open'] * 100
                    follower_moves = (follower_ohlc['close'] - follower_ohlc['open']) / follower_ohlc['open'] * 100
                    for timestamp in leader_moves.index:
                        leader_move = leader_moves[timestamp]
                        if abs(leader_move) > self.config.threshold:
                            if timestamp in follower_moves.index:
                                follower_move = follower_moves[timestamp]
                                if abs(follower_move) < abs(leader_move) * 0.5:
                                    volume_confirmed, volume_ratio = self.check_volume_confirmation(
                                        follower_df, timestamp)
                                    if volume_confirmed:
                                        confidence = min(0.9, abs(correlation) * 0.7 + 
                                                       min(abs(leader_move) / 10, 0.3))
                                        signal = LagSignal(
                                            timestamp=timestamp,
                                            leader_asset=leader_asset,
                                            follower_asset=follower_asset,
                                            leader_move_pct=leader_move,
                                            follower_move_pct=follower_move,
                                            lag_time_minutes=0,
                                            correlation=correlation,
                                            volume_ratio=volume_ratio,
                                            direction=Direction.LONG if leader_move > 0 else Direction.SHORT,
                                            confidence=confidence,
                                            threshold_used=self.config.threshold
                                        )
                                        signals.append(signal)
        else:
            # All-followers mode: original logic
            for leader_asset in self.config.leader_assets:
                if leader_asset not in leader_data:
                    continue
                for follower_asset in self.config.follower_assets:
                    if follower_asset not in follower_data:
                        continue
                    leader_df = leader_data[leader_asset]
                    follower_df = follower_data[follower_asset]
                    correlation = self.calculate_correlation(leader_df, follower_df)
                    if abs(correlation) < self.config.min_correlation:
                        continue
                    timeframe = self.config.move_calculation_window
                    leader_ohlc = leader_df.resample(timeframe).agg({
                        'open': 'first', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                    follower_ohlc = follower_df.resample(timeframe).agg({
                        'open': 'first', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                    leader_moves = (leader_ohlc['close'] - leader_ohlc['open']) / leader_ohlc['open'] * 100
                    follower_moves = (follower_ohlc['close'] - follower_ohlc['open']) / follower_ohlc['open'] * 100
                    for timestamp in leader_moves.index:
                        leader_move = leader_moves[timestamp]
                        if abs(leader_move) > self.config.threshold:
                            if timestamp in follower_moves.index:
                                follower_move = follower_moves[timestamp]
                                if abs(follower_move) < abs(leader_move) * 0.5:
                                    volume_confirmed, volume_ratio = self.check_volume_confirmation(
                                        follower_df, timestamp)
                                    if volume_confirmed:
                                        confidence = min(0.9, abs(correlation) * 0.7 + 
                                                       min(abs(leader_move) / 10, 0.3))
                                        signal = LagSignal(
                                            timestamp=timestamp,
                                            leader_asset=leader_asset,
                                            follower_asset=follower_asset,
                                            leader_move_pct=leader_move,
                                            follower_move_pct=follower_move,
                                            lag_time_minutes=0,
                                            correlation=correlation,
                                            volume_ratio=volume_ratio,
                                            direction=Direction.LONG if leader_move > 0 else Direction.SHORT,
                                            confidence=confidence,
                                            threshold_used=self.config.threshold
                                        )
                                        signals.append(signal)
        self.signals.extend(signals)
        logger.info(f"Generated {len(signals)} new signals")
        return signals
    
    def optimize_thresholds(self, leader_data: Dict[str, pd.DataFrame],
                          follower_data: Dict[str, pd.DataFrame],
                          thresholds: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]) -> Dict[str, Any]:
        """
        Optimize thresholds for signal quality.
        
        Args:
            leader_data: Dictionary of leader asset dataframes
            follower_data: Dictionary of follower asset dataframes
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        results = {}
        
        for threshold in thresholds:
            logger.info(f"Testing threshold: {threshold}%")
            
            total_signals = 0
            successful_signals = 0
            all_lag_times = []
            
            for leader_asset in self.config.leader_assets:
                if leader_asset not in leader_data:
                    continue
                    
                for follower_asset in self.config.follower_assets:
                    if follower_asset not in follower_data:
                        continue
                    
                    # Get lag times for this threshold
                    lag_times, valid_lags = self.measure_lag_times(
                        leader_data[leader_asset], 
                        follower_data[follower_asset], 
                        threshold
                    )
                    
                    total_signals += len(lag_times)
                    successful_signals += len(valid_lags)
                    all_lag_times.extend(valid_lags)
            
            hit_rate = successful_signals / total_signals if total_signals > 0 else 0
            median_lag = np.median(all_lag_times) if all_lag_times else None
            
            results[threshold] = {
                'total_signals': total_signals,
                'successful_signals': successful_signals,
                'hit_rate': hit_rate,
                'median_lag': median_lag,
                'signal_frequency_per_day': total_signals / 30  # Assuming 30 days of data
            }
            
            logger.info(f"  Hit rate: {hit_rate*100:.1f}%, "
                       f"Total signals: {total_signals}, "
                       f"Median lag: {median_lag:.0f} minutes" if median_lag else "N/A")
        
        return results
    
    def backtest(self, historical_data: Dict[str, pd.DataFrame], 
                initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            historical_data: Dictionary of asset dataframes
            initial_capital: Initial capital for backtest
            
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        open_positions = []
        trades = []
        equity_curve = [initial_capital]
        
        # Generate signals
        leader_data = {k: v for k, v in historical_data.items() 
                      if k in self.config.leader_assets}
        follower_data = {k: v for k, v in historical_data.items() 
                        if k in self.config.follower_assets}
        
        signals = self.generate_signals(leader_data, follower_data)
        
        # Sort signals by timestamp
        signals.sort(key=lambda x: x.timestamp)
        
        # Process signals
        for signal in signals:
            if len(open_positions) >= self.config.max_positions:
                continue
            
            # Calculate position size
            position_size = capital * self.config.risk_per_trade
            
            # Get current price
            follower_df = follower_data[signal.follower_asset]
            current_price = follower_df.loc[signal.timestamp, 'close']
            
            # Calculate stop loss and take profit
            if signal.direction == Direction.LONG:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                take_profit = current_price * (1 + self.config.take_profit_pct)
            else:
                stop_loss = current_price * (1 + self.config.stop_loss_pct)
                take_profit = current_price * (1 - self.config.take_profit_pct)
            
            # Open position
            position = {
                'signal': signal,
                'entry_time': signal.timestamp,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'direction': signal.direction
            }
            
            open_positions.append(position)
        
        # Process exits (simplified - would need more sophisticated exit logic)
        # This is a basic implementation - you'd want to add proper exit logic
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_capital': equity_curve[-1] if equity_curve else initial_capital,
            'return_pct': (equity_curve[-1] - initial_capital) / initial_capital * 100 if equity_curve else 0,
            'equity_curve': equity_curve,
            'signals_generated': len(signals)
        }
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """
        Plot analysis results.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.signals:
            logger.warning("No signals to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Signal distribution by asset pair
        asset_pairs = [f"{s.leader_asset}-{s.follower_asset}" for s in self.signals]
        pair_counts = pd.Series(asset_pairs).value_counts()
        pair_counts.plot(kind='bar', ax=axes[0, 0], title='Signals by Asset Pair')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Leader move distribution
        leader_moves = [s.leader_move_pct for s in self.signals]
        axes[0, 1].hist(leader_moves, bins=30, alpha=0.7)
        axes[0, 1].set_title('Leader Move Distribution')
        axes[0, 1].set_xlabel('Move Percentage')
        
        # Plot 3: Confidence vs Correlation
        confidences = [s.confidence for s in self.signals]
        correlations = [s.correlation for s in self.signals]
        axes[1, 0].scatter(correlations, confidences, alpha=0.6)
        axes[1, 0].set_title('Confidence vs Correlation')
        axes[1, 0].set_xlabel('Correlation')
        axes[1, 0].set_ylabel('Confidence')
        
        # Plot 4: Volume ratio distribution
        volume_ratios = [s.volume_ratio for s in self.signals]
        axes[1, 1].hist(volume_ratios, bins=30, alpha=0.7)
        axes[1, 1].set_title('Volume Ratio Distribution')
        axes[1, 1].set_xlabel('Volume Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Analysis plots saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print strategy summary."""
        if not self.signals:
            logger.info("No signals generated yet")
            return
        
        logger.info("=== Lag-Based Strategy Summary ===")
        logger.info(f"Total signals generated: {len(self.signals)}")
        logger.info(f"Asset pairs: {len(set([f'{s.leader_asset}-{s.follower_asset}' for s in self.signals]))}")
        
        # Signal statistics
        confidences = [s.confidence for s in self.signals]
        correlations = [s.correlation for s in self.signals]
        leader_moves = [s.leader_move_pct for s in self.signals]
        
        logger.info(f"Average confidence: {np.mean(confidences):.3f}")
        logger.info(f"Average correlation: {np.mean(correlations):.3f}")
        logger.info(f"Average leader move: {np.mean(leader_moves):.2f}%")
        
        # Direction breakdown
        long_signals = len([s for s in self.signals if s.direction == Direction.LONG])
        short_signals = len([s for s in self.signals if s.direction == Direction.SHORT])
        logger.info(f"Long signals: {long_signals}, Short signals: {short_signals}")
        
        if self.trades:
            logger.info(f"Total trades: {len(self.trades)}")
            win_rate = len([t for t in self.trades if t.pnl > 0]) / len(self.trades)
            logger.info(f"Win rate: {win_rate:.1%}")


# Example usage and testing functions
def create_sample_data() -> Dict[str, pd.DataFrame]:
    """Create sample data for testing."""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='5min')
    
    # Create sample leader data (BTC)
    np.random.seed(42)
    btc_returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    btc_prices = 50000 * np.exp(np.cumsum(btc_returns))
    
    btc_data = pd.DataFrame({
        'open': btc_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': btc_prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
        'low': btc_prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
        'close': btc_prices,
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    # Create correlated follower data (WIF)
    wif_returns = btc_returns * 0.7 + np.random.normal(0, 0.03, len(dates))  # 70% correlation
    wif_prices = 2.0 * np.exp(np.cumsum(wif_returns))
    
    wif_data = pd.DataFrame({
        'open': wif_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': wif_prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
        'low': wif_prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
        'close': wif_prices,
        'volume': np.random.uniform(50, 500, len(dates))
    }, index=dates)
    
    return {
        'BTC': btc_data,
        'WIF': wif_data
    }


def test_strategy():
    """Test the lag-based strategy with sample data."""
    # Create sample data
    data = create_sample_data()
    
    # Initialize strategy
    config = StrategyConfig(
        leader_assets=['BTC'],
        follower_assets=['WIF'],
        threshold=1.0,
        max_lag_minutes=30
    )
    
    strategy = LagBasedStrategy(config)
    
    # Run analysis
    print("=== Move Distribution Analysis ===")
    btc_moves = strategy.analyze_move_distribution(data['BTC'], 'BTC')
    
    print("\n=== Lag Time Analysis ===")
    lag_times, valid_lags = strategy.measure_lag_times(data['BTC'], data['WIF'])
    
    print("\n=== Threshold Optimization ===")
    optimization_results = strategy.optimize_thresholds(data, data)
    
    print("\n=== Signal Generation ===")
    signals = strategy.generate_signals(data, data)
    
    print("\n=== Strategy Summary ===")
    strategy.print_summary()
    
    return strategy, data


if __name__ == "__main__":
    # Run test
    strategy, data = test_strategy() 